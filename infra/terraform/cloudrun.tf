# ── Artifact Registry — stores our Docker images ─────────────────────────────
resource "google_artifact_registry_repository" "main" {
  location      = var.region
  repository_id = "malaysia-inflation-intel"
  format        = "DOCKER"
  description   = "Docker images for Malaysia Inflation Intelligence"

  # Every deploy pushes new images, and old ones were piling up (36 GB, billed
  # monthly). Delete versions older than 7 days, but always keep the 5 newest
  # of each image so a rollback is still possible. Keep rules win over delete.
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "delete-older-than-7-days"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "604800s"
    }
  }
  cleanup_policies {
    id     = "keep-5-newest"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }
}

# ── Evaluation log token ──────────────────────────────────────────────────────
# The API logs an LLM call only when the caller sends this token, which only the
# evaluation workflow has. Copy it into the GitHub secret EVAL_LOG_TOKEN with:
#   terraform output -raw eval_log_token
resource "random_password" "eval_log_token" {
  length  = 40
  special = false
}

resource "google_secret_manager_secret" "eval_log_token" {
  secret_id = "eval-log-token"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "eval_log_token" {
  secret      = google_secret_manager_secret.eval_log_token.id
  secret_data = random_password.eval_log_token.result
}

resource "google_secret_manager_secret_iam_member" "api_eval_log_token" {
  secret_id = google_secret_manager_secret.eval_log_token.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.vertex.email}"
}

output "eval_log_token" {
  value     = random_password.eval_log_token.result
  sensitive = true
}

# ── FastAPI prediction service ────────────────────────────────────────────────
resource "google_cloud_run_v2_service" "api" {
  name                = "inflation-api"
  location            = var.region
  deletion_protection = false

  depends_on = [
    google_secret_manager_secret_version.eval_log_token,
    google_secret_manager_secret_iam_member.api_eval_log_token,
  ]

  template {
    service_account = google_service_account.vertex.email

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/malaysia-inflation-intel/inflation-api:latest"

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      # Calls carrying this token (the evaluation run) are logged; nothing else is.
      env {
        name = "EVAL_LOG_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.eval_log_token.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = "gemini-api-key"
            version = "latest"
          }
        }
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        # Bill CPU only while a request is being handled. The forecast table is
        # re-checked lazily on requests, so nothing needs CPU between them.
        cpu_idle          = true
        startup_cpu_boost = true
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 30
        period_seconds        = 10
        failure_threshold     = 5
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
  }

  # GitHub Actions deploys with gcloud, which stamps client/client_version and
  # writes service-level scaling defaults (0). Ignore that noise so `terraform
  # plan` only shows real changes. Instance limits live in template.scaling,
  # which Terraform still manages.
  lifecycle {
    ignore_changes = [client, client_version, scaling]
  }
}

# ── Streamlit app ─────────────────────────────────────────────────────────────
resource "google_cloud_run_v2_service" "streamlit" {
  name                = "inflation-app"
  location            = var.region
  deletion_protection = false

  template {
    service_account = google_service_account.streamlit.email

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/malaysia-inflation-intel/inflation-app:latest"

      env {
        name  = "API_URL"
        value = "https://inflation-api-lxxpkjdxja-as.a.run.app"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }
    }

    scaling {
      # Scale to zero. A warm instance (min 1, CPU always on) cost about RM10 a day,
      # most of the bill; the price is a cold start of a few seconds after idle.
      min_instance_count = 0
      max_instance_count = 2
    }
  }

  # GitHub Actions deploys with gcloud, which stamps client/client_version and
  # writes service-level scaling defaults (0). Ignore that noise so `terraform
  # plan` only shows real changes. Instance limits live in template.scaling,
  # which Terraform still manages.
  lifecycle {
    ignore_changes = [client, client_version, scaling]
  }
}

# ── Allow public access to both services ─────────────────────────────────────
resource "google_cloud_run_v2_service_iam_member" "api_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "streamlit_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.streamlit.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}