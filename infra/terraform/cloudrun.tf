# ── Artifact Registry — stores our Docker images ─────────────────────────────
resource "google_artifact_registry_repository" "main" {
  location      = var.region
  repository_id = "malaysia-inflation-intel"
  format        = "DOCKER"
  description   = "Docker images for Malaysia Inflation Intelligence"
}

# ── FastAPI prediction service ────────────────────────────────────────────────
resource "google_cloud_run_v2_service" "api" {
  name                = "inflation-api"
  location            = var.region
  deletion_protection = false

  template {
    service_account = google_service_account.vertex.email

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/malaysia-inflation-intel/inflation-api:latest"

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "MODEL_GCS_PATH"
        value = "models/cpi_forecast_20260619_194024.joblib"
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
    service_account = google_service_account.ingestion.email

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
      }
    }

    scaling {
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