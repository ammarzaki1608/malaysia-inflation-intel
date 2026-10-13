# ── Artifact Registry image path for the pipeline job ────────────────────────
# (reuses the same repository you already created in cloudrun.tf)

# ── Cloud Run Job — runs the ingestion + dbt pipeline on trigger ─────────────
resource "google_cloud_run_v2_job" "data_pipeline" {
  name                = "inflation-data-pipeline"
  location            = var.region
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.ingestion.email

      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/malaysia-inflation-intel/data-pipeline:latest"

        resources {
          limits = {
            cpu    = "2"
            memory = "2Gi"
          }
        }
      }

      max_retries = 1
      timeout     = "1800s"
    }
  }

  # CI (.github/workflows/pipeline-image.yml) pins the job to each new image
  # by commit SHA; don't let `terraform apply` revert it to :latest.
  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image]
  }
}

# ── Service account specifically for triggering the job ──────────────────────
# Cloud Scheduler needs its own identity to invoke the Cloud Run Job
resource "google_service_account" "scheduler" {
  account_id   = "pipeline-scheduler-sa"
  display_name = "Cloud Scheduler - Pipeline Trigger"
}

resource "google_project_iam_member" "scheduler_run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.scheduler.email}"
}

# ── Cloud Scheduler — triggers the job on the 3rd of every month ─────────────
resource "google_cloud_scheduler_job" "monthly_pipeline_trigger" {
  name      = "monthly-data-pipeline-trigger"
  region    = var.region
  schedule  = "0 3 3 * *" # 3:00 AM UTC, on the 3rd day of every month
  time_zone = "UTC"

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${google_cloud_run_v2_job.data_pipeline.name}:run"

    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }
}