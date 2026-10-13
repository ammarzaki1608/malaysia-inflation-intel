# ── Alerting: tell a human when the monthly pipeline fails ───────────────────
# Without this, a failed run (ingest, dbt, tests or forecast publishing) sits
# silently in the Cloud Run Job's execution list. Set var.alert_email to enable.

resource "google_project_service" "monitoring" {
  count              = var.alert_email == "" ? 0 : 1
  service            = "monitoring.googleapis.com"
  disable_on_destroy = false
}

resource "google_monitoring_notification_channel" "pipeline_email" {
  count        = var.alert_email == "" ? 0 : 1
  display_name = "Inflation pipeline alerts"
  type         = "email"
  labels = {
    email_address = var.alert_email
  }

  depends_on = [google_project_service.monitoring]
}

resource "google_monitoring_alert_policy" "pipeline_failed" {
  count        = var.alert_email == "" ? 0 : 1
  display_name = "Monthly data pipeline failed"
  combiner     = "OR"

  conditions {
    display_name = "inflation-data-pipeline execution failed"
    condition_threshold {
      filter = join(" AND ", [
        "resource.type = \"cloud_run_job\"",
        "resource.labels.job_name = \"${google_cloud_run_v2_job.data_pipeline.name}\"",
        "metric.type = \"run.googleapis.com/job/completed_execution_count\"",
        "metric.labels.result = \"failed\"",
      ])
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_SUM"
      }

      trigger {
        count = 1
      }
    }
  }

  documentation {
    mime_type = "text/markdown"
    content   = <<-EOT
      The monthly pipeline (ingest -> dbt run -> dbt test -> forecast_batch.py) failed.

      Check the execution logs:
      `gcloud run jobs executions list --job inflation-data-pipeline --region ${var.region}`

      The API keeps serving the last published forecasts until a run succeeds.
    EOT
  }

  notification_channels = [google_monitoring_notification_channel.pipeline_email[0].id]
  alert_strategy {
    auto_close = "604800s"
  }
}
