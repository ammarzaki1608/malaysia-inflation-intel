# Raw data lake — parquet files land here from ingestion
resource "google_storage_bucket" "raw" {
  name          = "${var.project_id}-raw"
  location      = var.region
  force_destroy = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition { age = 90 }
    action { type = "Delete" }
  }
}

# Processed data — dbt outputs, feature tables exported here
resource "google_storage_bucket" "processed" {
  name     = "${var.project_id}-processed"
  location = var.region
}

# ML artifacts — trained models, SHAP outputs, MLflow artifacts
resource "google_storage_bucket" "ml_artifacts" {
  name     = "${var.project_id}-ml-artifacts"
  location = var.region
}

# Terraform state — stores tfstate remotely so it's not on your laptop
resource "google_storage_bucket" "tf_state" {
  name          = "${var.project_id}-tf-state"
  location      = var.region
  force_destroy = false

  versioning {
    enabled = true
  }
}
