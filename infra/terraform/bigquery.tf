# Raw layer — direct loads from GCS, no transformations
resource "google_bigquery_dataset" "raw" {
  dataset_id  = "raw"
  description = "Raw DOSM and BNM data, loaded directly from GCS"
  location    = var.region
}

# Staging layer — dbt cleaning models
resource "google_bigquery_dataset" "staging" {
  dataset_id  = "staging"
  description = "dbt staging models — cleaned and typed"
  location    = var.region
}

# Mart layer — dbt analytical models, feature tables
resource "google_bigquery_dataset" "mart" {
  dataset_id  = "mart"
  description = "dbt mart models — analytical tables and ML features"
  location    = var.region
}
