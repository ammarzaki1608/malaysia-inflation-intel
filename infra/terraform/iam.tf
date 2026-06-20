# Service account for ingestion Cloud Functions
resource "google_service_account" "ingestion" {
  account_id   = "ingestion-sa"
  display_name = "Ingestion Service Account"
}

# Service account for dbt / BigQuery transformations
resource "google_service_account" "dbt" {
  account_id   = "dbt-sa"
  display_name = "dbt Transformation Service Account"
}

# Service account for Vertex AI training and serving
resource "google_service_account" "vertex" {
  account_id   = "vertex-sa"
  display_name = "Vertex AI Service Account"
}

# Ingestion SA — can read/write GCS and load into BigQuery
resource "google_project_iam_member" "ingestion_gcs" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

resource "google_project_iam_member" "ingestion_bq" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

# dbt SA — can read/write BigQuery
resource "google_project_iam_member" "dbt_bq" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.dbt.email}"
}

resource "google_project_iam_member" "dbt_bq_job" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.dbt.email}"
}

# Vertex SA — can use Vertex AI and read GCS
resource "google_project_iam_member" "vertex_ai" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex.email}"
}

resource "google_project_iam_member" "vertex_gcs" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.vertex.email}"
}

resource "google_project_iam_member" "vertex_bq_job" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.vertex.email}"
}

resource "google_project_iam_member" "vertex_bq_read" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.vertex.email}"
}