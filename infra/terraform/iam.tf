# Who can do what.
#
# Each identity has one job and only the access that job needs:
#
#   ingestion-sa       runs the monthly pipeline job (ingest -> dbt -> forecasts)
#   vertex-sa          runs the API
#   streamlit-app-sa   runs the public Streamlit app; needs no access to anything
#   github-deployer    GitHub Actions: pushes images and deploys them
#   pipeline-scheduler-sa   Cloud Scheduler: starts the monthly job (scheduler.tf)
#
# Storage access is granted per bucket, not across the project.

# ── Monthly pipeline job ─────────────────────────────────────────────────────
resource "google_service_account" "ingestion" {
  account_id   = "ingestion-sa"
  display_name = "Monthly pipeline job"
}

# Writes raw files to the raw bucket; BigQuery load jobs read them back from it.
resource "google_storage_bucket_iam_member" "ingestion_raw" {
  bucket = google_storage_bucket.raw.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.ingestion.email}"
}

# Publishes the forecast table and scorecard, and reads earlier ones to score them.
resource "google_storage_bucket_iam_member" "ingestion_ml_artifacts" {
  bucket = google_storage_bucket.ml_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.ingestion.email}"
}

# Loads the raw tables and builds the dbt models.
resource "google_project_iam_member" "ingestion_bq" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

resource "google_project_iam_member" "ingestion_bq_job" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

# ── API ──────────────────────────────────────────────────────────────────────
resource "google_service_account" "vertex" {
  account_id   = "vertex-sa"
  display_name = "API service"
}

# Reads the forecast table, the scorecard and the FAISS index.
resource "google_storage_bucket_iam_member" "api_ml_artifacts" {
  bucket = google_storage_bucket.ml_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.vertex.email}"
}

# Queries the marts for the trend and leaderboard endpoints. (Its write access
# to the LLM log dataset is in bigquery.tf.)
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

# The Gemini API key. The secret itself was created by hand, outside Terraform.
resource "google_secret_manager_secret_iam_member" "api_gemini_key" {
  secret_id = "gemini-api-key"
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.vertex.email}"
}

# ── Streamlit app ────────────────────────────────────────────────────────────
# The app only calls the API over HTTPS, so its identity is given no roles. If
# the public app were ever compromised, there is nothing in the project it
# could read or change.
resource "google_service_account" "streamlit" {
  account_id   = "streamlit-app-sa"
  display_name = "Streamlit app"
}

# ── GitHub Actions, without a stored key ─────────────────────────────────────
# Workload Identity Federation: GitHub signs a short-lived token for each
# workflow run, and Google exchanges it for credentials that expire within the
# hour. There is no long-lived key to store in GitHub, leak or rotate.
resource "google_project_service" "sts" {
  service            = "sts.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "iamcredentials" {
  service            = "iamcredentials.googleapis.com"
  disable_on_destroy = false
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  depends_on                = [google_project_service.sts, google_project_service.iamcredentials]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions"
  display_name                       = "GitHub Actions"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  # Only this repository, and only workflows running from main. A workflow
  # edited on another branch cannot obtain credentials.
  attribute_condition = "assertion.repository == \"${var.github_repository}\" && assertion.ref == \"refs/heads/main\""
}

resource "google_service_account" "deployer" {
  account_id   = "github-deployer"
  display_name = "GitHub Actions deployer"
}

# Workflows from this repository may act as the deployer.
resource "google_service_account_iam_member" "deployer_from_github" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

# Push images to this one repository.
resource "google_artifact_registry_repository_iam_member" "deployer_push" {
  location   = google_artifact_registry_repository.main.location
  repository = google_artifact_registry_repository.main.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer.email}"
}

# Deploy new revisions and update the job. Cannot change who may call a service.
resource "google_project_iam_member" "deployer_run" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

# Deploying a revision that runs as a service account requires permission to
# "act as" it. Granted on the three runtime accounts only, not project-wide.
resource "google_service_account_iam_member" "deployer_acts_as" {
  for_each = {
    api      = google_service_account.vertex.name
    app      = google_service_account.streamlit.name
    pipeline = google_service_account.ingestion.name
  }
  service_account_id = each.value
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

output "workload_identity_provider" {
  description = "Value for workload_identity_provider in the GitHub workflows."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deployer_service_account" {
  value = google_service_account.deployer.email
}
