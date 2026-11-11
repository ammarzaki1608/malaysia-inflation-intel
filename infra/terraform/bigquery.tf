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

# Observability — one row per LLM call made by our own evaluation run.
# Calls from the app's users are never logged (see log_eval_call in api.py).
resource "google_bigquery_dataset" "observability" {
  dataset_id  = "observability"
  description = "LLM call logs for evaluation traffic only: latency, tokens, retrieved context, errors"
  location    = var.region
}

resource "google_bigquery_table" "llm_calls" {
  dataset_id          = google_bigquery_dataset.observability.dataset_id
  table_id            = "llm_calls"
  description         = "One row per LLM call made by the evaluation run. Written by ml/serving/api.py log_llm_call."
  deletion_protection = false

  # Only evaluation questions are stored; old runs are not worth keeping forever.
  time_partitioning {
    type          = "DAY"
    field         = "ts"
    expiration_ms = 7776000000 # 90 days
  }

  schema = jsonencode([
    { name = "call_id", type = "STRING", mode = "REQUIRED", description = "Random id for this call" },
    { name = "ts", type = "TIMESTAMP", mode = "REQUIRED", description = "When the call finished (UTC)" },
    { name = "endpoint", type = "STRING", mode = "REQUIRED", description = "/ask or /prescriptive-insight" },
    { name = "status", type = "STRING", mode = "REQUIRED", description = "ok or error" },
    { name = "model", type = "STRING", mode = "NULLABLE", description = "Generation model" },
    { name = "api_revision", type = "STRING", mode = "NULLABLE", description = "Cloud Run revision that served the call" },
    { name = "forecast_version", type = "STRING", mode = "NULLABLE", description = "Forecast table version loaded at the time" },
    { name = "question", type = "STRING", mode = "NULLABLE", description = "The evaluation question (/ask only)" },
    { name = "question_chars", type = "INTEGER", mode = "NULLABLE" },
    { name = "state", type = "STRING", mode = "NULLABLE" },
    { name = "division", type = "STRING", mode = "NULLABLE" },
    { name = "horizon_months", type = "INTEGER", mode = "NULLABLE" },
    { name = "context_type", type = "STRING", mode = "NULLABLE", description = "/prescriptive-insight only" },
    { name = "used_live_forecast_data", type = "BOOLEAN", mode = "NULLABLE", description = "Whether forecast numbers were put in the prompt" },
    { name = "retrieved_ids", type = "STRING", mode = "REPEATED", description = "Knowledge-base entries retrieved, best first" },
    { name = "retrieved_distances", type = "FLOAT", mode = "REPEATED", description = "L2 distance of each retrieved entry (lower is closer)" },
    { name = "retrieval_ms", type = "INTEGER", mode = "NULLABLE", description = "Embedding the question and searching the index" },
    { name = "generation_ms", type = "INTEGER", mode = "NULLABLE", description = "The model call" },
    { name = "total_ms", type = "INTEGER", mode = "NULLABLE", description = "Whole request, excluding this log write" },
    { name = "prompt_chars", type = "INTEGER", mode = "NULLABLE" },
    { name = "prompt_tokens", type = "INTEGER", mode = "NULLABLE" },
    { name = "output_tokens", type = "INTEGER", mode = "NULLABLE" },
    { name = "total_tokens", type = "INTEGER", mode = "NULLABLE" },
    { name = "answer", type = "STRING", mode = "NULLABLE", description = "The model's answer (/ask only)" },
    { name = "answer_chars", type = "INTEGER", mode = "NULLABLE" },
    { name = "finish_reason", type = "STRING", mode = "NULLABLE", description = "Why the model stopped (STOP, MAX_TOKENS, ...)" },
    { name = "error_type", type = "STRING", mode = "NULLABLE", description = "Exception class when status is error" },
    { name = "error_message", type = "STRING", mode = "NULLABLE", description = "First 500 characters of the error" },
  ])
}

# The API's service account may add rows to this dataset only.
resource "google_bigquery_dataset_iam_member" "api_llm_log_writer" {
  dataset_id = google_bigquery_dataset.observability.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.vertex.email}"
}
