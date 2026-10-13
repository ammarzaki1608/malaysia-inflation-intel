#!/bin/bash
# ── run_pipeline.sh ────────────────────────────────────────────────────────
# Orchestrates the full monthly data refresh:
#   1. Ingest fresh data from data.gov.my into GCS + BigQuery raw tables
#   2. Run dbt to rebuild staging views and mart tables
#   3. Run dbt tests to catch any data quality regressions
#   4. Retrain the forecaster on the new data and publish the forecast
#      table to GCS (the API picks it up within an hour)
#
# Exits non-zero on ANY failure — this is what makes the Cloud Run Job
# report as "Failed" in GCP, which is what triggers alerting.
set -e  # exit immediately if any command fails

echo "=================================================="
echo "  Malaysia Inflation Intelligence — Monthly Refresh"
echo "  Started at: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=================================================="

echo ""
echo "── Step 1/4: Ingesting fresh data from data.gov.my ──"
python ingest.py

echo ""
echo "── Step 2/4: Running dbt transformations ──"
cd /app/transform/dbt_project
dbt run --profiles-dir .

echo ""
echo "── Step 3/4: Running dbt tests ──"
dbt test --profiles-dir .

echo ""
echo "── Step 4/4: Retraining forecaster and publishing forecasts ──"
cd /app/ml/training
python forecast_batch.py --source bigquery

echo ""
echo "=================================================="
echo "  Pipeline completed successfully at: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=================================================="