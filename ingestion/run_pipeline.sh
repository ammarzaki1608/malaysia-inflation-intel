#!/bin/bash
# ── run_pipeline.sh ────────────────────────────────────────────────────────
# Orchestrates the full monthly data refresh:
#   1. Ingest fresh data from data.gov.my into GCS + BigQuery raw tables
#   2. Run dbt to rebuild staging views and mart tables
#   3. Run dbt tests to catch any data quality regressions
#
# Exits non-zero on ANY failure — this is what makes the Cloud Run Job
# report as "Failed" in GCP, which is what triggers alerting.
set -e  # exit immediately if any command fails

echo "=================================================="
echo "  Malaysia Inflation Intelligence — Monthly Refresh"
echo "  Started at: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=================================================="

echo ""
echo "── Step 1/3: Ingesting fresh data from data.gov.my ──"
python ingest.py

echo ""
echo "── Step 2/3: Running dbt transformations ──"
cd /app/transform/dbt_project
dbt run --profiles-dir .

echo ""
echo "── Step 3/3: Running dbt tests ──"
dbt test --profiles-dir .

echo ""
echo "=================================================="
echo "  Pipeline completed successfully at: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=================================================="