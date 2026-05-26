"""
Ingestion pipeline for Malaysia Inflation Intelligence project.

Downloads parquet files from datta.gov.my, uploads to GCS raw buckett,
and loads into Bigquery raw dataset. Designed to be idempotent - 
safe to run multiple times on the same datta without duplicating rows.

Usage:
    python ingest.py                # ingest all datasets
    python ingest.py cpi_state     # ingest one dataset
"""

import sys
import io
import logging
from datetime import datetime, timezone

import pandas as pd
import requests
from google.cloud import storage, bigquery

from config import DATASETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def get_project_id():
    """Get GCP project ID from the active gcloud config."""
    client = storage.Client()
    return client.project

def download_parquet(url):
    """Download a parquet file from a URL and return as DataFrame."""
    logger.info(f"Downloading from {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    buffer = io.BytesIO(response.content)
    df = pd.read_parquet(buffer)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    logger.info(f" Downloaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def upload_to_gcs(df, project_id, gcs_path):
    """Upload DataFrame as parquet to GCS raw bucket."""
    bucket_name = f"{project_id}-raw"
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    versioned_path = f"{gcs_path.replace('.parquet', '')}_{timestamp}.parquet"
    latest_path = gcs_path

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)

    blob = bucket.blob(versioned_path)
    blob.upload_from_file(buffer, content_type="application/octet-stream")
    logger.info(f" Uploaded versioned: gs://{bucket_name}/{versioned_path}")

    buffer.seek(0)
    blob_latest = bucket.blob(latest_path)
    blob_latest.upload_from_file(buffer, content_type="application/octet-stream")
    logger.info(f" Uploaded latest: gs://{bucket_name}/{latest_path}")

    return f"gs://{bucket_name}/{latest_path}"

def load_to_bigquery(df, project_id, bq_table):
    """Load DataFrame into BigQuery, replacing existing data (full refresh).

    Uses WRITE_TRUNCATE to ensure idempotency — running twice
    produces the same result, never duplicates rows.
    """
    client = bigquery.Client(project=project_id)
    full_table_id = f"{project_id}.{bq_table}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
    job.result()

    table = client.get_table(full_table_id)
    logger.info(f" Loaded {table.num_rows:,} rows into {full_table_id}")
    return table.num_rows

def validate_data(df, dataset_name):
    """Run basic data quality checks. Raises ValueError on failure."""
    errors = []

    if len(df) == 0:
        errors.append(f"{dataset_name}: Dataframe is empty")

    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        errors.append(f"{dataset_name}: Fully null columns: {null_cols}")
    
    if "date" in df.columns:
        null_dates = df["date"].isnull().sum()
        if null_dates > 0:
            errors.append(f"{dataset_name}: {null_dates} null dates found")

    min_rows = {"cpi_state": 100, "cpi_national": 500, "ppi_heaadline": 100}
    if dataset_name in min_rows and len(df) < min_rows[dataset_name]:
        errors.append(
            f"{dataset_name}: Only {len(df)} rows, expected <= {min_rows[dataset_name]}"
        )

    if errors:
        for e in errors:
            logger.error(f" VALIDATION FAILED: {e}")
        raise ValueError(f"Data validation failed for {dataset_name} ")
    
    logger.info(f" Validation passed: {len(df):,} rows, {df.isnull().sum().sum()} total nulls")

def ingest_dataset(name, config):
        """Full ingestion pipeline for a single dataset."""
        logger.info(f"{'='*60}")
        logger.info(f"Ingesting: {name}")
        logger.info(f" Source: {config['url']}")
        logger.info(f" Target: {config['bq_table']}")

        project_id = get_project_id()

        df = download_parquet(config["url"])
        validate_data(df, name)
        gcs_url = upload_to_gcs(df, project_id, config["gcs_path"])
        row_count = load_to_bigquery(df, project_id, config["bq_table"])

        result = {
            "dataset": name,
            "rows": row_count,
            "columns": len(df.columns),
            "gcs_url": gcs_url,
            "bq_table": config["bq_table"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "success",
        }

        logger.info(f" Completed: {name} ({row_count:,} rows)")
        return result    
    
def main():
        """Run ingestion for all datasets or a specific one."""
        if len(sys.argv) > 1:
            target = sys.argv[1]
            if target not in DATASETS:
                logger.error(f"Unknown dataset: {target}")
                logger.error(f"Available: {', '.join(DATASETS.keys(()))}")
                sys.exit(1)
            dataset_to_ingest = {target: DATASETS[target]}
        else:
            dataset_to_ingest = DATASETS

        logger.info(f"Starting ingestion: {len(dataset_to_ingest)} dataset(s)")
        results = []

        for name, config in dataset_to_ingest.items():
            try:
                result = ingest_dataset(name, config)
                results.append(result)
            except Exception as e:
                logger.error(f"FAILED: {name} - {e}")
                results.append({"dataset": name, "status": "failed", "error": str(e)})

        logger.info(f"\n{'='*60}")
        logger.info("INGESTION SUMMARY")
        succeeded = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r ["status"] == "failed")
        logger.info(f" Succeeded: {succeeded}/{len(results)}")
        logger.info(f" Failed: {failed}/{len(results)}")

        for r in results:
            status = "OK" if r["status"] == "success" else "FAIL"
            rows = f"{r.get('rows', 0):,}" if "rows" in r else "N/A"
            logger.info(f" [{status}] {r['dataset']}: {rows} rows")

        if failed > 0:
            sys.exit(1)    

if __name__ == "__main__":
    main()