FROM python:3.12-slim

WORKDIR /app

# Install system dependencies dbt needs
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install all pipeline dependencies (ingest, dbt + BigQuery adapter, forecaster)
# from one fully pinned lock file, so rebuilds are reproducible
COPY ingestion/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy ingestion code
COPY ingestion/ ./

# Copy the dbt project so it's available at /app/transform/dbt_project
COPY transform/dbt_project ./transform/dbt_project

# Copy the batch forecaster (step 4 of run_pipeline.sh)
COPY ml/training/backtest.py ml/training/forecast_batch.py ./ml/training/

# Make the pipeline script executable
RUN chmod +x run_pipeline.sh

# This container runs once and exits — no web server, no port exposed
ENTRYPOINT ["./run_pipeline.sh"]