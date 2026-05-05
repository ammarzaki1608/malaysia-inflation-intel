# Malaysia Personal Inflation Intelligence

An end-to-end data science project that ingests official DOSM and BNM data,
forecasts category-level CPI using ML models, and serves a personalised
inflation calculator for Malaysian citizens.

## Architecture
GCP | BigQuery | Cloud Composer | Vertex AI | dbt | MLflow | Streamlit

## Datasets
- DOSM Monthly CPI by State (data.gov.my)
- DOSM Producer Price Index (data.gov.my)
- DOSM Household Income & Expenditure Survey (data.gov.my)
- BNM Interest Rates / OPR (data.gov.my)

## Project Phases
- Phase 1: GCP setup & infrastructure (Terraform)
- Phase 2: Data ingestion pipeline (Airflow + Cloud Functions)
- Phase 3: Data modelling & feature engineering (dbt + BigQuery)
- Phase 4: ML modelling & experiment tracking (Vertex AI + MLflow)
- Phase 5: API, app & dashboard (FastAPI + Streamlit + Looker Studio)
- Phase 6: MLOps, migration & launch

## Stack
Python 3.11 | Terraform | dbt | Apache Airflow | Prophet | XGBoost | LSTM
