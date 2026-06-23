# Malaysia Personal Inflation Intelligence

> An end-to-end production data science system that ingests official Malaysian government data monthly, forecasts CPI inflation by state and category using XGBoost, and serves a citizen-facing web app where Malaysians can calculate their **personal inflation rate** and plan relocation decisions.

**Live Demo:** [inflation-app-lxxpkjdxja-as.a.run.app](https://inflation-app-lxxpkjdxja-as.a.run.app)  
**API Docs:** [inflation-api-lxxpkjdxja-as.a.run.app/docs](https://inflation-api-lxxpkjdxja-as.a.run.app/docs)

---

## The Problem

Malaysia's official CPI is a single national average. A person in Kelantan spending 45% of income on food experiences completely different inflation to a professional in KL spending 30% on housing. This project makes that difference visible and quantifiable — using real DOSM and BNM data.

---

## What It Does

**Personal Inflation Calculator** — enter your spending weights across 13 CPI categories and your state, get your personal inflation rate vs the state average with SHAP-powered explanations of what's driving it.

**Relocation Planner** — compare cost of living between any two Malaysian states. See the category-level price difference, how your personal inflation rate changes, and the estimated monthly salary needed to maintain your current standard of living in the destination state.

---

## Architecture

```
data.gov.my / OpenDOSM
        │
        ▼
[ingestion/ingest.py]  ←── Python, requests, pandas
        │  Download parquet → validate → upload to GCS → load to BigQuery (WRITE_TRUNCATE)
        ▼
[GCS raw bucket]  ←── Versioned + latest copies of every raw file
        │
        ▼
[BigQuery raw dataset]  ←── Exact DOSM schema, no transformation
        │
        ▼
[dbt transform layer]  ←── 7 staging views + 2 mart tables, 23 data quality tests
        │  Staging: type casting, null filtering, income band classification
        │  Mart: lag features (1/3/6/12mo), rolling averages (3/12mo),
        │        PPI lead indicators, OPR rate, COVID flag
        ▼
[BigQuery mart dataset]  ←── mart_cpi_features (43,900 rows), mart_income_bands
        │
        ▼
[ml/training/train.py]  ←── Prophet baseline + XGBoost, walk-forward CV, MLflow, SHAP
        │  Model saved to GCS ml-artifacts bucket
        ▼
[ml/serving/api.py]  ←── FastAPI, loads model from GCS on startup
        │  /predict — personal inflation rate + SHAP
        │  /compare — relocation cost comparison + salary estimate
        ▼
[app/streamlit/app.py]  ←── Citizen-facing web app, Plotly charts
        │
        ▼
[Cloud Run]  ←── Both services containerised with Docker, deployed via Terraform + GitHub Actions
```

---

## Key Results

| Metric | Prophet (baseline) | XGBoost (production) |
|---|---|---|
| MAE | 5.549 | **0.733** |
| RMSE | 7.447 | **1.624** |
| MAPE | 5.06% | **0.63%** |
| Directional accuracy | 0.75% | **96.98%** |

Training data: 38,752 rows across 208 series (16 states × 13 CPI divisions), 2010–2026, walk-forward cross-validation with 5 splits.

Top SHAP features: `cpi_ma_3` (3-month CPI trend), `cpi_lag_1` (last month CPI), `ppi_lag_3` (producer prices 3 months prior).

---

## Tech Stack

| Layer | Tools |
|---|---|
| Infrastructure | Terraform, Google Cloud Platform |
| Storage | Google Cloud Storage (GCS), BigQuery |
| Ingestion | Python, pandas, requests, google-cloud-bigquery |
| Transformation | dbt, SQL (BigQuery dialect) |
| ML Training | XGBoost, Prophet, scikit-learn, SHAP, MLflow |
| Serving | FastAPI, uvicorn, joblib |
| Frontend | Streamlit, Plotly |
| Deployment | Docker, Cloud Run, Artifact Registry |
| CI/CD | GitHub Actions |
| Auth | GCP Service Accounts, IAM (least-privilege) |

---

## Data Sources

All data is publicly available from the Malaysian government with no authentication required.

| Dataset | Source | Rows | Frequency |
|---|---|---|---|
| CPI by State (14 divisions × 16 states) | DOSM / data.gov.my | 43,904 | Monthly |
| CPI National | DOSM / data.gov.my | 7,784 | Monthly |
| Producer Price Index (PPI) | DOSM / data.gov.my | 575 | Monthly |
| PPI by SITC | DOSM / data.gov.my | 5,175 | Monthly |
| Household Income & Expenditure (HIES) | DOSM / data.gov.my | 16 | Survey (5yr) |
| Interest Rates / OPR | BNM / data.gov.my | 5,712 | Monthly |

---

## Project Structure

```
malaysia-inflation-intel/
├── infra/terraform/          # GCP infrastructure as code
│   ├── main.tf               # Provider + version constraints
│   ├── storage.tf            # GCS buckets (raw, processed, ml-artifacts, tf-state)
│   ├── bigquery.tf           # BigQuery datasets (raw, staging, mart)
│   ├── iam.tf                # Service accounts + least-privilege bindings
│   └── cloudrun.tf           # Cloud Run services + Artifact Registry
├── ingestion/
│   ├── config.py             # Dataset registry (URLs, GCS paths, BQ tables)
│   └── ingest.py             # Download → validate → GCS → BigQuery
├── transform/dbt_project/
│   └── models/
│       ├── staging/          # 7 views: type casting, null filtering, OPR filter
│       └── mart/             # mart_cpi_features (ML table), mart_income_bands
├── ml/
│   ├── training/
│   │   ├── prepare_features.py   # BigQuery → clean DataFrame + walk-forward splits
│   │   └── train.py              # Prophet + XGBoost + MLflow + SHAP + GCS save
│   └── serving/
│       ├── api.py                # FastAPI prediction service
│       ├── Dockerfile
│       └── requirements.txt
├── app/streamlit/
│   ├── app.py                # Personal calculator + relocation planner
│   ├── Dockerfile
│   └── requirements.txt
└── .github/workflows/
    ├── terraform.yml         # Validate Terraform on every PR
    └── deploy.yml            # Build + push + deploy on merge to main
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- GCP project with BigQuery and GCS access
- `gcloud auth application-default login`
- dbt with BigQuery adapter: `pip install dbt-bigquery`

### 1 — Ingest data
```bash
cd ingestion
pip install -r requirements.txt
python ingest.py
```

### 2 — Run dbt transformations
```bash
cd transform/dbt_project
dbt run
dbt test
```

### 3 — Train the model
```bash
cd ml/training
python train.py
# Add --division 01 --state Selangor for a quick single-series test
```

### 4 — Start the API
```bash
cd ml/serving
uvicorn api:app --reload --port 8080
```

### 5 — Start the Streamlit app
```bash
cd app/streamlit
API_URL=http://localhost:8080 streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Deployment

Infrastructure is fully defined in Terraform. Deploying to GCP:

```bash
# Build and push images
docker build -t asia-southeast1-docker.pkg.dev/{project}/malaysia-inflation-intel/inflation-api:latest ml/serving/
docker push asia-southeast1-docker.pkg.dev/{project}/malaysia-inflation-intel/inflation-api:latest

docker build -t asia-southeast1-docker.pkg.dev/{project}/malaysia-inflation-intel/inflation-app:latest app/streamlit/
docker push asia-southeast1-docker.pkg.dev/{project}/malaysia-inflation-intel/inflation-app:latest

# Deploy
cd infra/terraform
terraform apply
```

CI/CD via GitHub Actions runs automatically on push to `main` when `ml/serving/` or `app/streamlit/` files change. Requires `GCP_SA_KEY` secret set in GitHub repository settings.

---

## GCP Resource Identifiers

| Resource | Value |
|---|---|
| Project ID | modular-rush-495414-i2 |
| Region | asia-southeast1 (Singapore) |
| API service | inflation-api (Cloud Run) |
| App service | inflation-app (Cloud Run) |
| Model artifact | gs://modular-rush-495414-i2-ml-artifacts/models/cpi_forecast_20260619_194024.joblib |

---

## Known Limitations

- CPI is state-level only — no district or city granularity
- HIES income data is from the 2022 survey — approximately 3–4 years old
- OPR data lags 1–2 months behind BNM publication
- Walk-forward CV splits cover 2013–2015; performance on 2020–2026 patterns is not fully evaluated
- Single model across all 208 series — per-division models would likely improve accuracy

---

## Author

**Muhammad Ammar Ahmad Zaki** — Computer Science and AI graduate, George Town, Penang.  
[LinkedIn](www.linkedin.com/in/muhammad-ammar-ahmad-zaki) · [Email](mailto:ammarzaki160802@gmail.com)

---

*Data sourced from [data.gov.my](https://data.gov.my) and [OpenDOSM](https://open.dosm.gov.my). Published under the terms of the Malaysian Open Data initiative.*
