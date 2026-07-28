# Malaysia Personal Inflation Intelligence

> A production data science system that ingests official Malaysian government data monthly, forecasts CPI inflation by state and category using XGBoost, and serves a bilingual, citizen-facing app where Malaysians can calculate their **personal inflation rate**, plan relocation decisions, explore real historical trends, and ask questions in plain language — grounded in a retrieval-augmented generation (RAG) layer.

**Live App:** [inflation-app-lxxpkjdxja-as.a.run.app](https://inflation-app-lxxpkjdxja-as.a.run.app)
**API Docs:** [inflation-api-lxxpkjdxja-as.a.run.app/docs](https://inflation-api-lxxpkjdxja-as.a.run.app/docs)

Available in **English** and **Bahasa Malaysia**.

---

## The Problem

Malaysia's official CPI is a single national average. A person in Kelantan spending 45% of income on food experiences completely different inflation to a professional in KL spending 30% on housing. This project makes that gap visible, quantifiable, and personal — using real DOSM and BNM data, refreshed automatically every month.

---

## What It Does

**📊 Personal Budget Forecast** — enter your salary and real spending in ringgit per category. Get a 3/6/12-month recursive forecast per category with honestly-widening confidence intervals, a plain-language interpretation of your result against the live national headline rate, and an AI-generated general budgeting suggestion grounded in your own numbers.

**🗺️ Relocation Planner** — compare cost of living between any two Malaysian states. Destination spending is auto-suggested using real DOSM price ratios (editable if your habits will change), with a salary-needed estimate and the same AI-generated insight layer.

**📈 Data Explorer** — a national overview (headline inflation, category leaderboard, state leaderboard) before drilling into historical CPI/PPI/OPR trends for any state and category, with hover-annotated real policy events (e.g. the 2024 SST rate increase, the 2024 targeted diesel subsidy).

**💬 Ask Anything** — a chat interface embedded in both forecast pages. Answers general questions about how the app works or personal questions about your own forecast, grounded in a retrieval-augmented knowledge base — never inventing numbers, and fetching live forecast data only when a question actually needs it.

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
[BigQuery raw dataset]  ←── Exact DOSM/BNM schema, no transformation
        │
        ▼
[dbt transform layer]  ←── 7 staging views + 2 mart tables, 23 data quality tests
        │  Staging: type casting, null filtering, income band classification
        │  Mart: lag features (1/3/6/12mo), rolling averages (3/12mo), PPI leads, OPR
        ▼
[BigQuery mart dataset]  ←── mart_cpi_features (43,900+ rows), mart_income_bands
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
[ml/training/train.py]                    [Cloud Scheduler]
  Prophet baseline + XGBoost,                (cron: 3rd of every month)
  walk-forward CV, MLflow, SHAP                     │
  Model saved to GCS                                ▼
        │                              [Cloud Run Job: inflation-data-pipeline]
        │                                (re-runs ingest.py → dbt run → dbt test)
        ▼
[ml/serving/api.py]  ←── FastAPI, model + FAISS index loaded from GCS on startup
        │  /forecast            — recursive multi-month CPI forecast
        │  /price-ratio         — price ratio between two states, one category
        │  /trends              — historical CPI/PPI/OPR time series
        │  /category-leaderboard, /state-leaderboard — national/state rankings
        │  /ask                 — RAG-grounded Q&A (Gemini + FAISS)
        │  /prescriptive-insight — AI-generated general budgeting suggestion
        ▼
[app/streamlit/app.py]  ←── Bilingual (EN/BM) citizen-facing app, Plotly charts
        │
        ▼
[Cloud Run]  ←── Both services containerised with Docker, deployed via Terraform + GitHub Actions
```

---

## The RAG Layer

```
Knowledge base (25 hand-curated entries)
        │  embedded once via gemini-embedding-001 (RETRIEVAL_DOCUMENT)
        ▼
FAISS flat L2 index  ←── stored in GCS, loaded once at API startup
        │
        │  live question ──► embedded via gemini-embedding-001 (RETRIEVAL_QUERY)
        ▼
Top-k retrieval  +  live forecast data (only if the question needs it)
        ▼
gemini-3.1-flash-lite  ←── temperature 0.3, strict "never invent facts" system instruction
        ▼
Grounded answer, with retrieved-context IDs returned for traceability
```

**Evaluation:** a custom Gemini-based LLM-as-judge harness (built after the standard `ragas` library hit an unresolvable dependency conflict — see Known Issues below) scores every answer on faithfulness, answer relevancy, and context precision against a versioned 10-question test set, gating deployment at a 0.7 threshold.

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

**RAG evaluation** (10-question test set): faithfulness 1.00, answer relevancy 0.95, context precision 1.00.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Infrastructure | Terraform, Google Cloud Platform |
| Storage | Google Cloud Storage (GCS), BigQuery, Secret Manager |
| Ingestion | Python, pandas, requests, google-cloud-bigquery |
| Transformation | dbt, SQL (BigQuery dialect) |
| ML Training | XGBoost, Prophet, scikit-learn, SHAP, MLflow |
| GenAI / RAG | Gemini API (gemini-embedding-001, gemini-3.1-flash-lite), FAISS |
| Serving | FastAPI, uvicorn, joblib, slowapi (rate limiting) |
| Frontend | Streamlit (bilingual EN/BM), Plotly |
| Automation | Cloud Scheduler, Cloud Run Jobs |
| Deployment | Docker, Cloud Run, Artifact Registry |
| CI/CD | GitHub Actions |
| Auth | GCP Service Accounts, IAM (least-privilege), Secret Manager |

---

## Data Sources

All data is publicly available from the Malaysian government with no authentication required, and refreshed automatically every month via a scheduled Cloud Run Job.

| Dataset | Source | Rows | Frequency |
|---|---|---|---|
| CPI by State (13 divisions × 16 states) | DOSM / data.gov.my | 43,900+ | Monthly |
| CPI National (headline + by division) | DOSM / data.gov.my | 7,700+ | Monthly |
| Producer Price Index (PPI) | DOSM / data.gov.my | 575+ | Monthly |
| PPI by SITC | DOSM / data.gov.my | 5,175+ | Monthly |
| Household Income & Expenditure (HIES) | DOSM / data.gov.my | 16 | Survey (2022 edition) |
| Interest Rates / OPR | BNM / data.gov.my | 5,700+ | Monthly |

---

## Project Structure

```
malaysia-inflation-intel/
├── infra/terraform/           # GCP infrastructure as code
│   ├── main.tf, storage.tf, bigquery.tf, iam.tf
│   ├── cloudrun.tf             # API + Streamlit services, Secret Manager wiring
│   └── scheduler.tf            # Cloud Run Job + Cloud Scheduler (monthly pipeline)
├── ingestion/
│   ├── ingest.py, config.py, requirements.txt
│   ├── run_pipeline.sh         # ingest → dbt run → dbt test, for the scheduled Job
│   └── Dockerfile
├── transform/dbt_project/
│   └── models/{staging,mart}/
├── ml/
│   ├── training/                # prepare_features.py, train.py
│   └── serving/
│       ├── api.py               # FastAPI — forecast, RAG, leaderboard endpoints
│       ├── rag/
│       │   ├── knowledge_base.py    # 25 hand-curated entries
│       │   ├── build_index.py       # embeds corpus → FAISS index → GCS
│       │   └── retriever.py         # loads index, embeds queries, searches
│       ├── Dockerfile, requirements.txt
├── evaluation/
│   ├── test_questions.py        # versioned RAG evaluation test set
│   ├── scorer.py                # custom Gemini LLM-as-judge (faithfulness, relevancy, precision)
│   ├── run_eval.py               # runner + CI gate
│   └── knowledge_base_mirror.py  # mirrors the real knowledge base for scoring context
├── app/streamlit/
│   ├── app.py                   # bilingual UI — all 4 modes
│   ├── Dockerfile, requirements.txt
└── .github/workflows/
    ├── terraform.yml             # validate Terraform on every PR
    └── deploy.yml                # build + push + self-correcting deploy on merge to main
```

---

## Running Locally

### Prerequisites
- Python 3.11+ (API/app), Python 3.12 (Docker base image)
- GCP project with BigQuery, GCS, Secret Manager access
- `gcloud auth application-default login`
- dbt with BigQuery adapter: `pip install dbt-bigquery`
- A Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

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
```

### 4 — Build the RAG index (one-time, or after editing knowledge_base.py)
```bash
cd ml/serving/rag
export GEMINI_API_KEY="your-key"
python build_index.py
```

### 5 — Start the API
```bash
cd ml/serving
export GEMINI_API_KEY="your-key"
uvicorn api:app --reload --port 8081
```

### 6 — Start the Streamlit app
```bash
cd app/streamlit
API_URL=http://localhost:8081 streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

### 7 — Run the RAG evaluation harness (optional)
```bash
cd evaluation
export GEMINI_API_KEY="your-key"
export API_URL=http://localhost:8081
python run_eval.py
```

---

## Deployment

Infrastructure is fully defined in Terraform, including Secret Manager-backed API keys and the automated monthly pipeline. CI/CD via GitHub Actions runs automatically on push to `main` when `ml/serving/` or `app/streamlit/` files change, and dynamically fetches each service's real deployed URL rather than guessing it — self-correcting if Cloud Run's URL format ever changes.

```bash
cd infra/terraform
terraform apply
```

Requires `GCP_SA_KEY` secret set in GitHub repository settings.

---

## Public Launch Safety

- **Rate limiting** — 30 req/min on data endpoints, 15 req/min on LLM-touching endpoints (`/ask`, `/prescriptive-insight`), enforced via `slowapi`, verified with a 35-request burst test
- **Input validation** — state/category parameters checked before any BigQuery query runs
- **GCP budget alert** — configured with alerts at 50/90/100% of monthly threshold
- **Cloud Run scaling caps** — hard ceiling on concurrent compute regardless of traffic (max 3 API instances, 2 app instances)
- **Secrets** — Gemini API key stored in Secret Manager, injected at runtime, never in plaintext Terraform

---

## GCP Resource Identifiers

| Resource | Value |
|---|---|
| Project ID | modular-rush-495414-i2 |
| Region | asia-southeast1 (Singapore) |
| API service | inflation-api (Cloud Run) |
| App service | inflation-app (Cloud Run) |
| Data pipeline job | inflation-data-pipeline (Cloud Run Job) |
| Pipeline schedule | 3rd of every month, 03:00 UTC (Cloud Scheduler) |

---

## Known Issues / Engineering Notes

- **RAGAS dependency conflict.** The standard RAG evaluation library `ragas` (both 0.4.3 and 0.3.9) hit an unpatched upstream bug — `ragas.llms.base` imports `ChatVertexAI` from a `langchain_community` path that was relocated upstream, confirmed via the library's own GitHub issues (#2741, #2745). The documented workaround (pinning `langchain-community==0.2.17`) then surfaced a separate, unrelated `numpy==1.26.4` binary incompatibility on Python 3.13/Windows. Rather than continue chasing a fragile dependency chain, `evaluation/scorer.py` implements the same LLM-as-judge approach directly on the Gemini API.
- **Category-specific causal reasoning.** The `/ask` endpoint can explain *how* a metric like "biggest leak" is calculated, but doesn't yet have per-category economic narratives explaining *why* a specific category (e.g. Transport) is moving the way it is, beyond general model mechanics. Documented in `knowledge_base.py` as `category_specific_causes_limitation`.
- **State-level granularity.** CPI and price data are reported at the state level, not district/city level — DOSM's district-level "Decent Living Expenditure Index" (released Dec 2025) is a logged future improvement, not yet integrated.
- **Recursive forecasting.** Multi-month forecasts feed each prediction back in as input for the next month; accuracy naturally decreases with horizon length since later predictions increasingly rely on the model's own prior outputs rather than confirmed data. Confidence bands widen accordingly.

---

## Roadmap

- LangGraph-based budget advisor agent (tool-calling across `/forecast`, `/price-ratio`, `/ask`)
- LLM observability — logging retrieved context, prompts, and latency per `/ask` call to BigQuery
- Automated model retraining gate — compare newly retrained models against the deployed one before promotion
- Prediction logging for real-world drift monitoring
- District-level cost-of-living integration (DOSM PAKW dataset)

---

## Author

**Muhammad Ammar Ahmad Zaki** — Computer Science and AI graduate, George Town, Penang.
· [Email](mailto:ammarzaki160802@gmail.com)

---

*Data sourced from [data.gov.my](https://data.gov.my) and [OpenDOSM](https://open.dosm.gov.my). Published under the terms of the Malaysian Open Data initiative.*