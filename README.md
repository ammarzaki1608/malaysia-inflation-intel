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

**📊 Personal Budget Forecast** — enter your salary and real spending in ringgit per category. See your **personal inflation over the past 12 months** next to the official rate over the same period (like-for-like). Then see a 3/6/12-month forecast per category with backtest-calibrated 80% ranges, plus an AI-generated general budgeting suggestion grounded in your own numbers.

**🗺️ Relocation Planner** — compare cost of living between any two Malaysian states. Destination spending starts from the difference in average household spending (DOSM HIES 2022) and is fully editable. Each state's prices are then forecast forward, with a salary-needed estimate and the same AI-generated insight layer. (It deliberately doesn't divide one state's CPI index by another's: index levels measure change since a base period, not how expensive a state is.)

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
[ml/training/forecast_batch.py]           [Cloud Scheduler]
  ETS + direct-growth XGBoost average,       (cron: 3rd of every month)
  80% ranges from backtest errors                   │
  forecasts/latest.json saved to GCS                ▼
        │                              [Cloud Run Job: inflation-data-pipeline]
        │                                (ingest.py → dbt run → dbt test → forecast_batch.py)
        ▼
[ml/serving/api.py]  ←── FastAPI, forecast table + FAISS index loaded from GCS (table re-checked hourly)
        │  /forecast            — 1–12 month CPI forecast with 80% range
        │  /price-ratio         — price ratio between two states, one category
        │  /trends              — historical CPI/PPI/OPR time series
        │  /category-leaderboard, /state-leaderboard — national/state rankings
        │  /forecast-accuracy   — live track record of published forecasts
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
Knowledge base (27 hand-curated entries)
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

**Evaluation:** a custom Gemini-based LLM-as-judge harness (built after the standard `ragas` library hit an unresolvable dependency conflict — see Known Issues below) scores answers against a versioned 14-question test set on five metrics:
- **faithfulness**: no facts beyond the retrieved context
- **answer relevancy**
- **context precision**
- **correctness** against the expected answer
- **refusal**: for out-of-scope questions, such as an exact future inflation figure or a fund recommendation, the bot must decline rather than invent

It runs in CI against the live API after every API deploy and fails the workflow below a 0.7 threshold. The judge model is configurable (`JUDGE_MODEL`), so answers can be graded by a different model from the one that wrote them. If `knowledge_base.py` changes, the API re-embeds it at startup, so the chatbot never answers from stale text.

---

## Key Results

> 📖 **[Case study: my model was 97% accurate, until I tested it properly](docs/case-study.md)**: how the original metrics turned out to be leakage, how an honest backtest showed the model losing to "no change", and how the redesign fixed it.

Forecasts are evaluated with a **rolling-origin backtest** (`ml/training/backtest.py`). For every month from Jan 2018 to Jul 2026, every model is retrained using only the data available at that date. It then forecasts 1–12 months ahead for all 16 states × 13 categories and is scored against what actually happened. A self-test corrupts all future data and confirms no model's forecast changes.

**Error relative to the naive forecast ("prices stay where they are")**: below 1.00 beats naive.

| Model | 1 month | 3 months | 6 months | 12 months |
|---|---:|---:|---:|---:|
| Naive (no change) | 1.00 | 1.00 | 1.00 | 1.00 |
| Trend continues (12-month drift) | 1.17 | 1.17 | 1.18 | 1.18 |
| ETS per series | 1.06 | 1.01 | 0.99 | 0.96 |
| **XGBoost, direct growth model** | 1.04 | 0.97 | 0.95 | 0.99 |
| **ETS + XGBoost average** | 1.03 | **0.97** | **0.94** | **0.94** |
| XGBoost, original design (currently deployed) | 2.23 | 1.37 | 1.22 | 1.11 |

Since 2022, the ETS + XGBoost average is 10–21% more accurate than naive (0.97 / 0.90 / 0.86 / 0.79). It also gets the direction of price change right 70–77% of the time. No model reliably beats naive one month ahead: about 40% of month-to-month category changes are exactly zero.

**What changed and why.** The original model reported MAE 0.73 and 97% directional accuracy, but those numbers didn't reflect real forecasting:
- **Leakage.** Its moving-average features included the month being predicted (fixed in the dbt mart).
- **Wrong test.** It was only tested one step ahead, while the app forecasts recursively up to 12 months.
- **Mixed-up direction metric.** Directional accuracy compared neighbouring rows from different series.

Under an honest backtest it was about 2× worse than naive. The redesign:
- predicts the **% change over each horizon**, with a separate model per horizon, instead of feeding predictions back in;
- tells the model **which state and category** each series is;
- adds **price stickiness**, **cross-state category trends** and **last year's same-period change** as features;
- trains on **absolute error**, which is what it is scored on.

**Live track record.** A backtest is still a reconstruction. So every month the pipeline also scores **every forecast it actually published** against the CPI DOSM has released since. It reports error against "no change" and how often the 80% range held, by horizon. Nothing in this scorecard could have been seen when the forecast was made. It's published to `forecasts/scorecard.json` and served at `GET /forecast-accuracy`, and it fills in as published forecasts reach their target months.

Full tables by period and metric: `docs/backtest_report.md`.

**RAG evaluation:** the earlier 10-question run (faithfulness 1.00, relevancy 0.95, precision 1.00) used questions closely matching the knowledge base, with no out-of-scope cases and no correctness check. The expanded 14-question set above replaces it; its results come from the CI run.

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
| Bank interest rates (commercial banks' base rate; this dataset has no OPR series) | BNM / data.gov.my | 5,700+ | Monthly |

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
│       │   ├── knowledge_base.py    # 27 hand-curated entries
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

### 3 — Train and publish forecasts
```bash
cd ml/training
python forecast_batch.py                      # trains, calibrates ranges, uploads forecasts/latest.json to GCS
# offline: python forecast_batch.py --source files --data-dir ../../data --local-out forecasts.json
```
This runs automatically every month as step 4 of the pipeline job. `train.py` is the legacy recursive model, kept only as the API's fallback.

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
# or serve a local forecast file: FORECAST_LOCAL_PATH=../training/forecasts.json uvicorn api:app --port 8081
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
terraform apply -var="alert_email=you@example.com"   # emails you if the monthly pipeline fails
```

All three images use **fully pinned** requirements (every transitive dependency), and every pull request builds and smoke-tests them (`.github/workflows/pr-build.yml`). An unpinned rebuild once broke the Streamlit app (pyarrow vs NumPy). The pinned pipeline environment was also how a missing `scikit-learn` in the pipeline image was caught.

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
- **Forecast ranges are pooled across categories.** The 80% ranges come from the last 36 months of backtest errors. In an out-of-sample test, ranges built from Aug 2020–Jul 2023 covered 81% of outcomes from Aug 2023 onward. Per-category ranges were tested but overfit (62–72% coverage), so every category currently gets the same % width at a given horizon.
- **Short-horizon forecasts are inherently weak.** Monthly CPI is sticky, and even the best model only modestly beats "no change" at 1–3 months. Forecasts should be presented with that uncertainty.

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