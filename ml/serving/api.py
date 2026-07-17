"""
FastAPI prediction service for Malaysia Inflation Intelligence.

Loads the trained XGBoost model from GCS on startup and serves
recursive multi-month forecasts via REST API.

Endpoints:
    GET  /health              — liveness check
    GET  /forecast             — multi-month recursive CPI forecast
    GET  /price-ratio          — price ratio between two states for one category
    GET  /states               — list of available states
    GET  /categories           — list of CPI categories
"""

import os
import logging
import tempfile
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage, bigquery
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "rag"))
from retriever import load_index_from_gcs, retrieve
from google import genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ID  = os.getenv("GCP_PROJECT_ID", "modular-rush-495414-i2")
GCS_BUCKET  = f"{PROJECT_ID}-ml-artifacts"
MODEL_PATH  = os.getenv("MODEL_GCS_PATH", "models/cpi_forecast_20260619_194024.joblib")

app = FastAPI(
    title="Malaysia Inflation Intelligence API",
    description="Predicts personal CPI inflation for Malaysian citizens",
    version="1.0.0",
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and explainer — loaded once at startup
model      = None
bq_client  = None

# ── constants ────────────────────────────────────────────────────────────────
STATES = [
    "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
    "Pahang", "Perak", "Perlis", "Pulau Pinang", "Sabah",
    "Sarawak", "Selangor", "Terengganu",
    "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"
]

CATEGORIES = {
    "01": "Food & Non-Alcoholic Beverages",
    "02": "Alcoholic Beverages & Tobacco",
    "03": "Clothing & Footwear",
    "04": "Housing, Water, Electricity & Gas",
    "05": "Furnishings & Household Equipment",
    "06": "Health",
    "07": "Transport",
    "08": "Information & Communication",
    "09": "Recreation, Sport & Culture",
    "10": "Education",
    "11": "Restaurants & Accommodation",
    "12": "Insurance & Financial Services",
    "13": "Personal Care & Miscellaneous",
}

FEATURE_COLS = [
    "cpi_lag_1", "cpi_lag_3", "cpi_lag_6", "cpi_lag_12",
    "cpi_ma_3", "cpi_ma_12",
    "ppi_index", "ppi_lag_1", "ppi_lag_2", "ppi_lag_3",
    "opr_rate",
]


def load_model_from_gcs() -> None:
    """Load trained XGBoost model from GCS into global model variable."""
    global model

    logger.info(f"Loading model from gs://{GCS_BUCKET}/{MODEL_PATH}")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(MODEL_PATH)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        blob.download_to_filename(f.name)
        model = joblib.load(f.name)

    logger.info("Model loaded successfully")


def get_latest_features(state: str, division: str) -> dict:
    """
    Pull the most recent feature row for a given state and division
    from BigQuery mart_cpi_features.

    This gives the model real current data to forecast from rather
    than requiring the app to pass all 11 features manually.
    """
    global bq_client
    if bq_client is None:
        bq_client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        select
            cpi_lag_1, cpi_lag_3, cpi_lag_6, cpi_lag_12,
            cpi_ma_3, cpi_ma_12,
            ppi_index, ppi_lag_1, ppi_lag_2, ppi_lag_3,
            opr_rate, cpi_index, date
        from `{PROJECT_ID}.dbt_.mart_cpi_features`
        where state = @state
          and division = @division
          and cpi_lag_1 is not null
        order by date desc
        limit 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("state",    "STRING", state),
            bigquery.ScalarQueryParameter("division", "STRING", division),
        ]
    )

    rows = list(bq_client.query(query, job_config=job_config).result())

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for state={state}, division={division}"
        )

    return dict(rows[0])


@app.on_event("startup")
async def startup_event():
    """Load model and RAG index when API starts up."""
    load_model_from_gcs()
    load_index_from_gcs()


# ── Core prediction logic ─────────────────────────────────────────────────────

def get_feature_history(
        state: str,
        division: str,
        months_back: int = 15,
) -> list[dict]:
    """
    Pull the last N months of real feature data from BigQuery,
    ordered oldest to newest. This is the real paper strip
    we slide the prediction window across.
    """
    global bq_client
    if bq_client is None:
        bq_client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        select
            date, cpi_index,
            cpi_lag_1, cpi_lag_3, cpi_lag_6, cpi_lag_12,
            cpi_ma_3, cpi_ma_12,
            ppi_index, ppi_lag_1, ppi_lag_2, ppi_lag_3,
            opr_rate
        from `{PROJECT_ID}.dbt_.mart_cpi_features`
        where state = @state
            and division = @division
            and cpi_lag_1 is not null
        order by date desc
        limit @months_back
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("state", "STRING", state),
            bigquery.ScalarQueryParameter("division", "STRING", division),
            bigquery.ScalarQueryParameter("months_back", "INT64", months_back),
        ]
    )

    rows = list(bq_client.query(query, job_config=job_config).result())

    if not rows:
        return []

    history = [dict(row) for row in rows]
    history.reverse()

    # Forward-fill OPR nulls — BNM publishes with a 1-2 month lag,
    # so the most recent months are often null. Carry the last
    # known real rate forward, matching the training preprocessing.
    last_known_opr = None
    for row in history:
        if row.get("opr_rate") is not None:
            last_known_opr = row["opr_rate"]
        else:
            row["opr_rate"] = last_known_opr

    return history

def build_features_from_history(working_history: list[dict]) -> np.ndarray:
    """
    Build one model input row by reading backwards from the end
    of working_history - the plucking boxes off the strip step.
    
    working_history[-1] is the most recent value (real or predicted).
    working_history[-2] is one before that. And so on.
    """
    n = len(working_history)

    def index_at(steps_back: int) -> float:
        """Get the cpi_index value N steps back from the end."""
        pos = n - steps_back
        if pos < 0:
            # Not enough history - fall back to the oldest value we have
            pos = 0
        return working_history[pos]["cpi_index"]
    
    cpi_lag_1  = index_at(1)
    cpi_lag_3  = index_at(3)
    cpi_lag_6  = index_at(6)
    cpi_lag_12 = index_at(12)

    # Rolling average - take the last 3 (or 12) values in the strip
    last_3  = [working_history[i]["cpi_index"] for i in range(max(0, n-3), n)]
    last_12 = [working_history[i]["cpi_index"] for i in range(max(0, n-12), n)]
    cpi_ma_3  = sum(last_3) / len(last_3)
    cpi_ma_12 = sum(last_12) / len(last_12)

    # PPI and OPR — these are NOT part of the recursive loop.
    # We hold them constant at their last known real value, since
    # we have no model that forecasts PPI or OPR independently.
    latest = working_history[-1]
    ppi_index = latest.get("ppi_index", 0.0) or 0.0
    ppi_lag_1 = latest.get("ppi_lag_1", 0.0) or 0.0
    ppi_lag_2 = latest.get("ppi_lag_2", 0.0) or 0.0
    ppi_lag_3 = latest.get("ppi_lag_3", 0.0) or 0.0
    opr_rate  = latest.get("opr_rate", 0.0) or 0.0

    row = [
        cpi_lag_1, cpi_lag_3, cpi_lag_6, cpi_lag_12,
        cpi_ma_3, cpi_ma_12,
        ppi_index, ppi_lag_1, ppi_lag_2, ppi_lag_3,
        opr_rate,
    ]

    return np.array([row])

def append_prediction(
    working_history: list[dict],
    predicted_index: float,
) -> list[dict]:
    """
    Tape a new prediction onto the end of the strip.

    Predicted boxes copy PPI/OPR from the previous box, since
    we hold those constant throughout the recursive loop.
    """
    previous_box = working_history[-1]

    new_row = {
        "cpi_index": predicted_index,
        "ppi_index": previous_box.get("ppi_index", 0.0),
        "ppi_lag_1": previous_box.get("ppi_lag_1", 0.0),
        "ppi_lag_2": previous_box.get("ppi_lag_2", 0.0),
        "ppi_lag_3": previous_box.get("ppi_lag_3", 0.0),
        "opr_rate":  previous_box.get("opr_rate", 0.0),
    }

    return working_history + [new_row]

def recursive_forecast(
    state: str,
    division: str,
    horizon_months: int,
) -> list[dict]:
    """
    Forecast N months ahead by calling the model repeatedly,
    rebuilding lag features from previous predictions each step.

    Returns a list of monthly predictions with a widening
    confidence interval to reflect compounding uncertainty.
    """
    history = get_feature_history(state, division, months_back=15)

    if len(history) < 13:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough history for {state}/{division} to forecast"
        )

    latest_real_date = str(history[-1]["date"])

    working_history = history.copy()
    forecasts = []

    # Base uncertainty per step — grows the further we forecast,
    # modelling the compounding error we discussed. This is a
    # simple linear widening; a more rigorous approach would use
    # the residual std from walk-forward CV at each horizon.
    base_uncertainty_pct = 0.5

    for step in range(1, horizon_months + 1):
        X = build_features_from_history(working_history)
        predicted_index = float(model.predict(X)[0])

        uncertainty_pct = base_uncertainty_pct * step
        lower_bound = predicted_index * (1 - uncertainty_pct / 100)
        upper_bound = predicted_index * (1 + uncertainty_pct / 100)

        forecasts.append({
            "step": step,
            "predicted_index": round(predicted_index, 3),
            "lower_bound": round(lower_bound, 3),
            "upper_bound": round(upper_bound, 3),
            "confidence_band_pct": round(uncertainty_pct, 2),
        })

        working_history = append_prediction(working_history, predicted_index)

    return {
        "forecasts": forecasts,
        "latest_real_date": latest_real_date,
    }
     
# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness check — returns ok if model is loaded."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "project": PROJECT_ID,
    }


@app.get("/states")
async def list_states():
    """Return list of all available Malaysian states."""
    return {"states": STATES}


@app.get("/categories")
async def list_categories():
    """Return list of all CPI categories with division codes."""
    return {
        "categories": [
            {"code": code, "name": name}
            for code, name in CATEGORIES.items()
        ]
    }


@app.get("/forecast")
@limiter.limit("30/minute")
async def forecast(
    request: Request,
    state: str,
    division: str,
    horizon_months: int = 6,
):
    """
    Multi-month recursive CPI forecast for one state and category.

    Example:
        GET /forecast?state=Kedah&division=01&horizon_months=6

    Returns a list of monthly predictions with widening
    confidence bands.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown state: {state}")

    if division not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown division code: {division}")

    if horizon_months not in (3, 6, 12):
        raise HTTPException(
            status_code=400,
            detail="horizon_months must be 3, 6, or 12"
        )

    result = recursive_forecast(state, division, horizon_months)

    return {
        "state": state,
        "division": division,
        "category_name": CATEGORIES.get(division, division),
        "horizon_months": horizon_months,
        "forecast": result["forecasts"],
        "latest_real_date": result["latest_real_date"],
    }

@app.get("/price-ratio")
@limiter.limit("30/minute")
async def price_ratio(
    request: Request,
    origin_state: str,
    destination_state: str,
    division: str,
):
    """
    Returns the current price ratio between two states for one category.
    Used to auto-scale a user's spending when comparing states.
    """
    if origin_state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown origin state: {origin_state}")

    if destination_state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown destination state: {destination_state}")

    if division not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown division code: {division}")

    try:
        origin_features = get_latest_features(origin_state, division)
        dest_features = get_latest_features(destination_state, division)
    except HTTPException:
        raise

    origin_index = origin_features.get("cpi_index")
    dest_index = dest_features.get("cpi_index")

    if not origin_index or not dest_index:
        raise HTTPException(
            status_code=404,
            detail="Could not find CPI data for one of the states"
        )

    ratio = dest_index / origin_index

    return {
        "origin_state": origin_state,
        "destination_state": destination_state,
        "division": division,
        "origin_index": round(origin_index, 2),
        "destination_index": round(dest_index, 2),
        "ratio": round(ratio, 4),
        "pct_difference": round((ratio - 1) * 100, 2),
    }

@app.get("/trends")
@limiter.limit("30/minute")
async def trends(
    request: Request,
    state: str,
    division: str,
    months_back: int =60,
):
    """
    Returns a raw historical CPI/PPI/OPR time series for charting - 
    NOT a forecast, hust the real recorded data over time.
    
    Example:
        GET /trends?state=Selangor&division=01&months_back=60
    """
    if state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown state: {state}")
    
    if division not in CATEGORIES:
        raise HTTPException(status_code=400, detatil=f"Unknown division code: {division}")
    
    if months_back not in (24, 36, 60, 120):
        raise HTTPException(
            status_code=400,
            detail="months_back must be 24,36, 60, or 120"
        )
    
    history = get_feature_history(state, division, months_back=months_back)

    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data found for {state}/{division}"
        )
    
    return {
        "state": state,
        "division": division,
        "category_name": CATEGORIES.get(division, division),
        "months_back": months_back,
        "series": [
            {
                "date": str(row["date"]),
                "cpi_index": row["cpi_index"],
                "cpi_ma_3": row["cpi_ma_3"],
                "ppi_index": row["ppi_index"],
                "opr_rate": row["opr_rate"],
            }
            for row in history
        ],
    }

@app.get("/income-snapshot")
@limiter.limit("30/minute")
async def income_snapshot(request: Request, state: str):
    """
    Returns the HIES 2022 income/expenditure snapshot for one state.
    This is a static survey snapshot, not a monthly time series.
    """
    if state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown sate: {state}")
    
    global bq_client
    if bq_client is None:
        bq_client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        select
            income_median, income_mean, expenditure_mean,
            gini, poverty_rate, income_band
        from `{PROJECT_ID}.dbt_.mart_income_bands`
        where state = @state
        limit 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("state", "STRING", state),
        ]
    )

    rows = list(bq_client.query(query, job_config=job_config).result())

    if not rows:
        raise HTTPException(status_code=404, detail=f"No income data found for {state}")
    
    return dict(rows[0])

@app.get("/headline-inflation")
@limiter.limit("30/minute")
async def headline_inflation(request: Request):
    """
    Returns the current national headline CPI year-on-year change,
    computed from the official raw.cpi_national table.

    This replaces the previously hardcoded 1.5% reference value
    used throughout the app for "official inflation" comparisons.
    """
    global bq_client
    if bq_client is None:
        bq_client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        select date, index
        from `{PROJECT_ID}.raw.cpi_national`
        where division = 'overall'
        order by date desc
        limit 13
    """

    rows = list(bq_client.query(query).result())

    if len(rows) < 13:
        raise HTTPException(
            status_code=404,
            detail="Not enough national CPI history to compute headline inflation"
        )

    rows = [dict(r) for r in rows]
    latest = rows[0]
    year_ago = rows[12]

    yoy_change_pct = round(
        (latest["index"] - year_ago["index"]) / year_ago["index"] * 100,
        2
    )

    return {
        "headline_inflation_pct": yoy_change_pct,
        "latest_date": str(latest["date"]),
        "compared_to_date": str(year_ago["date"]),
    }

class AskRequest(BaseModel):
    question: str
    state: str
    division: str
    horizon_months: int = 6


GENERATION_MODEL = "gemini-3.1-flash-lite"

_ask_genai_client = None


@app.post("/ask")
@limiter.limit("15/minute")
async def ask(request: Request, body: AskRequest):
    """
    Answers a user's question about their inflation forecast,
    grounded in retrieved domain knowledge and their actual live
    forecast numbers. The LLM never invents figures — it only
    explains numbers this API already computed.
    """
    global _ask_genai_client
    if _ask_genai_client is None:
        _ask_genai_client = genai.Client()

    if body.state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown state: {body.state}")
    if body.division not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown division code: {body.division}")

    # Step 1 — retrieve relevant domain knowledge
    retrieved = retrieve(body.question, top_k=3)
    knowledge_context = "\n\n".join([f"- {r['text']}" for r in retrieved])

    # Step 2 — get the user's actual live forecast (never invented)
    forecast_result = recursive_forecast(body.state, body.division, body.horizon_months)
    category_name = CATEGORIES.get(body.division, body.division)
    forecast_summary = (
        f"Category: {category_name} in {body.state}\n"
        f"Forecast horizon: {body.horizon_months} months\n"
        f"Latest real data date: {forecast_result['latest_real_date']}\n"
        f"Month-by-month forecast: {forecast_result['forecasts']}"
    )

    # Step 3 — build a grounded prompt
    system_instruction = (
        "You are a helpful assistant explaining Malaysian inflation forecasts to everyday users. "
        "You must ONLY use the domain knowledge and forecast data provided below — never invent "
        "numbers, dates, or facts not present in this context. If the question cannot be answered "
        "from the provided context, say so honestly rather than guessing. Keep answers concise, "
        "in plain language, and directly relevant to the user's question."
    )

    prompt = f"""DOMAIN KNOWLEDGE:
{knowledge_context}

LIVE FORECAST DATA:
{forecast_summary}

USER QUESTION:
{body.question}

Answer the user's question using only the domain knowledge and forecast data above."""

    # Step 4 — generate the grounded answer
    response = _ask_genai_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config={
            "system_instruction": system_instruction,
            "max_output_tokens": 300,
            "temperature": 0.3,
        },
    )

    return {
        "question": body.question,
        "answer": response.text,
        "retrieved_context": [r["id"] for r in retrieved],
        "based_on_forecast": {
            "state": body.state,
            "division": body.division,
            "category_name": category_name,
            "horizon_months": body.horizon_months,
        },
    }

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)