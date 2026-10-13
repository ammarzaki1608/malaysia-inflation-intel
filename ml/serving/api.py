"""
FastAPI prediction service for Malaysia Inflation Intelligence.

Serves CPI forecasts precomputed monthly by ml/training/forecast_batch.py
(ETS + direct-growth XGBoost average, with 80% ranges calibrated on backtest
errors). The table is loaded from GCS at startup and re-checked hourly.
If it is unavailable, the API falls back to the legacy recursive XGBoost.

Endpoints:
    GET  /health              — liveness check
    GET  /forecast             — multi-month recursive CPI forecast
    GET  /price-ratio          — price ratio between two states for one category
    GET  /states               — list of available states
    GET  /categories           — list of CPI categories
"""

import os
import json
import time
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
MODEL_PATH  = os.getenv("MODEL_GCS_PATH", "models/cpi_forecast_20260619_194024.joblib")  # legacy fallback
FORECAST_GCS_PATH   = os.getenv("FORECAST_GCS_PATH", "forecasts/latest.json")
FORECAST_LOCAL_PATH = os.getenv("FORECAST_LOCAL_PATH")      # for local testing only
FORECAST_REFRESH_S  = int(os.getenv("FORECAST_REFRESH_SECONDS", "3600"))

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

# Precomputed forecast table: {(state, division): [row for h=1..12]}
forecast_rows: dict = {}
forecast_meta: dict = {}
_forecast_generation = None
_forecast_checked_at = 0.0

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


def load_forecast_table(force: bool = False) -> bool:
    """
    Load the precomputed forecast JSON (from FORECAST_LOCAL_PATH if set, else GCS).
    Skips the download when the GCS object's generation has not changed.
    Returns True if a table is loaded after the call.
    """
    global forecast_rows, forecast_meta, _forecast_generation, _forecast_checked_at
    _forecast_checked_at = time.time()

    if FORECAST_LOCAL_PATH:
        with open(FORECAST_LOCAL_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        generation = os.path.getmtime(FORECAST_LOCAL_PATH)
    else:
        blob = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET).blob(FORECAST_GCS_PATH)
        blob.reload()
        generation = blob.generation
        if not force and generation == _forecast_generation and forecast_rows:
            return True
        payload = json.loads(blob.download_as_text())

    rows: dict = {}
    for r in payload["rows"]:
        rows.setdefault((r["state"], r["division"]), []).append(r)
    for v in rows.values():
        v.sort(key=lambda r: r["h"])

    forecast_rows, forecast_meta, _forecast_generation = rows, payload["meta"], generation
    logger.info(f"Loaded forecast table {forecast_meta.get('version')} "
                f"(base {forecast_meta.get('base_date')}, {len(rows)} series)")
    return bool(forecast_rows)


def maybe_refresh_forecast_table() -> None:
    """Re-check for a newer table at most every FORECAST_REFRESH_S seconds; never raises."""
    if time.time() - _forecast_checked_at < FORECAST_REFRESH_S:
        return
    try:
        load_forecast_table()
    except Exception as e:
        logger.warning(f"Forecast table refresh failed, keeping current: {e}")


def get_forecast(state: str, division: str, horizon_months: int) -> dict:
    """
    Forecast for one series. Uses the precomputed table; falls back to the
    legacy recursive model only if the table has no entry for this series.
    """
    maybe_refresh_forecast_table()
    rows = forecast_rows.get((state, division))
    if rows:
        steps = [
            {
                "step": r["h"],
                "target_date": r["target_date"],
                "predicted_index": r["predicted_index"],
                "lower_bound": r["lower_bound"],
                "upper_bound": r["upper_bound"],
                "confidence_band_pct": r["confidence_band_pct"],
                "base_index": r["base_index"],
            }
            for r in rows if r["h"] <= horizon_months
        ]
        return {
            "forecasts": steps,
            "latest_real_date": rows[0]["base_date"],
            "latest_real_index": rows[0]["base_index"],
            "past_12m_change_pct": rows[0].get("past_12m_change_pct"),
            "model_version": forecast_meta.get("version"),
            "interval_level": forecast_meta.get("interval_level"),
            "source": "precomputed",
        }

    if model is None:
        try:
            load_model_from_gcs()
        except Exception as e:
            logger.error(f"Legacy model unavailable: {e}")
            raise HTTPException(status_code=503, detail="Forecasts are not available yet.")
    result = recursive_forecast(state, division, horizon_months)
    result["source"] = "legacy_recursive"
    return result


@app.on_event("startup")
async def startup_event():
    """Load the forecast table (or the legacy model as a fallback) and the RAG index."""
    try:
        load_forecast_table(force=True)
    except Exception as e:
        logger.warning(f"Forecast table not loaded ({e}); falling back to legacy model")
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
        "forecast_table_loaded": bool(forecast_rows),
        "forecast_version": forecast_meta.get("version"),
        "forecast_base_date": forecast_meta.get("base_date"),
        "legacy_model_loaded": model is not None,
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

    Returns monthly predictions with 80% ranges calibrated on backtest
    errors. Each step carries base_index (the latest real CPI) so clients can
    measure change from today rather than from the first forecast month.
    """
    if state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown state: {state}")

    if division not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown division code: {division}")

    if horizon_months not in (3, 6, 12):
        raise HTTPException(
            status_code=400,
            detail="horizon_months must be 3, 6, or 12"
        )

    return _forecast_payload(state, division, horizon_months)


def _forecast_payload(state: str, division: str, horizon_months: int) -> dict:
    result = get_forecast(state, division, horizon_months)
    return {
        "state": state,
        "division": division,
        "category_name": CATEGORIES.get(division, division),
        "horizon_months": horizon_months,
        "forecast": result["forecasts"],
        "latest_real_date": result["latest_real_date"],
        "latest_real_index": result.get("latest_real_index"),
        "past_12m_change_pct": result.get("past_12m_change_pct"),
        "model_version": result.get("model_version"),
        "interval_level": result.get("interval_level"),
        "source": result["source"],
    }


@app.get("/forecasts")
@limiter.limit("30/minute")
async def forecasts_bulk(
    request: Request,
    state: str,
    divisions: str,
    horizon_months: int = 6,
):
    """
    Forecasts for several categories of one state in ONE request.

    The app needs every category a user spends on (up to 13, or 26 for the
    Relocation Planner). Calling /forecast per category burned through the
    per-IP rate limit after a click or two — and all app users share the app
    server's IP. Example: GET /forecasts?state=Selangor&divisions=01,04,07&horizon_months=6
    """
    if state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown state: {state}")
    if horizon_months not in (3, 6, 12):
        raise HTTPException(status_code=400, detail="horizon_months must be 3, 6, or 12")
    divs = [d.strip() for d in divisions.split(",") if d.strip()]
    bad = [d for d in divs if d not in CATEGORIES]
    if not divs or bad:
        raise HTTPException(status_code=400, detail=f"Unknown or missing division codes: {bad or divisions}")

    out, errors = {}, {}
    for d in dict.fromkeys(divs):          # de-duplicate, keep order
        try:
            out[d] = _forecast_payload(state, d, horizon_months)
        except HTTPException as e:
            errors[d] = e.detail
    return {"state": state, "horizon_months": horizon_months, "forecasts": out, "errors": errors}

@app.get("/price-ratio")
@limiter.limit("30/minute")
async def price_ratio(
    request: Request,
    origin_state: str,
    destination_state: str,
    division: str,
):
    """
    DEPRECATED — no longer used by the app.

    Returns destination CPI index / origin CPI index. Each state's index measures
    price CHANGE relative to its own base period, not price LEVEL, so this ratio
    does not say which state is more expensive. Kept only for backwards
    compatibility; the Relocation Planner now uses HIES household expenditure.
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

@app.get("/category-leaderboard")
@limiter.limit("30/minute")
async def category_leaderboard(request: Request):
    """
    Returns year-on-year % change for every CPI category, nationally,
    ranked fastest-rising first. Used for Data Explorer's overview section.
    """
    global bq_client
    if bq_client is None:
        bq_client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        select division, date, index
        from `{PROJECT_ID}.raw.cpi_national`
        where division != 'overall'
        order by division, date desc
    """

    rows = list(bq_client.query(query).result())

    # Group by division, keep the latest and ~12-months-ago value for each
    by_division = {}
    for row in rows:
        by_division.setdefault(row["division"], []).append(row)

    results = []
    for division, entries in by_division.items():
        if len(entries) < 13:
            continue
        latest = entries[0]
        year_ago = entries[12]
        pct_change = round((latest["index"] - year_ago["index"]) / year_ago["index"] * 100, 2)
        results.append({
            "division": division,
            "category_name": CATEGORIES.get(division, division),
            "yoy_change_pct": pct_change,
        })

    results.sort(key=lambda x: x["yoy_change_pct"], reverse=True)

    return {"categories": results}

@app.get("/state-leaderboard")
@limiter.limit("30/minute")
async def state_leaderboard(request: Request, division: str):
    """
    Year-on-year % change in one category's CPI for every state, fastest-rising
    first. (Ranking by index LEVEL would be meaningless: each state's index
    measures change from its own base period, not how expensive the state is.)
    """
    if division not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown division code: {division}")

    maybe_refresh_forecast_table()
    results = [
        {"state": st_, "yoy_change_pct": rows[0]["past_12m_change_pct"],
         "cpi_index": rows[0]["base_index"], "latest_date": rows[0]["base_date"]}
        for (st_, div), rows in forecast_rows.items()
        if div == division and rows and rows[0].get("past_12m_change_pct") is not None
    ]

    if not results:  # forecast table unavailable -> compute from the mart
        global bq_client
        if bq_client is None:
            bq_client = bigquery.Client(project=PROJECT_ID)
        query = f"""
            select state, cpi_index, cpi_lag_12, date
            from `{PROJECT_ID}.dbt_.mart_cpi_features`
            where division = @division
              and cpi_index is not null
            qualify row_number() over (partition by state order by date desc) = 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("division", "STRING", division)]
        )
        for r in bq_client.query(query, job_config=job_config).result():
            if r["cpi_lag_12"]:
                results.append({
                    "state": r["state"],
                    "yoy_change_pct": round((r["cpi_index"] / r["cpi_lag_12"] - 1) * 100, 2),
                    "cpi_index": r["cpi_index"], "latest_date": str(r["date"]),
                })

    results.sort(key=lambda x: x["yoy_change_pct"], reverse=True)
    return {"division": division, "category_name": CATEGORIES.get(division, division), "states": results}

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
    state: str | None = None
    division: str | None = None
    horizon_months: int = 6


GENERATION_MODEL = "gemini-3.1-flash-lite"
_ask_genai_client = None

# Keywords suggesting the question is about the user's OWN specific
# numbers, not a general "how does this app work" question. This is a
# lightweight heuristic — not perfect, but free and fast, avoiding an
# extra LLM call just to classify intent.
PERSONAL_QUESTION_SIGNALS = [
    "my forecast", "my spending", "my category", "my leak", "my surplus",
    "my number", "why is my", "why does my", "this category", "this forecast",
    "unjuran saya", "perbelanjaan saya", "kategori saya", "bocor saya",
]


def _needs_live_forecast(question: str) -> bool:
    q = question.lower()
    return any(signal in q for signal in PERSONAL_QUESTION_SIGNALS)


@app.post("/ask")
@limiter.limit("15/minute")
async def ask(request: Request, body: AskRequest):
    """
    Answers a question about the app in general, OR about the user's
    specific forecast — grounded in retrieved domain knowledge, and in
    live forecast data only when the question actually needs it.
    """
    global _ask_genai_client
    if _ask_genai_client is None:
        _ask_genai_client = genai.Client()

    # Step 1 — retrieve relevant domain knowledge (always happens)
    retrieved = retrieve(body.question, top_k=3)
    knowledge_context = "\n\n".join([f"- {r['text']}" for r in retrieved])

    # Step 2 — only fetch live forecast data if the question actually
    # seems to be about the user's own numbers, and state/division
    # were actually provided
    forecast_summary = "Not applicable — this question does not require the user's live forecast data."
    used_live_data = False

    if body.state and body.division and _needs_live_forecast(body.question):
        if body.state not in STATES:
            raise HTTPException(status_code=400, detail=f"Unknown state: {body.state}")
        if body.division not in CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Unknown division code: {body.division}")

        forecast_result = get_forecast(body.state, body.division, body.horizon_months)
        category_name = CATEGORIES.get(body.division, body.division)
        forecast_summary = (
            f"Category: {category_name} in {body.state}\n"
            f"Forecast horizon: {body.horizon_months} months\n"
            f"Latest real data date: {forecast_result['latest_real_date']}\n"
            f"Month-by-month forecast: {forecast_result['forecasts']}"
        )
        used_live_data = True

    # Step 3 — build a grounded prompt
    system_instruction = (
        "You are a helpful assistant for a Malaysian personal inflation forecasting app. "
        "You must ONLY use the domain knowledge and forecast data provided below — never invent "
        "numbers, dates, or facts not present in this context. If the question cannot be answered "
        "from the provided context, say so honestly rather than guessing. Keep answers concise, "
        "in plain language. If the question is a general question about how the app works, answer "
        "from the domain knowledge alone without needing specific numbers."
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
        "used_live_forecast_data": used_live_data,
    }

class PrescriptiveRequest(BaseModel):
    context_type: str  # "personal_calculator" or "relocation_planner"
    summary_data: dict


@app.post("/prescriptive-insight")
@limiter.limit("15/minute")
async def prescriptive_insight(request: Request, body: PrescriptiveRequest):
    """
    Generates a short, action-oriented suggestion grounded in the
    user's own report numbers. Deliberately stays in general-budgeting
    territory — never recommends specific financial products, insurers,
    or investments, since this app is not a licensed financial advisor.
    """
    global _ask_genai_client
    if _ask_genai_client is None:
        _ask_genai_client = genai.Client()

    system_instruction = (
        "You are a budgeting assistant for a Malaysian personal inflation app. "
        "Generate ONE short paragraph (2-3 sentences) offering a general, actionable "
        "budgeting suggestion based on the data provided. "
        "STRICT RULES: "
        "1. Never recommend specific financial products, insurers, banks, or investments. "
        "2. Never use directive language like 'you should switch to X' or 'invest in Y'. "
        "3. Only suggest general actions like reviewing a specific expense category, "
        "building a buffer, or planning ahead — never specific financial products. "
        "4. Stay grounded only in the numbers provided — never invent figures. "
        "5. Keep it concise and practical, not alarmist."
    )

    if body.context_type == "personal_calculator":
        prompt = f"""REPORT DATA:
Biggest leak category: {body.summary_data.get('biggest_leak_name')} (rising {body.summary_data.get('biggest_leak_pct')}%)
Best category: {body.summary_data.get('biggest_saving_name')}
Current monthly surplus: RM{body.summary_data.get('current_surplus')}
Projected monthly surplus: RM{body.summary_data.get('projected_surplus')}
Horizon: {body.summary_data.get('horizon_months')} months

Generate a short, general budgeting suggestion based on this report."""
    else:
        prompt = f"""RELOCATION COMPARISON DATA:
Origin state: {body.summary_data.get('origin_state')}
Destination state: {body.summary_data.get('destination_state')}
Monthly cost difference: RM{body.summary_data.get('cost_diff_rm')}
Recommended minimum salary: RM{body.summary_data.get('salary_needed')}
Horizon: {body.summary_data.get('horizon_months')} months

Generate a short, general suggestion for someone considering this move."""

    response = _ask_genai_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config={
            "system_instruction": system_instruction,
            "max_output_tokens": 150,
            "temperature": 0.4,
        },
    )

    return {"insight": response.text}


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)