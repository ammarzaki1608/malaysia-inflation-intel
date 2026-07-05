"""
FastAPI prediction service for Malaysia Inflation Intelligence.

Loads the trained XGBoost model from GCS on startup and serves
predictions with SHAP explanations via REST API.

Endpoints:
    GET  /health              — liveness check
    POST /predict             — single series prediction
    POST /compare             — two-state comparison (relocation planner)
    GET  /states              — list of available states
    GET  /categories          — list of CPI categories
"""

import os
import io
import json
import logging
import tempfile
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage, bigquery

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and explainer — loaded once at startup
model      = None
explainer  = None
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
    global model, explainer

    logger.info(f"Loading model from gs://{GCS_BUCKET}/{MODEL_PATH}")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(MODEL_PATH)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        blob.download_to_filename(f.name)
        model = joblib.load(f.name)

    explainer = shap.TreeExplainer(model)
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


def get_income_data(state: str) -> dict:
    """Pull HIES income data for a given state from BigQuery."""
    global bq_client
    if bq_client is None:
        bq_client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        select
            income_median,
            income_mean,
            expenditure_mean,
            gini,
            poverty_rate,
            income_band
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
        return {}

    return dict(rows[0])


@app.on_event("startup")
async def startup_event():
    """Load model when API starts up."""
    load_model_from_gcs()

# ── Pydantic models — define the shape of API requests and responses ─────────

class SpendingProfile(BaseModel):
    """
    A user's spending weights across CPI categories.
    Keys are division codes ('01' to '13').
    Values are percentages that must sum to 100.
    """
    weights: dict[str, float]
    state: str

    def validate_weights(self):
        total = sum(self.weights.values())
        if abs(total - 100.0) > 1.0:
            raise HTTPException(
                status_code=400,
                detail=f"Spending weights must sum to 100. Got {total:.1f}"
            )
        for div in self.weights:
            if div not in CATEGORIES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown division code: {div}"
                )


class PredictionResult(BaseModel):
    state: str
    personal_inflation_rate: float
    headline_cpi_change: float
    category_breakdown: list[dict]
    shap_explanation: list[dict]
    income_context: dict
    latest_date: str


class ComparisonRequest(BaseModel):
    """Request for relocation planner — compare two states."""
    origin_state: str
    destination_state: str
    origin_weights: dict[str, float]
    destination_weights: dict[str, float]


class ComparisonResult(BaseModel):
    origin: PredictionResult
    destination: PredictionResult
    cost_difference_pct: float
    salary_needed: float
    summary: str


# ── Core prediction logic ─────────────────────────────────────────────────────

def predict_category(state: str, division: str) -> dict:
    """
    Generate a forecast for one CPI category in one state.

    Pulls latest features from BigQuery, runs XGBoost prediction,
    and computes SHAP values for explainability.

    Returns a dict with predicted value, current value, change,
    and SHAP feature importance.
    """
    # Get latest real features from BigQuery
    features = get_latest_features(state, division)

    # Build feature array in correct order
    X = np.array([[features.get(col, 0.0) or 0.0 for col in FEATURE_COLS]])

    # Generate prediction
    predicted_index = float(model.predict(X)[0])
    current_index   = float(features.get("cpi_index", predicted_index))

    # Calculate percentage change
    pct_change = ((predicted_index - current_index) / current_index * 100
                  if current_index != 0 else 0.0)

    # Compute SHAP values for this prediction
    shap_vals   = explainer.shap_values(X)[0]
    shap_output = [
        {"feature": col, "value": float(val), "feature_value": float(X[0][i])}
        for i, (col, val) in enumerate(zip(FEATURE_COLS, shap_vals))
    ]
    shap_output.sort(key=lambda x: abs(x["value"]), reverse=True)

    return {
        "division":        division,
        "category_name":   CATEGORIES.get(division, division),
        "current_index":   round(current_index, 2),
        "predicted_index": round(predicted_index, 2),
        "pct_change":      round(pct_change, 3),
        "shap_values":     shap_output[:5],  # top 5 drivers only
        "latest_date":     str(features.get("date", "")),
    }

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

    return forecasts

def calculate_personal_inflation(
    state: str,
    weights: dict[str, float],
) -> PredictionResult:
    """
    Calculate personal inflation rate for a given state
    and spending profile.

    For each category in weights, gets a CPI forecast and
    weights it by the user's spending percentage.
    """
    category_results = []
    weighted_inflation = 0.0
    headline_change    = 0.0
    latest_date        = ""

    # Equal weight baseline for headline comparison
    n_cats = len(weights)

    for division, weight in weights.items():
        if weight <= 0:
            continue

        try:
            result = predict_category(state, division)
            result["weight"] = weight
            result["weighted_contribution"] = round(
                result["pct_change"] * weight / 100, 4
            )
            category_results.append(result)

            # Add to personal weighted inflation
            weighted_inflation += result["pct_change"] * (weight / 100)

            # Add to equal-weight headline (for comparison)
            headline_change += result["pct_change"] * (1 / n_cats)

            if not latest_date:
                latest_date = result["latest_date"]

        except Exception as e:
            logger.warning(f"Skipping {state}/{division}: {e}")
            continue

    # Sort categories by absolute weighted contribution
    category_results.sort(
        key=lambda x: abs(x["weighted_contribution"]),
        reverse=True
    )

    # Aggregate SHAP across all categories
    shap_agg: dict[str, float] = {}
    for cat in category_results:
        for sv in cat.get("shap_values", []):
            feat = sv["feature"]
            shap_agg[feat] = shap_agg.get(feat, 0.0) + abs(sv["value"])

    shap_summary = sorted(
        [{"feature": k, "importance": round(v, 4)}
         for k, v in shap_agg.items()],
        key=lambda x: x["importance"],
        reverse=True
    )

    income = get_income_data(state)

    return PredictionResult(
        state=state,
        personal_inflation_rate=round(weighted_inflation, 3),
        headline_cpi_change=round(headline_change, 3),
        category_breakdown=category_results,
        shap_explanation=shap_summary[:5],
        income_context={
            "income_median":     income.get("income_median"),
            "income_band":       income.get("income_band"),
            "expenditure_mean":  income.get("expenditure_mean"),
            "gini":              income.get("gini"),
            "poverty_rate":      income.get("poverty_rate"),
        },
        latest_date=latest_date,
    )
     
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


@app.post("/predict", response_model=PredictionResult)
async def predict(profile: SpendingProfile):
    """
    Calculate personal inflation rate for a given spending profile.

    Takes a state and spending weights across CPI categories,
    returns personal inflation rate, category breakdown, SHAP
    explanation, and income context.

    Example request:
    {
        "state": "Kedah",
        "weights": {
            "01": 35,
            "04": 20,
            "07": 15,
            "11": 10,
            "10": 5,
            "06": 5,
            "09": 5,
            "03": 5
        }
    }
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Try again in a moment."
        )

    profile.validate_weights()

    logger.info(
        f"Prediction request: state={profile.state}, "
        f"categories={list(profile.weights.keys())}"
    )

    return calculate_personal_inflation(profile.state, profile.weights)


@app.post("/compare", response_model=ComparisonResult)
async def compare(request: ComparisonRequest):
    """
    Relocation planner — compare personal inflation between
    two Malaysian states with potentially different spending patterns.

    Origin = current state and spending habits.
    Destination = target state and expected new spending pattern.

    Returns side-by-side inflation rates, cost difference percentage,
    and estimated salary needed to maintain current standard of living.

    Example request:
    {
        "origin_state": "Kedah",
        "destination_state": "W.P. Kuala Lumpur",
        "origin_weights": {"01": 35, "04": 20, "07": 15, "11": 10,
                           "10": 5, "06": 5, "09": 5, "03": 5},
        "destination_weights": {"01": 25, "04": 35, "07": 10, "11": 15,
                                "10": 5, "06": 5, "09": 3, "03": 2}
    }
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Try again in a moment."
        )

    logger.info(
        f"Comparison request: {request.origin_state} → "
        f"{request.destination_state}"
    )

    # Calculate both profiles
    origin_result = calculate_personal_inflation(
        request.origin_state,
        request.origin_weights,
    )
    destination_result = calculate_personal_inflation(
        request.destination_state,
        request.destination_weights,
    )

    # Cost difference between the two states
    origin_inf  = origin_result.personal_inflation_rate
    dest_inf    = destination_result.personal_inflation_rate
    cost_diff   = round(dest_inf - origin_inf, 3)

    # Estimate salary needed in destination to maintain origin
    # standard of living — based on expenditure_mean ratio
    origin_expenditure = origin_result.income_context.get(
        "expenditure_mean", 3000
    ) or 3000
    dest_expenditure = destination_result.income_context.get(
        "expenditure_mean", 3000
    ) or 3000

    expenditure_ratio = (
        dest_expenditure / origin_expenditure
        if origin_expenditure > 0 else 1.0
    )

    # Salary needed = current origin expenditure scaled to destination
    # costs plus 30% buffer for savings and emergencies
    salary_needed = round(dest_expenditure * 1.30, 0)

    # Generate human-readable summary
    direction = "higher" if cost_diff > 0 else "lower"
    summary = (
        f"Moving from {request.origin_state} to "
        f"{request.destination_state}, your personal inflation rate "
        f"is expected to be {abs(cost_diff):.1f}% {direction}. "
        f"Based on median expenditure data, you would need approximately "
        f"RM{salary_needed:,.0f}/month salary to maintain your current "
        f"standard of living, accounting for higher costs and a 30% "
        f"savings buffer."
    )

    return ComparisonResult(
        origin=origin_result,
        destination=destination_result,
        cost_difference_pct=cost_diff,
        salary_needed=salary_needed,
        summary=summary,
    )

@app.get("/forecast")
async def forecast(
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

    if horizon_months not in (3, 6, 12):
        raise HTTPException(
            status_code=400,
            detail="horizon_months must be 3, 6, or 12"
        )

    results = recursive_forecast(state, division, horizon_months)

    return {
        "state": state,
        "division": division,
        "category_name": CATEGORIES.get(division, division),
        "horizon_months": horizon_months,
        "forecast": results,
    }

@app.get("/price-ratio")
async def price_ratio(
    origin_state: str,
    destination_state: str,
    division: str,
):
    """
    Returns the current price ratio between two states for one category.
    Used to auto-scale a user's spending when comparing states.
    """
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

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)