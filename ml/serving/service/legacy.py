"""
The legacy recursive XGBoost model. Fallback only.

Forecasts are normally served from the table that ml/training/forecast_batch.py
precomputes every month (see forecasts.py). This module is used only when that
table has no entry for a series, which in practice means the table could not
be loaded at all. Nothing here runs at startup: the model is downloaded the
first time a fallback forecast is actually needed.

Its answers are weaker than the table's: the range is a fixed 0.5% per month
ahead, not calibrated on backtest errors, and the steps carry no target date.
Responses built from it are marked `"source": "legacy_recursive"`.
"""

import os
import tempfile

import numpy as np
from fastapi import HTTPException
from google.cloud import storage

from service import data
from service.config import GCS_BUCKET, PROJECT_ID, logger

MODEL_PATH = os.getenv("MODEL_GCS_PATH", "models/cpi_forecast_20260619_194024.joblib")

model = None


def load_model_from_gcs() -> None:
    """Download the trained model from GCS into `model`."""
    global model
    import joblib          # imported here so a normal start does not pay for it

    logger.info(f"Loading legacy model from gs://{GCS_BUCKET}/{MODEL_PATH}")
    blob = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET).blob(MODEL_PATH)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        blob.download_to_filename(f.name)
        model = joblib.load(f.name)
    logger.info("Legacy model loaded")


def build_features_from_history(working_history: list[dict]) -> np.ndarray:
    """
    Build one model input row by reading backwards from the end of
    working_history. working_history[-1] is the most recent value (real or
    predicted), working_history[-2] the one before, and so on.
    """
    n = len(working_history)

    def index_at(steps_back: int) -> float:
        # not enough history: fall back to the oldest value we have
        return working_history[max(0, n - steps_back)]["cpi_index"]

    last_3 = [working_history[i]["cpi_index"] for i in range(max(0, n - 3), n)]
    last_12 = [working_history[i]["cpi_index"] for i in range(max(0, n - 12), n)]

    # PPI and OPR are not part of the recursive loop: they are held at their
    # last known real value, since nothing here forecasts them.
    latest = working_history[-1]
    row = [
        index_at(1), index_at(3), index_at(6), index_at(12),
        sum(last_3) / len(last_3), sum(last_12) / len(last_12),
        latest.get("ppi_index", 0.0) or 0.0,
        latest.get("ppi_lag_1", 0.0) or 0.0,
        latest.get("ppi_lag_2", 0.0) or 0.0,
        latest.get("ppi_lag_3", 0.0) or 0.0,
        latest.get("opr_rate", 0.0) or 0.0,
    ]
    return np.array([row])


def append_prediction(working_history: list[dict], predicted_index: float) -> list[dict]:
    """Add a prediction to the end of the history, carrying PPI and OPR forward unchanged."""
    previous = working_history[-1]
    return working_history + [{
        "cpi_index": predicted_index,
        "ppi_index": previous.get("ppi_index", 0.0),
        "ppi_lag_1": previous.get("ppi_lag_1", 0.0),
        "ppi_lag_2": previous.get("ppi_lag_2", 0.0),
        "ppi_lag_3": previous.get("ppi_lag_3", 0.0),
        "opr_rate": previous.get("opr_rate", 0.0),
    }]


def recursive_forecast(state: str, division: str, horizon_months: int) -> dict:
    """
    Forecast N months ahead by calling the model repeatedly, rebuilding the lag
    features from its own previous predictions at each step.
    """
    history = data.feature_history(state, division, months_back=15)
    if len(history) < 13:
        raise HTTPException(status_code=404, detail=f"Not enough history for {state}/{division} to forecast")

    working_history = history.copy()
    forecasts = []
    base_uncertainty_pct = 0.5          # widens linearly with each step; not calibrated

    for step in range(1, horizon_months + 1):
        predicted_index = float(model.predict(build_features_from_history(working_history))[0])
        uncertainty_pct = base_uncertainty_pct * step
        forecasts.append({
            "step": step,
            "predicted_index": round(predicted_index, 3),
            "lower_bound": round(predicted_index * (1 - uncertainty_pct / 100), 3),
            "upper_bound": round(predicted_index * (1 + uncertainty_pct / 100), 3),
            "confidence_band_pct": round(uncertainty_pct, 2),
        })
        working_history = append_prediction(working_history, predicted_index)

    return {"forecasts": forecasts, "latest_real_date": str(history[-1]["date"])}


def forecast(state: str, division: str, horizon_months: int) -> dict:
    """A fallback forecast, loading the model first if needed. 503 if it cannot be loaded."""
    if model is None:
        try:
            load_model_from_gcs()
        except Exception as e:
            logger.error(f"Legacy model unavailable: {e}")
            raise HTTPException(status_code=503, detail="Forecasts are not available yet.")
    result = recursive_forecast(state, division, horizon_months)
    result["source"] = "legacy_recursive"
    return result
