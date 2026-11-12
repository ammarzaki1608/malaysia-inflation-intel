"""
The precomputed forecast table and the published scorecard.

Both are JSON files written to GCS by ml/training/forecast_batch.py. They are
held in memory and re-checked at most once every FORECAST_REFRESH_S seconds.
"""

import json
import os
import time

from fastapi import HTTPException
from google.cloud import storage

from service.config import CATEGORIES, GCS_BUCKET, PROJECT_ID, logger

FORECAST_GCS_PATH = os.getenv("FORECAST_GCS_PATH", "forecasts/latest.json")
FORECAST_LOCAL_PATH = os.getenv("FORECAST_LOCAL_PATH")      # for local testing only
FORECAST_REFRESH_S = int(os.getenv("FORECAST_REFRESH_SECONDS", "3600"))
FORECAST_RETRY_S = 60       # how soon to try again while no table is loaded at all
SCORECARD_GCS_PATH = os.getenv("SCORECARD_GCS_PATH", "forecasts/scorecard.json")
SCORECARD_LOCAL_PATH = os.getenv("SCORECARD_LOCAL_PATH")    # for local testing only

# {(state, division): [row for h=1..12]}
forecast_rows: dict = {}
forecast_meta: dict = {}
_forecast_generation = None
_forecast_checked_at = 0.0

_scorecard = {"data": None, "checked_at": 0.0}


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
    """
    Re-check for a newer table at most every FORECAST_REFRESH_S seconds; never raises.
    While no table is loaded (the load at startup failed), try again every minute.
    """
    wait = FORECAST_REFRESH_S if forecast_rows else min(FORECAST_RETRY_S, FORECAST_REFRESH_S)
    if time.time() - _forecast_checked_at < wait:
        return
    try:
        load_forecast_table()
    except Exception as e:
        logger.warning(f"Forecast table refresh failed, keeping current: {e}")


def get_forecast(state: str, division: str, horizon_months: int) -> dict:
    """
    Forecast for one series, from the precomputed table. No model runs here.
    503 while no table is loaded; 404 if the table has no entry for this series.
    """
    maybe_refresh_forecast_table()
    if not forecast_rows:
        raise HTTPException(status_code=503, detail="Forecasts are not available right now.")
    rows = forecast_rows.get((state, division))
    if not rows:
        raise HTTPException(status_code=404, detail=f"No forecast is available for {state}/{division}.")

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


def forecast_payload(state: str, division: str, horizon_months: int) -> dict:
    """The body /forecast returns for one series."""
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


def load_scorecard():
    """The published scorecard, re-read at most hourly. None if there is none yet."""
    if time.time() - _scorecard["checked_at"] > FORECAST_REFRESH_S or _scorecard["data"] is None:
        _scorecard["checked_at"] = time.time()
        try:
            if SCORECARD_LOCAL_PATH:
                with open(SCORECARD_LOCAL_PATH, encoding="utf-8") as f:
                    _scorecard["data"] = json.load(f)
            else:
                blob = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET).blob(SCORECARD_GCS_PATH)
                _scorecard["data"] = json.loads(blob.download_as_text())
        except Exception as e:
            logger.warning(f"Scorecard not available: {e}")
    return _scorecard["data"]
