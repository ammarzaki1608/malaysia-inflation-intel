"""
Fixtures for the API tests.

The tests never touch GCP or Gemini. The API normally loads its forecast table
and RAG index at startup; here the startup hook is not run (TestClient is used
without a `with` block) and the state held by the `service` modules is filled in directly.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ml", "serving"))

import api  # noqa: E402
from service import forecasts  # noqa: E402
from service import llm_log as llm_logging  # noqa: E402
from service.config import limiter  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def make_rows(state: str, division: str, base_index: float, past_12m: float) -> list[dict]:
    """Twelve forecast rows for one series, shaped like forecast_batch.to_payload writes them."""
    rows = []
    for h in range(1, 13):
        predicted = round(base_index * (1 + 0.002 * h), 3)
        rows.append({
            "state": state, "division": division, "h": h,
            "base_date": "2026-09-01", "base_index": base_index,
            "past_12m_change_pct": past_12m,
            "target_date": f"{2026 + (9 + h - 1) // 12}-{(9 + h - 1) % 12 + 1:02d}-01",
            "predicted_index": predicted,
            "lower_bound": round(predicted * 0.99, 3),
            "upper_bound": round(predicted * 1.01, 3),
            "confidence_band_pct": 1.0,
        })
    return rows


@pytest.fixture
def forecast_table(monkeypatch):
    """A small precomputed forecast table: two states for Food, one for Transport."""
    table = {
        ("Selangor", "01"): make_rows("Selangor", "01", 130.0, 2.5),
        ("Kedah", "01"): make_rows("Kedah", "01", 128.0, 3.1),
        ("Selangor", "07"): make_rows("Selangor", "07", 120.0, -0.4),
    }
    monkeypatch.setattr(forecasts, "forecast_rows", table)
    monkeypatch.setattr(forecasts, "forecast_meta", {"version": "test_v1", "interval_level": 0.8,
                                               "base_date": "2026-09-01"})
    return table


@pytest.fixture
def client(monkeypatch):
    # "just checked": stops maybe_refresh_forecast_table() from calling GCS
    monkeypatch.setattr(forecasts, "_forecast_checked_at", time.time())
    monkeypatch.setattr(forecasts, "forecast_rows", {})
    monkeypatch.setattr(forecasts, "forecast_meta", {})
    monkeypatch.setattr(limiter, "enabled", False)
    monkeypatch.setattr(llm_logging, "LLM_LOG_ENABLED", False)      # never write to BigQuery from tests
    monkeypatch.setattr(llm_logging, "EVAL_LOG_TOKEN", "test-eval-token")
    return TestClient(api.app)


@pytest.fixture
def llm_log(monkeypatch):
    """Capture the rows log_llm_call would write, instead of writing them."""
    rows = []
    monkeypatch.setattr(llm_logging, "log_llm_call", rows.append)
    return rows
