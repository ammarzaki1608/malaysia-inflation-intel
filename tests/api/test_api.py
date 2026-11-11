"""Tests for ml/serving/api.py. No network: GCS, BigQuery and Gemini are never called."""

import json
from types import SimpleNamespace

import api
import pytest


# ── reference endpoints ──────────────────────────────────────────────────────

def test_health_reports_forecast_table(client, forecast_table):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["forecast_table_loaded"] is True
    assert body["forecast_version"] == "test_v1"


def test_states_and_categories(client):
    assert len(client.get("/states").json()["states"]) == 16
    cats = client.get("/categories").json()["categories"]
    assert [c["code"] for c in cats] == [f"{i:02d}" for i in range(1, 14)]


# ── /forecast ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("horizon", [3, 6, 12])
def test_forecast_returns_one_step_per_month(client, forecast_table, horizon):
    r = client.get("/forecast", params={"state": "Selangor", "division": "01", "horizon_months": horizon})
    assert r.status_code == 200
    body = r.json()
    assert [s["step"] for s in body["forecast"]] == list(range(1, horizon + 1))
    assert body["source"] == "precomputed"
    assert body["model_version"] == "test_v1"
    assert body["latest_real_index"] == 130.0
    assert body["category_name"] == "Food & Non-Alcoholic Beverages"
    for s in body["forecast"]:
        assert s["lower_bound"] <= s["predicted_index"] <= s["upper_bound"]


@pytest.mark.parametrize("params, fragment", [
    ({"state": "Atlantis", "division": "01"}, "Unknown state"),
    ({"state": "Selangor", "division": "99"}, "Unknown division"),
    ({"state": "Selangor", "division": "01", "horizon_months": 5}, "horizon_months"),
])
def test_forecast_rejects_bad_input(client, forecast_table, params, fragment):
    r = client.get("/forecast", params=params)
    assert r.status_code == 400
    assert fragment in r.json()["detail"]


def test_forecast_503_when_no_table_and_no_legacy_model(client, monkeypatch):
    def boom():
        raise RuntimeError("no GCS in tests")
    monkeypatch.setattr(api, "load_model_from_gcs", boom)
    r = client.get("/forecast", params={"state": "Selangor", "division": "01"})
    assert r.status_code == 503


# ── /forecasts (bulk) ────────────────────────────────────────────────────────

def test_bulk_forecasts_deduplicates_and_keeps_order(client, forecast_table):
    r = client.get("/forecasts", params={"state": "Selangor", "divisions": "07, 01,07", "horizon_months": 3})
    assert r.status_code == 200
    body = r.json()
    assert list(body["forecasts"]) == ["07", "01"]
    assert body["errors"] == {}
    assert len(body["forecasts"]["01"]["forecast"]) == 3


@pytest.mark.parametrize("divisions", ["", "01,99", ",,"])
def test_bulk_forecasts_rejects_bad_division_lists(client, forecast_table, divisions):
    r = client.get("/forecasts", params={"state": "Selangor", "divisions": divisions})
    assert r.status_code == 400


# ── validation on the BigQuery-backed endpoints (must fail before any query) ──

def test_trends_rejects_unknown_division_with_400(client):
    """Regression: a `detatil=` typo made this a 500 in production."""
    r = client.get("/trends", params={"state": "Selangor", "division": "99"})
    assert r.status_code == 400
    assert "Unknown division" in r.json()["detail"]


def test_trends_rejects_unknown_state_and_window(client):
    assert client.get("/trends", params={"state": "Atlantis", "division": "01"}).status_code == 400
    r = client.get("/trends", params={"state": "Selangor", "division": "01", "months_back": 7})
    assert r.status_code == 400


def test_income_snapshot_rejects_unknown_state(client):
    r = client.get("/income-snapshot", params={"state": "Atlantis"})
    assert r.status_code == 400
    assert "Unknown state" in r.json()["detail"]


def test_price_ratio_rejects_bad_input(client):
    ok = {"origin_state": "Selangor", "destination_state": "Kedah", "division": "01"}
    for key, bad in [("origin_state", "X"), ("destination_state", "X"), ("division", "99")]:
        assert client.get("/price-ratio", params={**ok, key: bad}).status_code == 400


# ── /state-leaderboard ───────────────────────────────────────────────────────

def test_state_leaderboard_ranks_fastest_rising_first(client, forecast_table):
    body = client.get("/state-leaderboard", params={"division": "01"}).json()
    assert [s["state"] for s in body["states"]] == ["Kedah", "Selangor"]      # 3.1% then 2.5%
    assert body["category_name"] == "Food & Non-Alcoholic Beverages"


def test_state_leaderboard_rejects_unknown_division(client, forecast_table):
    assert client.get("/state-leaderboard", params={"division": "99"}).status_code == 400


# ── /forecast-accuracy ───────────────────────────────────────────────────────

def test_forecast_accuracy_serves_the_published_scorecard(client, monkeypatch, tmp_path):
    card = {"scored_forecasts": 208, "by_horizon": {"1": {"rel_mae": 1.004, "coverage_80": 0.793}}}
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(card), encoding="utf-8")
    monkeypatch.setattr(api, "SCORECARD_LOCAL_PATH", str(path))
    monkeypatch.setattr(api, "_scorecard", {"data": None, "checked_at": 0.0})
    r = client.get("/forecast-accuracy")
    assert r.status_code == 200
    assert r.json() == card


def test_forecast_accuracy_404_before_first_scorecard(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "SCORECARD_LOCAL_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(api, "_scorecard", {"data": None, "checked_at": 0.0})
    assert client.get("/forecast-accuracy").status_code == 404


# ── /ask (RAG) ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("question, expected", [
    ("Why is my forecast so high?", True),
    ("Kenapa perbelanjaan saya naik?", True),
    ("WHY DOES MY transport cost rise?", True),
    ("What does B40 mean?", False),
    ("How often is the data refreshed?", False),
])
def test_needs_live_forecast_heuristic(question, expected):
    assert api._needs_live_forecast(question) is expected


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace retrieval and Gemini; record the prompt the model would have been sent."""
    calls = []

    def fake_retrieve(query, top_k=3):
        return [{"id": "kb_one", "text": "FACT ONE", "distance": 0.1},
                {"id": "kb_two", "text": "FACT TWO", "distance": 0.2}]

    def generate_content(model, contents, config):
        calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text="A grounded answer.")

    monkeypatch.setattr(api, "retrieve", fake_retrieve)
    monkeypatch.setattr(api, "_ask_genai_client",
                        SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    return calls


def test_ask_general_question_uses_knowledge_only(client, forecast_table, fake_llm):
    r = client.post("/ask", json={"question": "What does B40 mean?", "state": "Selangor", "division": "01"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "A grounded answer."
    assert body["retrieved_context"] == ["kb_one", "kb_two"]
    assert body["used_live_forecast_data"] is False

    prompt = fake_llm[0]["contents"]
    assert "FACT ONE" in prompt and "FACT TWO" in prompt
    assert "Not applicable" in prompt                      # no forecast numbers were injected
    assert "never invent" in fake_llm[0]["config"]["system_instruction"]


def test_ask_personal_question_includes_live_forecast(client, forecast_table, fake_llm):
    r = client.post("/ask", json={"question": "Why is my forecast rising?", "state": "Selangor",
                                  "division": "07", "horizon_months": 3})
    assert r.status_code == 200
    assert r.json()["used_live_forecast_data"] is True
    prompt = fake_llm[0]["contents"]
    assert "Transport in Selangor" in prompt
    assert "2026-09-01" in prompt                          # latest real data date from the table


def test_ask_personal_question_without_state_skips_live_data(client, forecast_table, fake_llm):
    r = client.post("/ask", json={"question": "Why is my forecast rising?"})
    assert r.status_code == 200
    assert r.json()["used_live_forecast_data"] is False


def test_ask_rejects_unknown_state_for_personal_question(client, forecast_table, fake_llm):
    r = client.post("/ask", json={"question": "Why is my forecast rising?", "state": "Atlantis", "division": "01"})
    assert r.status_code == 400
    assert fake_llm == []                                  # the model was never called


# ── /prescriptive-insight ────────────────────────────────────────────────────

def test_prescriptive_insight_grounds_prompt_in_report_numbers(client, fake_llm):
    r = client.post("/prescriptive-insight", json={
        "context_type": "personal_calculator",
        "summary_data": {"biggest_leak_name": "Transport", "biggest_leak_pct": 4.2,
                         "current_surplus": 800, "projected_surplus": 650, "horizon_months": 6},
    })
    assert r.status_code == 200
    assert r.json() == {"insight": "A grounded answer."}
    prompt = fake_llm[0]["contents"]
    assert "Transport" in prompt and "RM800" in prompt and "RM650" in prompt
    assert "Never recommend specific financial products" in fake_llm[0]["config"]["system_instruction"]
