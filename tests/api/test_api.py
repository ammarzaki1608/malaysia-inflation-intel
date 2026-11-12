"""Tests for the API (ml/serving). No network: GCS, BigQuery and Gemini are never called."""

import json
from types import SimpleNamespace

import api
import pytest
from fastapi.testclient import TestClient
from service import chat, chat_tools, data, forecasts, legacy
from service import llm_log as llm_logging


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
    monkeypatch.setattr(legacy, "load_model_from_gcs", boom)
    r = client.get("/forecast", params={"state": "Selangor", "division": "01"})
    assert r.status_code == 503


def test_forecast_falls_back_to_the_legacy_model_when_the_table_has_no_entry(client, monkeypatch):
    history = [{"date": f"2025-{m:02d}-01", "cpi_index": 100.0 + m, "ppi_index": 110.0, "opr_rate": 2.75}
               for m in range(1, 13)] + [{"date": "2026-01-01", "cpi_index": 113.0, "ppi_index": 110.0,
                                          "opr_rate": 2.75}]
    monkeypatch.setattr(data, "feature_history", lambda state, division, months_back=15: history)
    monkeypatch.setattr(legacy, "model", SimpleNamespace(predict=lambda X: [float(X[0][0]) + 1.0]))
    body = client.get("/forecast", params={"state": "Selangor", "division": "01", "horizon_months": 3}).json()
    assert body["source"] == "legacy_recursive"
    # each step feeds on the one before it: 113 -> 114 -> 115 -> 116
    assert [s["predicted_index"] for s in body["forecast"]] == [114.0, 115.0, 116.0]
    assert body["latest_real_date"] == "2026-01-01"


def test_startup_survives_a_missing_forecast_table_and_does_not_load_the_legacy_model(monkeypatch):
    """The legacy model is only downloaded when a fallback forecast is actually asked for."""
    def no_table(force=False):
        raise RuntimeError("forecasts/latest.json not found")

    def must_not_run():
        raise AssertionError("the legacy model was loaded at startup")
    monkeypatch.setattr(forecasts, "load_forecast_table", no_table)
    monkeypatch.setattr(forecasts, "forecast_rows", {})
    monkeypatch.setattr(legacy, "load_model_from_gcs", must_not_run)
    monkeypatch.setattr(chat, "load_index_from_gcs", lambda: None)
    with TestClient(api.app) as started:
        body = started.get("/health").json()
    assert body["status"] == "ok" and body["forecast_table_loaded"] is False


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


def test_the_deprecated_price_ratio_endpoint_is_gone(client):
    assert client.get("/price-ratio", params={"origin_state": "Selangor", "destination_state": "Kedah",
                                              "division": "01"}).status_code == 404


# ── BigQuery-backed endpoints, with the queries replaced ─────────────────────

def test_headline_inflation_compares_the_latest_month_with_a_year_earlier(client, monkeypatch):
    rows = [{"date": f"month-{i}", "index": 102.0 if i == 0 else 100.0} for i in range(13)]
    monkeypatch.setattr(data, "national_headline_cpi", lambda: rows)
    body = client.get("/headline-inflation").json()
    assert body == {"headline_inflation_pct": 2.0, "latest_date": "month-0", "compared_to_date": "month-12"}

    monkeypatch.setattr(data, "national_headline_cpi", lambda: rows[:5])
    assert client.get("/headline-inflation").status_code == 404


def test_category_leaderboard_ranks_fastest_rising_first_and_skips_short_histories(client, monkeypatch):
    def series(latest):
        return [{"index": latest}] + [{"index": 100.0}] * 12
    monkeypatch.setattr(data, "national_cpi_by_division",
                        lambda: {"01": series(103.0), "07": series(105.5), "06": series(101.0)[:6]})
    cats = client.get("/category-leaderboard").json()["categories"]
    assert [(c["division"], c["yoy_change_pct"]) for c in cats] == [("07", 5.5), ("01", 3.0)]
    assert cats[0]["category_name"] == "Transport"


def test_state_leaderboard_uses_the_mart_when_there_is_no_forecast_table(client, monkeypatch):
    monkeypatch.setattr(data, "latest_cpi_by_state", lambda division: [
        {"state": "Perak", "cpi_index": 102.0, "cpi_lag_12": 100.0, "date": "2026-09-01"},
        {"state": "Johor", "cpi_index": 104.0, "cpi_lag_12": 100.0, "date": "2026-09-01"},
        {"state": "Sabah", "cpi_index": 104.0, "cpi_lag_12": None, "date": "2026-09-01"},
    ])
    states = client.get("/state-leaderboard", params={"division": "01"}).json()["states"]
    assert [(s["state"], s["yoy_change_pct"]) for s in states] == [("Johor", 4.0), ("Perak", 2.0)]


def test_trends_and_income_snapshot_return_404_when_there_is_no_data(client, monkeypatch):
    monkeypatch.setattr(data, "feature_history", lambda state, division, months_back=15: [])
    monkeypatch.setattr(data, "income_snapshot", lambda state: None)
    assert client.get("/trends", params={"state": "Selangor", "division": "01"}).status_code == 404
    assert client.get("/income-snapshot", params={"state": "Selangor"}).status_code == 404


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
    monkeypatch.setattr(forecasts, "SCORECARD_LOCAL_PATH", str(path))
    monkeypatch.setattr(forecasts, "_scorecard", {"data": None, "checked_at": 0.0})
    r = client.get("/forecast-accuracy")
    assert r.status_code == 200
    assert r.json() == card


def test_forecast_accuracy_404_before_first_scorecard(client, monkeypatch, tmp_path):
    monkeypatch.setattr(forecasts, "SCORECARD_LOCAL_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(forecasts, "_scorecard", {"data": None, "checked_at": 0.0})
    assert client.get("/forecast-accuracy").status_code == 404


# ── /ask (RAG + tool calling) ────────────────────────────────────────────────

def prompt_of(call) -> str:
    """The text of the first message sent to the model in one recorded call."""
    return call["contents"][0]["parts"][0]["text"]


def tool_call(name, **args):
    return SimpleNamespace(name=name, args=args, id=f"call-{name}")


def model_turn(text=None, calls=()):
    """What Gemini returns: either an answer, or a request to run tools."""
    return SimpleNamespace(text=text, function_calls=list(calls),
                           candidates=[SimpleNamespace(content={"role": "model", "parts": ["<model turn>"]},
                                                       finish_reason=SimpleNamespace(name="STOP"))])


def install_llm(monkeypatch, script):
    """Fake retrieval and a fake Gemini that replies with `script` (the last entry repeats)."""
    calls = []

    def fake_retrieve(query, top_k=3):
        return [{"id": "kb_one", "text": "FACT ONE", "distance": 0.1},
                {"id": "kb_two", "text": "FACT TWO", "distance": 0.2}]

    def generate_content(model, contents, config):
        calls.append({"model": model, "contents": list(contents) if isinstance(contents, list) else contents,
                      "config": config})
        return script[min(len(calls), len(script)) - 1]

    monkeypatch.setattr(chat, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat, "_genai_client",
                        SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    return calls


@pytest.fixture
def fake_llm(monkeypatch):
    """A model that answers straight away, without asking for any tool."""
    return install_llm(monkeypatch, [SimpleNamespace(text="A grounded answer.")])


def tool_results_sent(call) -> list:
    """The tool results the API sent back to the model in one recorded call."""
    return [p.function_response for message in call["contents"][2:] if hasattr(message, "parts")
            for p in message.parts]


def test_ask_general_question_needs_one_model_call_and_no_tool(client, forecast_table, fake_llm):
    r = client.post("/ask", json={"question": "What does B40 mean?", "state": "Selangor", "division": "01"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "A grounded answer."
    assert body["retrieved_context"] == ["kb_one", "kb_two"]
    assert body["used_live_forecast_data"] is False
    assert body["tools_called"] == [] and body["tool_results"] == []

    assert len(fake_llm) == 1
    prompt = prompt_of(fake_llm[0])
    assert "FACT ONE" in prompt and "FACT TWO" in prompt
    # what the user has open is a hint; no forecast numbers are pasted in up front
    assert "State: Selangor. Category: Food & Non-Alcoholic Beverages." in prompt
    assert "130" not in prompt
    assert "never invent" in fake_llm[0]["config"]["system_instruction"]


def test_ask_offers_the_tools_and_runs_them_itself(client, fake_llm):
    client.post("/ask", json={"question": "What does B40 mean?"})
    config = fake_llm[0]["config"]
    names = [d["name"] for d in config["tools"][0]["function_declarations"]]
    assert sorted(names) == sorted(chat_tools.ASK_TOOL_FUNCTIONS)
    assert config["automatic_function_calling"] == {"disable": True}
    assert config["tool_config"]["function_calling_config"]["mode"] == "AUTO"
    assert "Nothing selected." in prompt_of(fake_llm[0])


def test_the_request_is_valid_for_the_gemini_sdk(client, forecast_table, monkeypatch):
    """The fake model accepts anything; the real SDK must accept the same config and messages."""
    from google.genai import types
    calls = install_llm(monkeypatch, [
        model_turn(calls=[tool_call("get_forecast", state="Selangor", category="Transport")]),
        model_turn(text="Done."),
    ])
    client.post("/ask", json={"question": "Will transport rise in Selangor?"})
    for call in calls:
        types.GenerateContentConfig.model_validate(call["config"])
    for message in calls[1]["contents"]:
        if isinstance(message, dict) and message.get("role") == "user":
            types.Content.model_validate(message)
    sent = calls[1]["contents"][2]
    assert isinstance(sent, types.Content) and sent.role == "user"


def test_ask_runs_the_tool_the_model_asks_for_and_returns_its_result(client, forecast_table, monkeypatch):
    calls = install_llm(monkeypatch, [
        model_turn(calls=[tool_call("get_forecast", state="Selangor", category="Transport", horizon_months=3)]),
        model_turn(text="Transport in Selangor is forecast to rise 0.6%."),
    ])
    r = client.post("/ask", json={"question": "Will transport get more expensive in Selangor?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Transport in Selangor is forecast to rise 0.6%."
    assert body["tools_called"] == ["get_forecast"]
    assert body["used_live_forecast_data"] is True
    result = body["tool_results"][0]["result"]
    assert result["category"] == "Transport" and len(result["forecast"]) == 3

    assert len(calls) == 2
    # the second call carries: the question, the model's own turn unchanged, then the tool result
    assert calls[1]["contents"][1] == {"role": "model", "parts": ["<model turn>"]}
    (sent,) = tool_results_sent(calls[1])
    assert sent.name == "get_forecast" and sent.id == "call-get_forecast"
    assert sent.response["forecast"][2]["expected_change_pct"] == 0.6


def test_ask_can_compare_two_states_in_one_round(client, forecast_table, monkeypatch):
    """The old keyword version could only ever see the one forecast open in the app."""
    calls = install_llm(monkeypatch, [
        model_turn(calls=[tool_call("get_forecast", state="Selangor", category="01"),
                          tool_call("get_forecast", state="Kedah", category="01")]),
        model_turn(text="Both rise about the same."),
    ])
    r = client.post("/ask", json={"question": "Is food rising faster in Selangor or Kedah?"})
    assert r.json()["tools_called"] == ["get_forecast", "get_forecast"]
    assert [s.response["state"] for s in tool_results_sent(calls[1])] == ["Selangor", "Kedah"]


def test_ask_forces_an_answer_after_the_round_limit(client, forecast_table, monkeypatch):
    again = model_turn(calls=[tool_call("rank_states", category="Food & Non-Alcoholic Beverages")])
    calls = install_llm(monkeypatch, [again, again, model_turn(text="Final answer.")])
    r = client.post("/ask", json={"question": "Which state?"})
    assert r.status_code == 200 and r.json()["answer"] == "Final answer."
    assert len(calls) == chat.ASK_MAX_TOOL_ROUNDS + 1
    modes = [c["config"]["tool_config"]["function_calling_config"]["mode"] for c in calls]
    assert modes == ["AUTO", "AUTO", "NONE"]                # the last call may not ask for tools


def test_ask_returns_503_if_the_model_never_answers(client, forecast_table, monkeypatch):
    forever = model_turn(calls=[tool_call("get_forecast_accuracy")])
    calls = install_llm(monkeypatch, [forever])
    r = client.post("/ask", json={"question": "How accurate?"})
    assert r.status_code == 503
    assert len(calls) == chat.ASK_MAX_TOOL_ROUNDS + 1         # it stopped; it did not loop forever


def test_ask_caps_the_number_of_tool_calls(client, forecast_table, monkeypatch):
    greedy = model_turn(calls=[tool_call("get_forecast", state="Selangor", category="01")] * 10)
    calls = install_llm(monkeypatch, [greedy, model_turn(text="Enough.")])
    r = client.post("/ask", json={"question": "Everything please"})
    assert len(r.json()["tools_called"]) == chat.ASK_MAX_TOOL_CALLS
    assert calls[1]["config"]["tool_config"]["function_calling_config"]["mode"] == "NONE"


def test_a_bad_tool_argument_goes_back_to_the_model_as_an_error(client, forecast_table, monkeypatch):
    calls = install_llm(monkeypatch, [
        model_turn(calls=[tool_call("get_forecast", state="Atlantis", category="Transport")]),
        model_turn(text="I don't have data for that state."),
    ])
    r = client.post("/ask", json={"question": "Transport in Atlantis?"})
    assert r.status_code == 200
    assert r.json()["used_live_forecast_data"] is False
    (sent,) = tool_results_sent(calls[1])
    assert "Unknown state 'Atlantis'" in sent.response["error"] and "Selangor" in sent.response["error"]


@pytest.mark.parametrize("name, args, expected", [
    ("delete_everything", {}, "Unknown tool"),
    ("get_forecast", {"planet": "Mars"}, "Wrong arguments"),
    ("get_forecast", {"state": "Selangor", "category": "Gold bars"}, "Unknown category"),
    ("get_forecast", {"state": "Perlis", "category": "Health"}, "No forecast is available"),
    ("rank_categories", {"state": "Atlantis"}, "Unknown state"),
    ("rank_states", {"category": "Gold bars"}, "Unknown category"),
])
def test_tools_report_problems_instead_of_raising(client, forecast_table, name, args, expected):
    assert expected in chat_tools.run_tool(name, args)["error"]


def test_a_crashing_tool_does_not_take_the_answer_down(client, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("storage is down")
    monkeypatch.setitem(chat_tools.ASK_TOOL_FUNCTIONS, "get_forecast_accuracy", boom)
    assert chat_tools.run_tool("get_forecast_accuracy", {}) == {"error": "get_forecast_accuracy is unavailable right now."}


def test_get_forecast_tool_reports_change_from_the_latest_actual_month(client, forecast_table):
    out = chat_tools.tool_get_forecast("Selangor", "Food & Non-Alcoholic Beverages", 3)
    assert out["latest_actual_month"] == "2026-09" and out["change_over_past_12_months_pct"] == 2.5
    assert [m["month"] for m in out["forecast"]] == ["2026-10", "2026-11", "2026-12"]
    assert [m["expected_change_pct"] for m in out["forecast"]] == [0.2, 0.4, 0.6]
    first = out["forecast"][0]
    assert first["range_low_pct"] < first["expected_change_pct"] < first["range_high_pct"]
    assert out["range_confidence_pct"] == 80
    # a code works as well as a name, and the horizon is kept inside 1..12
    assert chat_tools.tool_get_forecast("Selangor", "01", 99)["horizon_months"] == 12
    assert chat_tools.tool_get_forecast("Selangor", "transport", "soon")["horizon_months"] == 6


def test_rank_tools_put_the_biggest_rise_first(client, forecast_table, monkeypatch):
    rows = forecasts.forecast_rows[("Kedah", "01")]
    monkeypatch.setitem(forecasts.forecast_rows, ("Kedah", "01"),
                        [dict(r, predicted_index=r["base_index"] * 1.05) for r in rows])
    states = chat_tools.tool_rank_states("Food & Non-Alcoholic Beverages", 6)["states_biggest_rise_first"]
    assert [s["name"] for s in states] == ["Kedah", "Selangor"]         # only states with a forecast
    assert states[0]["forecast_change_pct"] == 5.0 and states[1]["forecast_change_pct"] == 1.2

    cats = chat_tools.tool_rank_categories("Selangor", 12)["categories_biggest_rise_first"]
    assert {c["name"] for c in cats} == {"Food & Non-Alcoholic Beverages", "Transport"}
    assert cats[0]["forecast_change_pct"] >= cats[1]["forecast_change_pct"]


def test_forecast_accuracy_tool(client, monkeypatch, tmp_path):
    monkeypatch.setattr(forecasts, "_scorecard", {"checked_at": 1e12, "data": {
        "latest_actual_date": "2026-09-01", "scored_forecasts": 208,
        "by_horizon": {"1": {"n": 208, "mae_pp": 0.268, "rel_mae": 1.004, "coverage_80": 0.793}}}})
    out = chat_tools.tool_get_forecast_accuracy()
    assert out["forecasts_scored"] == 208 and out["by_months_ahead"]["1"]["coverage_80"] == 0.793

    # nothing published yet (a missing local file, so the test never reaches GCS)
    monkeypatch.setattr(forecasts, "SCORECARD_LOCAL_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(forecasts, "_scorecard", {"checked_at": 0.0, "data": None})
    assert "error" in chat_tools.tool_get_forecast_accuracy()


@pytest.mark.parametrize("payload", [
    {"question": "Why is my forecast rising?", "state": "Atlantis", "division": "01"},
    {"question": "What does B40 mean?", "state": "Selangor", "division": "99"},
])
def test_ask_rejects_an_unknown_state_or_category(client, forecast_table, fake_llm, payload):
    r = client.post("/ask", json=payload)
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


# ── LLM observability ────────────────────────────────────────────────────────

# Only calls carrying the evaluation token are logged; conftest sets the token.
EVAL = {"X-Eval-Token": "test-eval-token"}


class _Usage:
    prompt_token_count = 412
    candidates_token_count = 58
    total_token_count = 470


def test_ask_logs_one_row_with_retrieval_latency_and_tokens(client, forecast_table, fake_llm, llm_log, monkeypatch):
    def generate_content(model, contents, config):
        return SimpleNamespace(text="A grounded answer.", usage_metadata=_Usage(),
                               candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))])
    monkeypatch.setattr(chat, "_genai_client",
                        SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))

    r = client.post("/ask", headers=EVAL, json={"question": "What does B40 mean?", "state": "Selangor", "division": "01"})
    assert r.status_code == 200
    assert len(llm_log) == 1
    row = llm_log[0]
    assert row["endpoint"] == "/ask" and row["status"] == "ok"
    assert row["model"] == chat.GENERATION_MODEL
    assert row["question"] == "What does B40 mean?" and row["answer"] == "A grounded answer."
    assert row["retrieved_ids"] == ["kb_one", "kb_two"]
    assert row["retrieved_distances"] == [0.1, 0.2]
    assert row["used_live_forecast_data"] is False
    assert row["forecast_version"] == "test_v1"
    assert (row["prompt_tokens"], row["output_tokens"], row["total_tokens"]) == (412, 58, 470)
    assert row["finish_reason"] == "STOP"
    for timing in ("retrieval_ms", "generation_ms", "total_ms"):
        assert isinstance(row[timing], int) and row[timing] >= 0


def test_ask_returns_503_and_logs_the_error_when_gemini_fails(client, forecast_table, fake_llm, llm_log, monkeypatch):
    """A Gemini outage used to surface as an unexplained 500."""
    def overloaded(model, contents, config):
        raise RuntimeError("503 UNAVAILABLE: the model is overloaded")
    monkeypatch.setattr(chat, "_genai_client",
                        SimpleNamespace(models=SimpleNamespace(generate_content=overloaded)))

    r = client.post("/ask", headers=EVAL, json={"question": "What does B40 mean?"})
    assert r.status_code == 503
    assert r.json()["detail"] == chat.LLM_UNAVAILABLE
    assert len(llm_log) == 1
    row = llm_log[0]
    assert row["status"] == "error" and row["error_type"] == "RuntimeError"
    assert "overloaded" in row["error_message"]
    assert row["retrieved_ids"] == ["kb_one", "kb_two"]       # retrieval had worked; generation failed
    assert "answer" not in row


def test_ask_returns_503_when_retrieval_fails(client, fake_llm, llm_log, monkeypatch):
    def broken(query, top_k=3):
        raise RuntimeError("embedding service unavailable")
    monkeypatch.setattr(chat, "retrieve", broken)
    r = client.post("/ask", headers=EVAL, json={"question": "What does B40 mean?"})
    assert r.status_code == 503
    assert llm_log[0]["status"] == "error" and "retrieved_ids" not in llm_log[0]
    assert fake_llm == []                                     # the model was never called


def test_bad_input_is_rejected_before_any_llm_work_and_not_logged(client, forecast_table, fake_llm, llm_log):
    r = client.post("/ask", headers=EVAL, json={"question": "Why is my forecast rising?", "state": "Atlantis", "division": "01"})
    assert r.status_code == 400
    assert llm_log == [] and fake_llm == []


def test_a_failing_log_write_never_breaks_the_answer(client, fake_llm, monkeypatch):
    class BrokenBigQuery:
        def insert_rows_json(self, *args, **kwargs):
            raise RuntimeError("BigQuery is down")
    monkeypatch.setattr(llm_logging, "LLM_LOG_ENABLED", True)
    monkeypatch.setattr(data, "_client", BrokenBigQuery())
    r = client.post("/ask", headers=EVAL, json={"question": "What does B40 mean?"})
    assert r.status_code == 200 and r.json()["answer"] == "A grounded answer."


def test_log_row_is_written_to_the_configured_table(client, fake_llm, monkeypatch):
    written = []

    class FakeBigQuery:
        def insert_rows_json(self, table, rows, **kwargs):
            written.append((table, rows))
            return []
    monkeypatch.setattr(llm_logging, "LLM_LOG_ENABLED", True)
    monkeypatch.setattr(data, "_client", FakeBigQuery())
    client.post("/ask", headers=EVAL, json={"question": "What does B40 mean?"})

    (table, rows), = written
    assert table == llm_logging.LLM_LOG_TABLE and table.endswith(".observability.llm_calls")
    row = rows[0]
    assert len(row["call_id"]) == 32 and row["ts"].endswith("+00:00")
    assert row["api_revision"] == "local"
    json.dumps(row)                                           # must be JSON-serialisable for the insert


def test_prescriptive_insight_logs_the_call_but_not_the_users_figures(client, fake_llm, llm_log):
    r = client.post("/prescriptive-insight", headers=EVAL, json={
        "context_type": "personal_calculator",
        "summary_data": {"biggest_leak_name": "Transport", "current_surplus": 800, "projected_surplus": 650},
    })
    assert r.status_code == 200
    row = llm_log[0]
    assert row["endpoint"] == "/prescriptive-insight" and row["context_type"] == "personal_calculator"
    assert "question" not in row and "answer" not in row
    assert "800" not in json.dumps(row) and "650" not in json.dumps(row)


def test_prescriptive_insight_returns_503_when_gemini_fails(client, llm_log, monkeypatch):
    def overloaded(model, contents, config):
        raise RuntimeError("503 UNAVAILABLE")
    monkeypatch.setattr(chat, "_genai_client",
                        SimpleNamespace(models=SimpleNamespace(generate_content=overloaded)))
    r = client.post("/prescriptive-insight", headers=EVAL, json={"context_type": "relocation_planner", "summary_data": {}})
    assert r.status_code == 503
    assert llm_log[0]["status"] == "error"


def test_log_row_records_the_tool_loop_and_adds_up_tokens(client, forecast_table, llm_log, monkeypatch):
    first = model_turn(calls=[tool_call("get_forecast", state="Selangor", category="Transport"),
                              tool_call("get_forecast", state="Atlantis", category="Transport")])
    second = model_turn(text="Answer.")
    first.usage_metadata = second.usage_metadata = _Usage()
    install_llm(monkeypatch, [first, second])
    client.post("/ask", headers=EVAL, json={"question": "Transport in Selangor?"})
    row = llm_log[0]
    assert row["llm_calls"] == 2
    assert row["tools_called"] == ["get_forecast", "get_forecast"] and row["tool_errors"] == 1
    assert row["used_live_forecast_data"] is True
    assert (row["prompt_tokens"], row["output_tokens"], row["total_tokens"]) == (824, 116, 940)


def test_a_users_question_is_never_logged(client, forecast_table, fake_llm, llm_log):
    """No token, no row: what users type into the chat is not recorded."""
    r = client.post("/ask", json={"question": "What does B40 mean?", "state": "Selangor", "division": "01"})
    assert r.status_code == 200 and r.json()["answer"] == "A grounded answer."
    assert llm_log == []


def test_a_users_insight_request_is_never_logged(client, fake_llm, llm_log):
    r = client.post("/prescriptive-insight", json={"context_type": "personal_calculator", "summary_data": {}})
    assert r.status_code == 200
    assert llm_log == []


def test_a_users_failed_call_is_not_logged_either(client, llm_log, monkeypatch):
    def overloaded(model, contents, config):
        raise RuntimeError("503 UNAVAILABLE")
    monkeypatch.setattr(chat, "_genai_client",
                        SimpleNamespace(models=SimpleNamespace(generate_content=overloaded)))
    r = client.post("/ask", json={"question": "What does B40 mean?"})
    assert r.status_code == 503
    assert llm_log == []


def test_a_wrong_token_is_not_logged(client, fake_llm, llm_log):
    client.post("/ask", headers={"X-Eval-Token": "guess"}, json={"question": "What does B40 mean?"})
    assert llm_log == []


def test_nothing_is_logged_when_the_server_has_no_token(client, fake_llm, llm_log, monkeypatch):
    """An unset token must not match an empty or missing header."""
    monkeypatch.setattr(llm_logging, "EVAL_LOG_TOKEN", "")
    client.post("/ask", json={"question": "What does B40 mean?"})
    client.post("/ask", headers={"X-Eval-Token": ""}, json={"question": "What does B40 mean?"})
    client.post("/ask", headers=EVAL, json={"question": "What does B40 mean?"})
    assert llm_log == []


def test_every_logged_field_has_a_column_in_the_terraform_table():
    """A field with no column makes BigQuery reject the whole row."""
    import os
    import re
    tf = open(os.path.join(os.path.dirname(__file__), "..", "..", "infra", "terraform", "bigquery.tf"),
              encoding="utf-8").read()
    columns = set(re.findall(r'\{ name\s*=\s*"(\w+)"', tf))
    logged = {
        "call_id", "ts", "api_revision", "endpoint", "model", "status", "question", "question_chars",
        "state", "division", "horizon_months", "used_live_forecast_data", "forecast_version",
        "retrieval_ms", "retrieved_ids", "retrieved_distances", "generation_ms", "prompt_chars",
        "answer", "answer_chars", "finish_reason", "prompt_tokens", "output_tokens", "total_tokens",
        "error_type", "error_message", "total_ms", "context_type",
        "llm_calls", "tools_called", "tool_errors",
    }
    assert logged <= columns, f"no column for: {sorted(logged - columns)}"
