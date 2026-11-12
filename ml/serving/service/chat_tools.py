"""
The tools the chatbot can call.

App knowledge is retrieved for every question. On top of that, the model may
ask for data by calling one of the tools below ("function calling"): it replies
with a tool name and arguments, the API runs the function and sends the result
back, and the model then writes its answer from that result.

This replaced a keyword list that guessed whether a question needed forecast
numbers, and could only ever supply the one forecast open in the app.

Every tool reads the forecast table or scorecard already held in memory, so a
tool call costs milliseconds and no database query. Tools never raise: a bad
argument comes back as {"error": ...} so the model can correct itself or tell
the user.
"""

from service import forecasts
from service.config import CATEGORIES, STATES, logger

_CATEGORY_BY_NAME = {name.lower(): code for code, name in CATEGORIES.items()}


def category_code(value):
    """Accept a category name ("Transport") or its code ("07"). None if unknown."""
    v = str(value or "").strip()
    if v in CATEGORIES:
        return v
    if v.zfill(2) in CATEGORIES:
        return v.zfill(2)
    return _CATEGORY_BY_NAME.get(v.lower())


def horizon(value) -> int:
    """Months ahead, 1 to 12. Anything unusable becomes 6."""
    try:
        return max(1, min(12, int(value)))
    except (TypeError, ValueError):
        return 6


def _pct_from_base(value: float, base: float) -> float:
    return round((value / base - 1) * 100, 2)


def _row_at(rows: list, h: int):
    return next((r for r in rows if r["h"] == h), None)


def _unknown_state(state) -> dict:
    return {"error": f"Unknown state '{state}'. Valid states: {', '.join(STATES)}."}


def _unknown_category(category) -> dict:
    return {"error": f"Unknown category '{category}'. Valid categories: {'; '.join(CATEGORIES.values())}."}


def tool_get_forecast(state=None, category=None, horizon_months=6) -> dict:
    code = category_code(category)
    if state not in STATES:
        return _unknown_state(state)
    if code is None:
        return _unknown_category(category)
    h = horizon(horizon_months)
    forecasts.maybe_refresh_forecast_table()
    rows = forecasts.forecast_rows.get((state, code))
    if not rows:
        return {"error": f"No forecast is available for {CATEGORIES[code]} in {state}."}
    return {
        "state": state,
        "category": CATEGORIES[code],
        "horizon_months": h,
        "latest_actual_month": str(rows[0]["base_date"])[:7],
        "change_over_past_12_months_pct": rows[0].get("past_12m_change_pct"),
        "forecast": [
            {"month": str(r["target_date"])[:7],
             "expected_change_pct": _pct_from_base(r["predicted_index"], r["base_index"]),
             "range_low_pct": _pct_from_base(r["lower_bound"], r["base_index"]),
             "range_high_pct": _pct_from_base(r["upper_bound"], r["base_index"])}
            for r in rows if r["h"] <= h
        ],
        "range_confidence_pct": round((forecasts.forecast_meta.get("interval_level") or 0.8) * 100),
        "note": "Each figure is the percent change in the price index compared with the latest "
                "actual month. These are not ringgit amounts.",
    }


def _ranking(series: list, h: int) -> list:
    """[(label, rows)] -> rows ranked by forecast change at h months, biggest rise first."""
    out = []
    for label, rows in series:
        row = _row_at(rows or [], h)
        if row is None:
            continue
        out.append({"name": label,
                    "forecast_change_pct": _pct_from_base(row["predicted_index"], row["base_index"]),
                    "change_over_past_12_months_pct": rows[0].get("past_12m_change_pct")})
    return sorted(out, key=lambda x: x["forecast_change_pct"], reverse=True)


def tool_rank_categories(state=None, horizon_months=6) -> dict:
    if state not in STATES:
        return _unknown_state(state)
    h = horizon(horizon_months)
    forecasts.maybe_refresh_forecast_table()
    ranked = _ranking([(name, forecasts.forecast_rows.get((state, code))) for code, name in CATEGORIES.items()], h)
    if not ranked:
        return {"error": f"No forecasts are available for {state}."}
    return {"state": state, "horizon_months": h, "categories_biggest_rise_first": ranked,
            "note": "forecast_change_pct is the expected percent change in each category's price "
                    f"index over the next {h} months."}


def tool_rank_states(category=None, horizon_months=6) -> dict:
    code = category_code(category)
    if code is None:
        return _unknown_category(category)
    h = horizon(horizon_months)
    forecasts.maybe_refresh_forecast_table()
    ranked = _ranking([(state, forecasts.forecast_rows.get((state, code))) for state in STATES], h)
    if not ranked:
        return {"error": f"No forecasts are available for {CATEGORIES[code]}."}
    return {"category": CATEGORIES[code], "horizon_months": h, "states_biggest_rise_first": ranked,
            "note": "This compares how fast prices are expected to change in each state. It does "
                    "not say which state is more expensive."}


def tool_get_forecast_accuracy() -> dict:
    data = forecasts.load_scorecard()
    if not data:
        return {"error": "No track record has been published yet."}
    return {
        "latest_actual_month": str(data.get("latest_actual_date"))[:7],
        "forecasts_scored": data.get("scored_forecasts"),
        "by_months_ahead": data.get("by_horizon"),
        "meaning": "mae_pp is the average error in percentage points. rel_mae below 1 means the "
                   "forecasts were more accurate than assuming prices stay the same. coverage_80 "
                   "is the share of actual values that fell inside the 80% range (ideal: 0.80). "
                   "Only horizons that have reached their target month are listed.",
    }


ASK_TOOL_FUNCTIONS = {
    "get_forecast": tool_get_forecast,
    "rank_categories": tool_rank_categories,
    "rank_states": tool_rank_states,
    "get_forecast_accuracy": tool_get_forecast_accuracy,
}

_STATE_PARAM = {"type": "string", "enum": STATES, "description": "Malaysian state or federal territory."}
_CATEGORY_PARAM = {"type": "string", "enum": list(CATEGORIES.values()), "description": "Spending category."}
_HORIZON_PARAM = {"type": "integer", "description": "Months ahead, 1 to 12. Default 6."}

# What the model is told about each tool.
ASK_TOOLS = [{"function_declarations": [
    {
        "name": "get_forecast",
        "description": "Month-by-month price forecast for ONE spending category in ONE state, with its "
                       "range and the change over the past 12 months. Call it once per state or "
                       "category when comparing two or three of them.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {"state": _STATE_PARAM, "category": _CATEGORY_PARAM, "horizon_months": _HORIZON_PARAM},
            "required": ["state", "category"],
        },
    },
    {
        "name": "rank_categories",
        "description": "All spending categories in one state, ranked by how much their prices are "
                       "forecast to rise. Use for 'which category will rise most in <state>'.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {"state": _STATE_PARAM, "horizon_months": _HORIZON_PARAM},
            "required": ["state"],
        },
    },
    {
        "name": "rank_states",
        "description": "All states ranked by how much one category's prices are forecast to rise. "
                       "Use for 'which state will see the biggest rise in <category>'.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {"category": _CATEGORY_PARAM, "horizon_months": _HORIZON_PARAM},
            "required": ["category"],
        },
    },
    {
        "name": "get_forecast_accuracy",
        "description": "The live track record: how the forecasts this app actually published compared "
                       "with the official figures released afterwards.",
        "parameters_json_schema": {"type": "object", "properties": {}},
    },
]}]


def run_tool(name: str, args: dict) -> dict:
    """Run one tool the model asked for. Never raises."""
    fn = ASK_TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        return fn(**args)
    except TypeError:
        return {"error": f"Wrong arguments for {name}: {sorted(args)}."}
    except Exception as e:                       # a tool must not take the whole answer down
        logger.warning(f"Tool {name} failed: {e.__class__.__name__}: {e}")
        return {"error": f"{name} is unavailable right now."}
