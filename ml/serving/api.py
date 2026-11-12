"""
FastAPI service for Real Ringgit (personal inflation forecasts for Malaysia).

Serves CPI forecasts precomputed monthly by ml/training/forecast_batch.py
(ETS + direct-growth XGBoost average, with 80% ranges calibrated on backtest
errors). The table is loaded from GCS at startup and re-checked hourly.

This file is the entry point (`uvicorn api:app`) and holds the data endpoints.
The logic behind them is in the `service` package next to it.

Endpoints:
    GET  /health                — liveness check
    GET  /states, /categories   — the valid states and spending categories
    GET  /forecast              — 3/6/12-month CPI forecast with 80% range
    GET  /forecasts             — the same for several categories in one call
    GET  /forecast-accuracy     — live track record of the published forecasts
    GET  /trends                — historical CPI, PPI and OPR for one series
    GET  /income-snapshot       — household income survey figures for a state
    GET  /category-leaderboard  — national year-on-year change by category
    GET  /state-leaderboard     — year-on-year change by state for one category
    GET  /headline-inflation    — national headline inflation
    POST /ask                   — the chatbot (service/chat.py)
    POST /prescriptive-insight  — a short budgeting suggestion (service/chat.py)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from service import chat, data, forecasts
from service.config import CATEGORIES, PROJECT_ID, STATES, limiter, logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the forecast table and the RAG index before serving."""
    try:
        forecasts.load_forecast_table(force=True)
    except Exception as e:
        # Keep serving: everything else still works, forecast requests get a
        # 503, and the table is tried again within a minute.
        logger.warning(f"Forecast table not loaded at startup: {e}")
    chat.load_index_from_gcs()
    yield


app = FastAPI(
    title="Real Ringgit API",
    description="Predicts personal CPI inflation for Malaysian citizens",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


def _check_state(state: str) -> None:
    if state not in STATES:
        raise HTTPException(status_code=400, detail=f"Unknown state: {state}")


def _check_division(division: str) -> None:
    if division not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown division code: {division}")


def _check_horizon(horizon_months: int) -> None:
    if horizon_months not in (3, 6, 12):
        raise HTTPException(status_code=400, detail="horizon_months must be 3, 6, or 12")


def _yoy_pct(latest: float, year_ago: float) -> float:
    return round((latest - year_ago) / year_ago * 100, 2)


# ── Reference ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness check, with what is currently loaded."""
    return {
        "status": "ok",
        "forecast_table_loaded": bool(forecasts.forecast_rows),
        "forecast_version": forecasts.forecast_meta.get("version"),
        "forecast_base_date": forecasts.forecast_meta.get("base_date"),
        "project": PROJECT_ID,
    }


@app.get("/states")
async def list_states():
    """Return list of all available Malaysian states."""
    return {"states": STATES}


@app.get("/categories")
async def list_categories():
    """Return list of all CPI categories with division codes."""
    return {"categories": [{"code": code, "name": name} for code, name in CATEGORIES.items()]}


# ── Forecasts ────────────────────────────────────────────────────────────────

@app.get("/forecast")
@limiter.limit("30/minute")
async def forecast(request: Request, state: str, division: str, horizon_months: int = 6):
    """
    Multi-month CPI forecast for one state and category.

    Example:
        GET /forecast?state=Kedah&division=01&horizon_months=6

    Returns monthly predictions with 80% ranges calibrated on backtest
    errors. Each step carries base_index (the latest real CPI) so clients can
    measure change from today rather than from the first forecast month.
    """
    _check_state(state)
    _check_division(division)
    _check_horizon(horizon_months)
    return forecasts.forecast_payload(state, division, horizon_months)


@app.get("/forecasts")
@limiter.limit("30/minute")
async def forecasts_bulk(request: Request, state: str, divisions: str, horizon_months: int = 6):
    """
    Forecasts for several categories of one state in ONE request.

    The app needs every category a user spends on (up to 13, or 26 for the
    Relocation Planner). Calling /forecast per category burned through the
    per-IP rate limit after a click or two — and all app users share the app
    server's IP. Example: GET /forecasts?state=Selangor&divisions=01,04,07&horizon_months=6
    """
    _check_state(state)
    _check_horizon(horizon_months)
    divs = [d.strip() for d in divisions.split(",") if d.strip()]
    bad = [d for d in divs if d not in CATEGORIES]
    if not divs or bad:
        raise HTTPException(status_code=400, detail=f"Unknown or missing division codes: {bad or divisions}")

    out, errors = {}, {}
    for d in dict.fromkeys(divs):          # de-duplicate, keep order
        try:
            out[d] = forecasts.forecast_payload(state, d, horizon_months)
        except HTTPException as e:
            errors[d] = e.detail
    return {"state": state, "horizon_months": horizon_months, "forecasts": out, "errors": errors}


@app.get("/forecast-accuracy")
@limiter.limit("30/minute")
async def forecast_accuracy(request: Request):
    """
    Live, out-of-sample track record of the forecasts actually published:
    each month's published forecasts scored against the CPI released since.
    Built by forecast_batch.py; refreshed hourly. Empty until the first
    published forecasts reach their target months.
    """
    card = forecasts.load_scorecard()
    if card is None:
        raise HTTPException(status_code=404, detail="No scorecard published yet.")
    return card


# ── Historical data (BigQuery) ───────────────────────────────────────────────

@app.get("/trends")
@limiter.limit("30/minute")
async def trends(request: Request, state: str, division: str, months_back: int = 60):
    """
    The recorded CPI, PPI and OPR history for one series, for charting.
    Not a forecast.

    Example:
        GET /trends?state=Selangor&division=01&months_back=60
    """
    _check_state(state)
    _check_division(division)
    if months_back not in (24, 36, 60, 120):
        raise HTTPException(status_code=400, detail="months_back must be 24, 36, 60, or 120")

    history = data.feature_history(state, division, months_back=months_back)
    if not history:
        raise HTTPException(status_code=404, detail=f"No historical data found for {state}/{division}")

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
    _check_state(state)
    snapshot = data.income_snapshot(state)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"No income data found for {state}")
    return snapshot


@app.get("/category-leaderboard")
@limiter.limit("30/minute")
async def category_leaderboard(request: Request):
    """
    Returns year-on-year % change for every CPI category, nationally,
    ranked fastest-rising first. Used for Data Explorer's overview section.
    """
    results = []
    for division, entries in data.national_cpi_by_division().items():
        if len(entries) < 13:
            continue
        results.append({
            "division": division,
            "category_name": CATEGORIES.get(division, division),
            "yoy_change_pct": _yoy_pct(entries[0]["index"], entries[12]["index"]),
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
    _check_division(division)

    forecasts.maybe_refresh_forecast_table()
    results = [
        {"state": st_, "yoy_change_pct": rows[0]["past_12m_change_pct"],
         "cpi_index": rows[0]["base_index"], "latest_date": rows[0]["base_date"]}
        for (st_, div), rows in forecasts.forecast_rows.items()
        if div == division and rows and rows[0].get("past_12m_change_pct") is not None
    ]

    if not results:  # forecast table unavailable -> compute from the mart
        for r in data.latest_cpi_by_state(division):
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
    The current national headline CPI year-on-year change, computed from the
    official raw.cpi_national table. The app uses it wherever it compares a
    personal figure with "official inflation".
    """
    rows = data.national_headline_cpi()
    if len(rows) < 13:
        raise HTTPException(status_code=404, detail="Not enough national CPI history to compute headline inflation")

    latest, year_ago = rows[0], rows[12]
    return {
        "headline_inflation_pct": _yoy_pct(latest["index"], year_ago["index"]),
        "latest_date": str(latest["date"]),
        "compared_to_date": str(year_ago["date"]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
