"""
The user's household profile: state, income, pay rise, horizon and spending.

Pure functions (no Streamlit) so the URL round-trip can be unit-tested
(see test_user_profile.py). The profile is stored in the page URL so a
refresh keeps it and a link can be bookmarked or shared.

URL keys (short on purpose, to keep links readable):
  s  = state            inc = monthly income (RM)    pr = pay rise (%), optional
  h  = horizon months   sp  = spending as "01:900,04:700" (division:RM)
"""

from household_spending import HES_2024_MONTHLY_RM, HES_2024_TOTAL_RM

DIVISIONS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]
HORIZONS = (3, 6, 12)

# Example starting amounts, not survey data: the user is asked to replace them.
EXAMPLE_SPENDING = {
    "01": 900, "04": 700, "07": 400, "11": 300,
    "10": 150, "06": 150, "03": 150, "09": 150,
}

MAX_INCOME = 50_000
MAX_CATEGORY = 10_000


def typical_spending(state: str) -> dict:
    """
    The average household's monthly spending per category in a state (DOSM
    Household Expenditure Survey 2024). Falls back to the Malaysia average.
    """
    row = HES_2024_MONTHLY_RM.get(state, HES_2024_MONTHLY_RM["Malaysia"])
    return {d: int(row.get(d, 0)) for d in DIVISIONS}


def typical_total(state: str) -> int:
    return int(HES_2024_TOTAL_RM.get(state, HES_2024_TOTAL_RM["Malaysia"]))


def default_profile(default_state: str) -> dict:
    return {
        "state": default_state,
        "income": 4500,
        "pay_rise": None,
        "horizon": 12,
        "spending": {d: EXAMPLE_SPENDING.get(d, 0) for d in DIVISIONS},
    }


def _clamp_int(value, lo, hi):
    return max(lo, min(hi, int(round(float(value)))))


def encode_profile(p: dict) -> dict:
    """Profile -> URL query params (all strings). Zero spending is left out."""
    out = {
        "s": p["state"],
        "inc": str(int(p["income"])),
        "h": str(int(p["horizon"])),
        "sp": ",".join(f"{d}:{int(v)}" for d, v in p["spending"].items() if v > 0),
    }
    if p.get("pay_rise") is not None:
        out["pr"] = f"{float(p['pay_rise']):g}"
    return out


def decode_profile(params: dict, states: list, default_state: str) -> tuple[dict, bool]:
    """
    URL query params -> (profile, came_from_url). Anything missing or invalid
    falls back to the default, so a hand-edited or truncated link never crashes
    the app. came_from_url is True only if the link carried a spending profile.
    """
    p = default_profile(default_state)
    if not params or "sp" not in params:
        return p, False

    if params.get("s") in states:
        p["state"] = params["s"]
    try:
        p["income"] = _clamp_int(params.get("inc", p["income"]), 0, MAX_INCOME)
    except (TypeError, ValueError):
        pass
    try:
        if params.get("pr") not in (None, ""):
            p["pay_rise"] = max(-50.0, min(100.0, float(params["pr"])))
    except (TypeError, ValueError):
        pass
    try:
        h = int(params.get("h", p["horizon"]))
        if h in HORIZONS:
            p["horizon"] = h
    except (TypeError, ValueError):
        pass

    spending = {d: 0 for d in DIVISIONS}
    for item in str(params.get("sp", "")).split(","):
        div, _, amount = item.partition(":")
        if div in spending:
            try:
                spending[div] = _clamp_int(amount, 0, MAX_CATEGORY)
            except (TypeError, ValueError):
                continue
    p["spending"] = spending
    return p, True
