"""
Budget answers: pure functions behind the personal calculator's four answers.

No Streamlit and no API calls, so they can be unit-tested on their own
(see test_budget.py). Each category forecast is a dict with:
  category_name, current_rm, change_rm, change_pct, projected_amounts (months 1..h)
"""


def rank_pressure(category_forecasts: dict) -> list:
    """
    Categories ordered by the extra RM per month they add by the end of the
    horizon (spending x expected rise), with each one's share of the total
    increase. Only categories that add cost count towards the share.
    """
    rows = [
        {"div": div, "name": c["category_name"], "change_rm": c["change_rm"],
         "change_pct": c["change_pct"], "current_rm": c["current_rm"]}
        for div, c in category_forecasts.items()
    ]
    rows.sort(key=lambda r: r["change_rm"], reverse=True)
    total_up = sum(r["change_rm"] for r in rows if r["change_rm"] > 0)
    for r in rows:
        r["share_pct"] = (
            round(r["change_rm"] / total_up * 100, 1)
            if total_up > 0 and r["change_rm"] > 0 else 0.0
        )
    return rows


def cumulative_extra(category_forecasts: dict, total_current: float) -> float:
    """Extra RM needed over months 1..h combined, versus spending today."""
    months = min(len(c["projected_amounts"]) for c in category_forecasts.values())
    return sum(
        sum(c["projected_amounts"][m] for c in category_forecasts.values()) - total_current
        for m in range(months)
    )


def salary_check(personal_past_12m_pct, pay_rise_pct, monthly_salary,
                 total_current, total_projected_final) -> dict:
    """
    Is pay keeping up?
    - backward: last 12 months' pay rise vs the user's own inflation over the
      same 12 months (real income change, in percentage points)
    - forward: the monthly raise needed to keep today's leftover unchanged once
      spending reaches its projected level
    """
    out = {"real_change_pts": None, "needed_raise_rm": None, "needed_raise_pct": None}
    if pay_rise_pct is not None and personal_past_12m_pct is not None:
        out["real_change_pts"] = round(pay_rise_pct - personal_past_12m_pct, 2)
    extra = total_projected_final - total_current
    out["needed_raise_rm"] = round(max(extra, 0.0), 2)
    if monthly_salary > 0:
        out["needed_raise_pct"] = round(max(extra, 0.0) / monthly_salary * 100, 2)
    return out


def what_if_savings(category_forecasts: dict, cuts_pct: dict) -> dict:
    """
    Effect of spending less in some categories, assuming prices still follow
    the forecast. cuts_pct = {division: percent to cut}. Returns the RM saved
    in the final forecast month and over the whole horizon.
    """
    monthly, total = 0.0, 0.0
    for div, cut in cuts_pct.items():
        c = category_forecasts.get(div)
        if not c or cut <= 0:
            continue
        share = cut / 100
        monthly += c["projected_amounts"][-1] * share
        total += sum(c["projected_amounts"]) * share
    return {"monthly_rm": round(monthly, 2), "horizon_rm": round(total, 2)}
