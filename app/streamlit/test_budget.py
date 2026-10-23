"""Unit tests for budget.py. Run: python -m unittest app/streamlit/test_budget.py"""

import unittest

from budget import cumulative_extra, rank_pressure, salary_check, what_if_savings


def cat(name, current, amounts):
    """A category forecast shaped like calculate_budget_forecast builds it."""
    final = amounts[-1]
    return {
        "category_name": name,
        "current_rm": current,
        "change_rm": round(final - current, 2),
        "change_pct": round((final - current) / current * 100, 2),
        "projected_amounts": amounts,
    }


# Food: big spend, small % rise. Clothing: small spend, big % rise.
CATS = {
    "01": cat("Food", 900, [905, 910, 918]),        # +18 RM, +2.0%
    "03": cat("Clothing", 50, [51, 52, 52.5]),      # +2.5 RM, +5.0%
    "07": cat("Transport", 400, [400, 398, 396]),   # -4 RM
}
TOTAL_NOW = 900 + 50 + 400


class RankPressure(unittest.TestCase):
    def test_ranks_by_ringgit_not_percent(self):
        rows = rank_pressure(CATS)
        self.assertEqual([r["name"] for r in rows], ["Food", "Clothing", "Transport"])
        # Clothing rises faster in %, but Food adds far more RM
        self.assertGreater(rows[1]["change_pct"], rows[0]["change_pct"])

    def test_shares_cover_only_rising_categories(self):
        rows = {r["name"]: r for r in rank_pressure(CATS)}
        self.assertAlmostEqual(rows["Food"]["share_pct"], round(18 / 20.5 * 100, 1))
        self.assertAlmostEqual(rows["Clothing"]["share_pct"], round(2.5 / 20.5 * 100, 1))
        self.assertEqual(rows["Transport"]["share_pct"], 0.0)

    def test_nothing_rising(self):
        rows = rank_pressure({"07": CATS["07"]})
        self.assertEqual(rows[0]["share_pct"], 0.0)


class CumulativeExtra(unittest.TestCase):
    def test_sums_every_month_not_just_the_last(self):
        # month totals: 1356, 1360, 1366.5 vs 1350 today -> 6 + 10 + 16.5
        self.assertAlmostEqual(cumulative_extra(CATS, TOTAL_NOW), 32.5)


class SalaryCheck(unittest.TestCase):
    def test_pay_fell_behind(self):
        out = salary_check(3.0, 2.0, 4000, 1350, 1366.5)
        self.assertEqual(out["real_change_pts"], -1.0)
        self.assertEqual(out["needed_raise_rm"], 16.5)
        self.assertAlmostEqual(out["needed_raise_pct"], round(16.5 / 4000 * 100, 2))

    def test_no_pay_rise_given(self):
        out = salary_check(3.0, None, 4000, 1350, 1366.5)
        self.assertIsNone(out["real_change_pts"])
        self.assertEqual(out["needed_raise_rm"], 16.5)

    def test_falling_costs_need_no_raise(self):
        out = salary_check(1.0, 0.0, 4000, 1350, 1340)
        self.assertEqual(out["needed_raise_rm"], 0.0)

    def test_zero_salary_has_no_percentage(self):
        self.assertIsNone(salary_check(1.0, 0.0, 0, 1350, 1366.5)["needed_raise_pct"])


class WhatIf(unittest.TestCase):
    def test_cut_food_by_ten_percent(self):
        out = what_if_savings(CATS, {"01": 10})
        self.assertAlmostEqual(out["monthly_rm"], 91.8)                       # 10% of 918
        self.assertAlmostEqual(out["horizon_rm"], round((905 + 910 + 918) * 0.1, 2))

    def test_no_cuts_and_unknown_categories(self):
        self.assertEqual(what_if_savings(CATS, {"01": 0, "99": 20}), {"monthly_rm": 0.0, "horizon_rm": 0.0})


if __name__ == "__main__":
    unittest.main()
