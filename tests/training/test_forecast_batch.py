"""Tests for ml/training/forecast_batch.py: forecast ranges, payload, live scorecard."""

import numpy as np
import pandas as pd
import pytest

import forecast_batch as fb


# ── build_table: ranges from calibration quantiles ───────────────────────────

def test_build_table_applies_log_error_quantiles():
    fc = pd.DataFrame({"state": "Selangor", "division": "01", "h": [1, 2],
                       "predicted_index": [100.0, 200.0]})
    q = pd.DataFrame({"h": [1, 2], "lo": [np.log(0.99), np.log(0.95)],
                      "hi": [np.log(1.02), np.log(1.10)], "n": [50, 50]})
    t = fb.build_table(fc, q)

    assert t["lower_bound"].tolist() == pytest.approx([99.0, 190.0])
    assert t["upper_bound"].tolist() == pytest.approx([102.0, 220.0])
    # half-width as a % of the forecast: (102-99)/2/100 and (220-190)/2/200
    assert t["confidence_band_pct"].tolist() == pytest.approx([1.5, 7.5])
    assert not {"lo", "hi", "n"} & set(t.columns)


def test_build_table_leaves_nan_when_a_horizon_was_not_calibrated():
    """main() refuses to publish such rows; this is what it checks for."""
    fc = pd.DataFrame({"h": [1, 2], "predicted_index": [100.0, 100.0]})
    q = pd.DataFrame({"h": [1], "lo": [-0.01], "hi": [0.01], "n": [10]})
    t = fb.build_table(fc, q)
    assert t["lower_bound"].isna().tolist() == [False, True]


# ── to_payload: the JSON the API serves ──────────────────────────────────────

def test_to_payload_shapes_rows_and_meta():
    table = pd.DataFrame({
        "state": ["Selangor"], "division": ["01"], "h": [1],
        "base_date": [pd.Timestamp("2026-09-01")], "target_date": [pd.Timestamp("2026-10-01")],
        "base_index": [130.12345], "past_12m_change_pct": [2.3456],
        "predicted_index": [130.56789], "lower_bound": [129.98765], "upper_bound": [131.23456],
        "confidence_band_pct": [0.4789],
    })
    skill = pd.Series({1: 1.0312, 3: 0.9712})
    p = fb.to_payload(table, skill, "20261103_030715")

    assert p["meta"]["version"] == "20261103_030715"
    assert p["meta"]["base_date"] == "2026-09-01"
    assert p["meta"]["interval_level"] == 0.80
    assert p["meta"]["calibration_rel_mae_vs_naive"] == {1: 1.031, 3: 0.971}
    row = p["rows"][0]
    assert row["base_date"] == "2026-09-01" and row["target_date"] == "2026-10-01"
    assert row["predicted_index"] == 130.568 and row["base_index"] == 130.123
    assert row["confidence_band_pct"] == 0.48 and row["past_12m_change_pct"] == 2.35


# ── score_published: the live track record ───────────────────────────────────

def _published(version, base_date, rows):
    return {"meta": {"version": version, "base_date": base_date},
            "rows": [{"state": "Selangor", "division": "01", "base_index": 100.0, **r} for r in rows]}


def _cpi(points):
    return pd.DataFrame([{"state": "Selangor", "division": "01", "date": pd.Timestamp(d), "cpi_index": v}
                         for d, v in points])


def test_score_published_compares_forecast_with_actual_and_naive():
    published = [_published("v1", "2026-08-01", [
        {"h": 1, "target_date": "2026-09-01", "predicted_index": 100.5, "lower_bound": 100.0, "upper_bound": 101.5},
    ])]
    card = fb.score_published(published, _cpi([("2026-08-01", 100.0), ("2026-09-01", 101.0)]))

    h1 = card["by_horizon"]["1"]
    assert card["scored_forecasts"] == 1
    assert h1["mae_pp"] == 0.5            # forecast 100.5 vs actual 101.0, on a base of 100
    assert h1["naive_mae_pp"] == 1.0      # "no change" said 100.0
    assert h1["rel_mae"] == 0.5
    assert h1["coverage_80"] == 1.0       # 101.0 is inside [100.0, 101.5]
    assert card["latest_actual_date"] == "2026-09-01"


def test_score_published_skips_targets_not_released_yet():
    published = [_published("v1", "2026-08-01", [
        {"h": 1, "target_date": "2026-09-01", "predicted_index": 101.0, "lower_bound": 100.0, "upper_bound": 102.0},
        {"h": 3, "target_date": "2026-11-01", "predicted_index": 103.0, "lower_bound": 101.0, "upper_bound": 105.0},
    ])]
    card = fb.score_published(published, _cpi([("2026-09-01", 101.0)]))
    assert card["scored_forecasts"] == 1
    assert list(card["by_horizon"]) == ["1"]


def test_score_published_keeps_only_the_newest_rerun_of_a_month():
    row = {"h": 1, "target_date": "2026-09-01", "lower_bound": 0.0, "upper_bound": 999.0}
    published = [
        _published("20260903_030000", "2026-08-01", [{**row, "predicted_index": 110.0}]),   # superseded
        _published("20260904_090000", "2026-08-01", [{**row, "predicted_index": 101.0}]),   # the re-run
    ]
    card = fb.score_published(published, _cpi([("2026-09-01", 101.0)]))
    assert card["scored_forecasts"] == 1
    assert card["by_horizon"]["1"]["mae_pp"] == 0.0
    assert card["published_months"] == ["2026-08-01"]


def test_score_published_counts_misses_outside_the_range():
    rows = [{"h": 1, "target_date": "2026-09-01", "predicted_index": 100.0, "lower_bound": 99.5, "upper_bound": 100.5}]
    card = fb.score_published([_published("v1", "2026-08-01", rows)], _cpi([("2026-09-01", 102.0)]))
    assert card["by_horizon"]["1"]["coverage_80"] == 0.0


def test_score_published_rel_mae_is_none_when_prices_did_not_move():
    rows = [{"h": 1, "target_date": "2026-09-01", "predicted_index": 100.2, "lower_bound": 99.0, "upper_bound": 101.0}]
    card = fb.score_published([_published("v1", "2026-08-01", rows)], _cpi([("2026-09-01", 100.0)]))
    assert card["by_horizon"]["1"]["rel_mae"] is None     # naive error is 0: no division by zero


def test_score_published_with_nothing_scoreable_returns_an_empty_card():
    card = fb.score_published([], _cpi([("2026-09-01", 100.0)]))
    assert card["scored_forecasts"] == 0 and card["by_horizon"] == {}
