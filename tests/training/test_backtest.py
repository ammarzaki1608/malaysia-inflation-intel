"""
Tests for ml/training/backtest.py: baselines, features, metrics and leakage.

Tests marked `slow` train real models on synthetic data (about a minute each).
Skip them with:  pytest -m "not slow"
"""

import numpy as np
import pandas as pd
import pytest

import backtest as bt
import forecast_batch as fb


# ── baseline forecasters ─────────────────────────────────────────────────────

def _history():
    """24 months x 2 series: one flat, one growing 1% a month."""
    t = np.arange(24)
    return np.column_stack([np.full(24, 100.0), 100.0 * 1.01 ** t])


def test_naive_repeats_the_last_value():
    P = bt.fc_naive(_history())
    assert P.shape == (2, bt.MAX_H)
    assert np.allclose(P[0], 100.0)
    assert np.allclose(P[1], 100.0 * 1.01 ** 23)


def test_drift_continues_the_last_12_months_of_growth():
    Y = _history()
    P = bt.fc_drift(Y)
    assert np.allclose(P[0], 100.0)                                         # flat stays flat
    assert np.allclose(P[1], Y[-1, 1] * 1.01 ** np.arange(1, 13))           # 1%/month continues


def test_seasonal_naive_repeats_last_years_change():
    Y = _history()
    P = bt.fc_snaive_growth(Y)
    assert P.shape == (2, bt.MAX_H)
    assert np.allclose(P[1], Y[-1, 1] * 1.01 ** np.arange(1, 13))


# ── features: the moving-average leak that started all this ──────────────────

def _one_series(n=20):
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    cpi = pd.DataFrame({"date": dates, "state": "S", "division": "01",
                        "cpi_index": np.arange(1.0, n + 1)})
    ppi = pd.DataFrame({"date": dates, "ppi_index": 100.0})
    opr = pd.DataFrame({"date": dates, "opr_rate": 3.0})
    return cpi, bt.build_exog(ppi, opr)


def test_fixed_moving_average_excludes_the_month_being_predicted():
    cpi, exog = _one_series()
    f = bt.build_features(cpi, exog, "fixed")
    row = f.iloc[10]                       # cpi_index = 11; previous three are 8, 9, 10
    assert row["cpi_ma_3"] == pytest.approx(9.0)
    assert row["cpi_ma_12"] == pytest.approx(np.mean(np.arange(1, 11)))
    assert row["cpi_lag_1"] == 10.0 and row["cpi_lag_3"] == 8.0


def test_leaky_mode_reproduces_the_original_bug():
    cpi, exog = _one_series()
    f = bt.build_features(cpi, exog, "leaky")
    assert f.iloc[10]["cpi_ma_3"] == pytest.approx(10.0)    # mean(9, 10, 11) includes the target


def test_build_features_rejects_unknown_mode():
    cpi, exog = _one_series()
    with pytest.raises(ValueError):
        bt.build_features(cpi, exog, "typo")


def test_direct_features_never_look_past_the_origin():
    """Changing the future must not change any feature at or before the origin."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=90, freq="MS")
    series = pd.MultiIndex.from_product([["A", "B"], ["01", "02"]], names=["state", "division"])
    Y = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.004, (90, 4)), axis=0))
    exog = pd.DataFrame({"ppi_index": 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 90))),
                         "opr_rate": 3.0}, index=dates)
    origin = 70

    clean = bt.build_direct_features(Y, series, dates, exog)
    Y2, exog2 = Y.copy(), exog.copy()
    Y2[origin + 1:] *= 50
    exog2.iloc[origin + 1:] *= 50
    poisoned = bt.build_direct_features(Y2, series, dates, exog2)

    numeric = [k for k in clean if not k.startswith("_")]
    assert numeric, "no feature blocks were built"
    for k in numeric:
        assert np.allclose(clean[k][: origin + 1], poisoned[k][: origin + 1], equal_nan=True), k

    # and the per-row frame the model sees at the origin is identical too
    s_idx, t_idx = np.arange(4), np.full(4, origin)
    for h in (1, 6, 12):
        a = bt._direct_rows(clean, h, t_idx, s_idx)
        b = bt._direct_rows(poisoned, h, t_idx, s_idx)
        pd.testing.assert_frame_equal(a, b)


# ── metrics ──────────────────────────────────────────────────────────────────

def _results():
    """Two origins, one series. Truth rises 2 index points; naive says no change."""
    rows = []
    for origin in ("2024-01-01", "2024-02-01"):
        base = dict(origin=pd.Timestamp(origin), h=1, state="S", division="01",
                    y_origin=100.0, y_true=102.0, scale=1.0)
        rows += [dict(base, model="naive", y_pred=100.0),
                 dict(base, model="perfect", y_pred=102.0),
                 dict(base, model="half", y_pred=101.0),
                 dict(base, model="wrong_way", y_pred=99.0)]
    return pd.DataFrame(rows)


def test_score_relative_error_and_direction():
    s = bt.score(_results())
    s = s[s["period"] == "all"].set_index("model")

    assert s.loc["naive", "rel_mae"] == pytest.approx(1.0)
    assert s.loc["naive", "mae_pp"] == pytest.approx(2.0)
    assert np.isnan(s.loc["naive", "dir_acc"])               # naive never predicts a direction
    assert s.loc["perfect", "rel_mae"] == pytest.approx(0.0)
    assert s.loc["perfect", "dir_acc"] == pytest.approx(1.0)
    assert s.loc["half", "rel_mae"] == pytest.approx(0.5)
    assert s.loc["wrong_way", "rel_mae"] == pytest.approx(1.5)
    assert s.loc["wrong_way", "dir_acc"] == pytest.approx(0.0)
    assert (s["n"] == 2).all()


def test_score_ignores_unchanged_months_in_directional_accuracy():
    r = _results()
    flat = r[r["model"] == "half"].assign(y_true=100.0)       # price did not move
    s = bt.score(pd.concat([r[r["model"] == "naive"].assign(y_true=100.0), flat]))
    s = s[s["period"] == "all"].set_index("model")
    assert np.isnan(s.loc["half", "dir_acc"])


def test_report_renders_a_table_per_metric():
    md = bt.report(bt.score(_results()), {"origins_with_missing_ppi": 0}, "synthetic")
    assert "Synthetic data" in md
    for metric in ("**rel_mae**", "**mae_pp**", "**mase**", "**dir_acc**"):
        assert metric in md
    assert "| naive | 1.00 |" in md


# ── synthetic data (used by every slow test below) ───────────────────────────

def test_synthetic_data_has_the_shape_of_the_real_thing():
    cpi, ppi, opr = bt.make_synthetic()
    assert cpi["state"].nunique() == 16
    assert set(cpi["division"]) == {f"{i:02d}" for i in range(1, 14)} | {"overall"}
    assert not cpi["cpi_index"].isna().any() and (cpi["cpi_index"] > 0).all()
    assert opr["date"].max() < cpi["date"].max()             # OPR is published late, as in production
    cpi2, _, _ = bt.make_synthetic()
    pd.testing.assert_frame_equal(cpi, cpi2)                 # seeded: reproducible


# ── slow: real models ────────────────────────────────────────────────────────

@pytest.mark.slow
def test_no_model_uses_data_after_the_forecast_origin():
    """
    The leakage self-test: multiply all data after the origin by 50 and check
    that no model's forecast changes. Raises AssertionError on any leak.
    """
    bt.self_test()


@pytest.mark.slow
def test_monthly_batch_end_to_end_on_synthetic_data():
    """calibrate -> forecast_now -> build_table -> to_payload, as main() runs them."""
    cpi, ppi, opr = bt.make_synthetic()
    cpi = cpi[cpi["state"].isin(["State_00", "State_01"])]

    q, skill = fb.calibrate(cpi, ppi, opr, months=15)
    assert q["h"].tolist() == list(range(1, 13))
    assert (q["lo"] < q["hi"]).all()
    assert q["lo"].is_monotonic_decreasing and q["hi"].is_monotonic_increasing   # ranges never shrink

    fc, _ = fb.forecast_now(cpi, ppi, opr)
    table = fb.build_table(fc, q)

    assert len(table) == 2 * 13 * 12                          # 2 states x 13 categories x 12 months
    assert "overall" not in set(table["division"])
    assert not table[["predicted_index", "lower_bound", "upper_bound"]].isna().any().any()
    assert (table["lower_bound"] < table["upper_bound"]).all()
    assert (table["base_date"] == cpi["date"].max()).all()
    # a monthly CPI forecast should stay within a sane distance of today's level
    assert ((table["predicted_index"] / table["base_index"]).between(0.8, 1.25)).all()

    payload = fb.to_payload(table, skill, "test")
    assert len(payload["rows"]) == len(table)
    assert set(payload["meta"]["calibration_rel_mae_vs_naive"]) == set(range(1, 13))
