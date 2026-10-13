"""
Rolling-origin backtest for the CPI forecaster.

WHY THIS EXISTS
---------------
train.py measures one-step-ahead error on rows whose features were built
from real data -- including cpi_ma_3 / cpi_ma_12 windows that contain the
month being predicted (target leakage). The API, however, forecasts 3/6/12
months ahead *recursively*, feeding its own predictions back in, with moving
averages built from past months only. So the published metrics describe a
situation the deployed system never faces.

This script answers the question users actually care about:

    "Standing at month T with only the data that existed at T, how far off
     is the forecast for T+h -- and is it better than a naive guess?"

HOW IT WORKS
------------
For every forecast origin T (monthly, from --start to the last month):
  1. Truncate the data to dates <= T. Nothing after T is ever visible.
  2. (Re)train each model on that truncated data only.
  3. Forecast h = 1..12 months ahead for every state x division series,
     using exactly the procedure ml/serving/api.py uses.
  4. Compare with what actually happened at T+h.

Models compared
  naive            y[T+h] = y[T]                      (the bar to beat)
  snaive_growth    repeat last year's change over the same months
  drift            continue the average monthly growth of the last 12 months
  ets              per-series AutoETS (statsforecast), if installed
  xgb_as_deployed  production XGBoost: leaky MA training + API recursion
  xgb_leak_fixed   same, but MA windows exclude the target month
  xgb_leaky_onestep  diagnostic only (h=1): scores the as-deployed model the
                   way train.py did, with leaky features. The gap between this
                   row and xgb_as_deployed at h=1 is the size of the leak.

Metrics (per model x horizon x period)
  mae_pp   mean absolute error of the cumulative % change from T to T+h,
           in percentage points. "Off by 0.4pp over 6 months."
  rmse_pp  same, root-mean-square (punishes big misses).
  mase     error scaled by each series' in-sample naive 1-step error.
  rel_mae  MAE / naive MAE at the same horizon. < 1.0 beats naive.
  dir_acc  share of cases where the forecast got the direction of the
           T -> T+h change right, computed WITHIN each series and only
           where the actual change was non-zero.

USAGE
-----
  python backtest.py --source bigquery            # your dbt staging views
  python backtest.py --source dosm                # public DOSM parquet, no GCP needed
  python backtest.py --source synthetic           # fake data, for testing the code
  python backtest.py --self-test                  # proves no future data leaks in

  Common options: --start 2018-01  --retrain-every 12  --ets-every 3  --no-ets
                  --out backtest_results
  Runtime is dominated by ETS (~50 ms per series per fit); more CPU cores help.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")

# ── Production constants (keep in sync with train.py / api.py) ───────────────

FEATURES = [
    "cpi_lag_1", "cpi_lag_3", "cpi_lag_6", "cpi_lag_12",
    "cpi_ma_3", "cpi_ma_12",
    "ppi_index", "ppi_lag_1", "ppi_lag_2", "ppi_lag_3",
    "opr_rate",
]
XGB_PARAMS = {
    "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1,
}
COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31")
API_HISTORY_ROWS = 15     # api.get_feature_history(months_back=15)
API_MIN_ROWS = 13         # api.recursive_forecast raises 404 below this
MAX_H = 12

PERIODS = {
    "all": (None, None),
    "pre_covid": (None, "2020-02-01"),
    "covid": ("2020-03-01", "2021-12-01"),
    "post_covid": ("2022-01-01", None),
}

DOSM_URLS = {
    "cpi_state": "https://storage.dosm.gov.my/cpi/cpi_2d_state.parquet",
    "ppi": "https://storage.dosm.gov.my/ppi/ppi.parquet",
    "rates": "https://storage.data.gov.my/finsector/interest_rates.parquet",
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
#    Every loader returns the same two frames:
#      cpi : date, state, division, cpi_index      (long format)
#      ppi : date, ppi_index                       (national, monthly)
#      opr : date, opr_rate
# ═════════════════════════════════════════════════════════════════════════════

def _month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def load_bigquery(project: str, dataset: str):
    """Read the same dbt staging views the mart is built from."""
    from google.cloud import bigquery
    client = bigquery.Client(project=project)
    q = lambda sql: client.query(sql).to_dataframe()
    cpi = q(f"select date, state, division, cpi_index from `{project}.{dataset}.stg_cpi_state`")
    ppi = q(f"select date, ppi_index from `{project}.{dataset}.stg_ppi_headline`")
    opr = q(f"select date, opr_rate from `{project}.{dataset}.stg_opr`")
    return _clean(cpi, ppi, opr)


def load_dosm():
    """Download the public parquet files and apply the staging-layer filters."""
    import requests

    def get(url):
        logger.info(f"Downloading {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return pd.read_parquet(io.BytesIO(r.content))

    raw_cpi, raw_ppi, raw_rates = (get(DOSM_URLS[k]) for k in ("cpi_state", "ppi", "rates"))

    # stg_cpi_state
    cpi = raw_cpi.dropna(subset=["date", "state", "division", "index"]).rename(
        columns={"index": "cpi_index"})[["date", "state", "division", "cpi_index"]]
    # stg_ppi_headline: series = 'abs'
    ppi = raw_ppi[raw_ppi["series"] == "abs"].rename(
        columns={"index": "ppi_index"})[["date", "ppi_index"]]
    # stg_opr: rate = 'br', bank = 'commercial'
    rates = raw_rates.dropna(subset=["date", "value"])
    opr = rates[(rates["rate"] == "br") & (rates["bank"] == "commercial")].rename(
        columns={"value": "opr_rate"})[["date", "opr_rate"]]
    return _clean(cpi, ppi, opr)


def _clean(cpi, ppi, opr):
    cpi = cpi.copy()
    cpi["date"] = _month(cpi["date"])
    cpi["division"] = cpi["division"].astype(str)
    cpi["cpi_index"] = cpi["cpi_index"].astype(float)
    ppi = ppi.assign(date=_month(ppi["date"])).groupby("date", as_index=False)["ppi_index"].mean()
    opr = opr.assign(date=_month(opr["date"])).groupby("date", as_index=False)["opr_rate"].mean()
    return cpi.sort_values(["state", "division", "date"]).reset_index(drop=True), ppi, opr


def make_synthetic(seed: int = 7, end: str = "2026-08-01"):
    """
    CPI-like fake data, ONLY for exercising the code. Results on it mean
    nothing about Malaysia. It deliberately includes the features of real CPI
    that trip models up: upward trend, regime change (2022-23), policy step
    shocks, seasonality, rounding to 1 decimal, and OPR published late.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", end, freq="MS")
    T = len(dates)
    states = [f"State_{i:02d}" for i in range(16)]
    divs = [f"{i:02d}" for i in range(1, 14)]

    ppi_g = np.zeros(T)
    for t in range(1, T):
        ppi_g[t] = 0.001 + 0.4 * (ppi_g[t - 1] - 0.001) + rng.normal(0, 0.008)
    ppi = 100 * np.exp(np.cumsum(ppi_g))

    opr_steps = [("2010-01", 2.75), ("2011-05", 3.0), ("2014-07", 3.25), ("2016-07", 3.0),
                 ("2018-01", 3.25), ("2019-05", 3.0), ("2020-03", 2.5), ("2020-07", 1.75),
                 ("2022-05", 2.0), ("2022-09", 2.5), ("2023-05", 3.0), ("2025-07", 2.75)]
    opr = pd.Series(np.nan, index=dates)
    for d, v in opr_steps:
        opr[pd.Timestamp(d)] = v
    opr = opr.ffill()
    opr.iloc[-2:] = np.nan  # BNM publishes later than DOSM

    month = dates.month.values
    hi_infl = (dates >= "2022-01-01") & (dates <= "2023-06-01")
    shocks = {"2015-04-01": 0.012, "2018-06-01": -0.008, "2024-03-01": 0.004}

    rows = []
    for d_i, div in enumerate(divs):
        mu = rng.uniform(0.0005, 0.0035)
        g = np.zeros(T)
        for t in range(1, T):
            ppi_pass = 0.12 * ppi_g[t - 3] if t >= 3 else 0.0
            g[t] = mu + 0.5 * (g[t - 1] - mu) + ppi_pass + rng.normal(0, 0.002)
        g += 0.0025 * hi_infl * (div in ("01", "07", "11"))
        g += 0.0015 * np.sin(2 * np.pi * month / 12) * (div in ("01", "03"))
        if div == "07":
            g[(dates >= "2020-04-01") & (dates <= "2020-08-01")] -= 0.012
            g[dates == "2024-06-01"] += 0.03
        for d, s in shocks.items():
            g[dates == d] += s
        for s in states:
            gs = g + rng.normal(0, 0.0002) + rng.normal(0, 0.0015, T)
            idx = np.round(100 * np.exp(np.cumsum(gs)), 1)
            rows.append(pd.DataFrame({"date": dates, "state": s, "division": div, "cpi_index": idx}))

    cpi = pd.concat(rows, ignore_index=True)
    overall = cpi.groupby(["date", "state"], as_index=False)["cpi_index"].mean().round(1)
    cpi = pd.concat([cpi, overall.assign(division="overall")], ignore_index=True)
    return (cpi.sort_values(["state", "division", "date"]).reset_index(drop=True),
            pd.DataFrame({"date": dates, "ppi_index": np.round(ppi, 1)}),
            pd.DataFrame({"date": dates, "opr_rate": opr.values}).dropna())


# ═════════════════════════════════════════════════════════════════════════════
# 2. FEATURES — a pandas copy of models/mart/mart_cpi_features.sql
#    ma_mode="leaky": rows between 2 preceding and CURRENT row  (as in prod)
#    ma_mode="fixed": rows between 3 preceding and 1 preceding  (target excluded,
#                     which is also what api.build_features_from_history computes)
# ═════════════════════════════════════════════════════════════════════════════

def build_exog(ppi: pd.DataFrame, opr: pd.DataFrame) -> pd.DataFrame:
    p = ppi.sort_values("date").copy()
    for k in (1, 2, 3):                       # lag over PPI rows, as in SQL
        p[f"ppi_lag_{k}"] = p["ppi_index"].shift(k)
    return p.merge(opr, on="date", how="outer").sort_values("date").reset_index(drop=True)


def build_features(cpi: pd.DataFrame, exog: pd.DataFrame, ma_mode: str) -> pd.DataFrame:
    df = cpi.sort_values(["state", "division", "date"]).copy()
    g = df.groupby(["state", "division"], sort=False)["cpi_index"]
    for k in (1, 3, 6, 12):
        df[f"cpi_lag_{k}"] = g.shift(k)
    if ma_mode == "leaky":
        df["cpi_ma_3"] = g.transform(lambda s: s.rolling(3, min_periods=1).mean())
        df["cpi_ma_12"] = g.transform(lambda s: s.rolling(12, min_periods=1).mean())
    elif ma_mode == "fixed":
        df["cpi_ma_3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df["cpi_ma_12"] = g.transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean())
    else:
        raise ValueError(ma_mode)
    df["is_covid_period"] = df["date"].between(COVID_START, COVID_END)
    return df.merge(exog, on="date", how="left")


def prepare_training(feat: pd.DataFrame, cutoff: pd.Timestamp):
    """
    prepare_features.handle_nulls + COVID exclusion, restricted to date <= cutoff.
    One deliberate difference: medians are computed from training rows only
    (production used the whole table, a small leak of its own).
    """
    df = feat[feat["date"] <= cutoff].sort_values("date").copy()
    df["opr_rate"] = df["opr_rate"].ffill()
    df = df.dropna(subset=["cpi_lag_1"])
    medians = df[FEATURES].median()
    df[FEATURES] = df[FEATURES].fillna(medians)
    df = df[~df["is_covid_period"]]
    return df, medians


# ═════════════════════════════════════════════════════════════════════════════
# 3. FORECASTERS
#    Every forecaster takes the wide history Y (dates x series) truncated at
#    the origin and returns an (n_series, MAX_H) array of predicted index levels.
# ═════════════════════════════════════════════════════════════════════════════

def fc_naive(Yhist: np.ndarray) -> np.ndarray:
    return np.repeat(Yhist[-1][:, None], MAX_H, axis=1)


def fc_snaive_growth(Yhist: np.ndarray) -> np.ndarray:
    """y[T+h] = y[T] * y[T+h-12] / y[T-12]: last year's same-period change."""
    base = Yhist[-13]                                   # y[T-12]
    later = Yhist[-12:]                                 # y[T-11] .. y[T]
    return (Yhist[-1] * (later / base)).T


def fc_drift(Yhist: np.ndarray) -> np.ndarray:
    g = (np.log(Yhist[-1]) - np.log(Yhist[-13])) / 12   # avg monthly log growth
    h = np.arange(1, MAX_H + 1)
    return Yhist[-1][:, None] * np.exp(g[:, None] * h[None, :])


def fc_ets(Yhist: np.ndarray, dates: pd.DatetimeIndex, window: int) -> np.ndarray:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS
    Y = Yhist[-window:]
    d = dates[-window:]
    n = Y.shape[1]
    long = pd.DataFrame({
        "unique_id": np.tile(np.arange(n), len(d)),
        "ds": np.repeat(d, n),
        "y": Y.reshape(-1),
    }).dropna()
    sf = StatsForecast(models=[AutoETS(season_length=12)], freq="MS", n_jobs=-1)
    out = sf.forecast(df=long, h=MAX_H)
    out = out.reset_index() if "unique_id" not in out.columns else out
    arr = np.full((n, MAX_H), np.nan)
    for uid, grp in out.groupby("unique_id"):
        arr[int(uid)] = grp.sort_values("ds")["AutoETS"].values
    return arr


def fc_xgb_recursive(model, Yhist: np.ndarray, exog_hist: pd.DataFrame) -> np.ndarray:
    """
    Vectorised copy of api.recursive_forecast / build_features_from_history /
    append_prediction, for all series at once:
      - lags and moving averages are read off the end of a growing strip
      - PPI and OPR are frozen at the origin row's values
      - OPR is forward-filled within the 15-row strip only
      - missing PPI/OPR become 0.0 (BigQuery NULL -> `or 0.0` in the API)
    """
    strip = Yhist[-API_HISTORY_ROWS:].T.copy()          # (n_series, 15)
    e = exog_hist.tail(API_HISTORY_ROWS).copy()
    e["opr_rate"] = e["opr_rate"].ffill()
    last = e.iloc[-1]
    ex = np.array([last.get(c, np.nan) for c in
                   ["ppi_index", "ppi_lag_1", "ppi_lag_2", "ppi_lag_3", "opr_rate"]], dtype=float)
    ex = np.nan_to_num(ex, nan=0.0)
    n = strip.shape[0]
    preds = np.zeros((n, MAX_H))
    for step in range(MAX_H):
        X = np.column_stack([
            strip[:, -1], strip[:, -3], strip[:, -6], strip[:, -12],
            strip[:, -3:].mean(axis=1), strip[:, -12:].mean(axis=1),
            np.repeat(ex[None, :], n, axis=0),
        ])
        p = model.predict(X)
        preds[:, step] = p
        strip = np.column_stack([strip, p])
    return preds


def train_xgb(train_df: pd.DataFrame):
    import xgboost as xgb
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(train_df[FEATURES].values, train_df["cpi_index"].values, verbose=False)
    return m


# ═════════════════════════════════════════════════════════════════════════════
# 4. BACKTEST LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(cpi, ppi, opr, start="2018-01-01", retrain_every=12,
                 use_ets=True, ets_window=120, ets_every=3, origin_step=1,
                 origins=None, models=None):
    t0 = time.time()
    exog = build_exog(ppi, opr)
    feats = {m: build_features(cpi, exog, m) for m in ("leaky", "fixed")}

    # wide matrix: rows = every calendar month, cols = series
    dates = pd.date_range(cpi["date"].min(), cpi["date"].max(), freq="MS")
    Y = cpi.pivot_table(index="date", columns=["state", "division"],
                        values="cpi_index").reindex(dates)
    series = Y.columns
    Yv = Y.values
    eval_mask = np.array([d != "overall" for _, d in series])   # API serves 01..13
    exog_cal = exog.set_index("date").reindex(dates)

    if origins is None:
        origins = [d for d in dates if d >= pd.Timestamp(start)][:-1][::origin_step]
    want = set(models or ["naive", "snaive_growth", "drift", "ets",
                          "xgb_as_deployed", "xgb_leak_fixed", "xgb_leaky_onestep"])
    if not use_ets:
        want.discard("ets")
    if "ets" in want:
        try:
            import statsforecast  # noqa: F401
        except ImportError:
            logger.warning("statsforecast not installed -> skipping ets")
            want.discard("ets")

    logger.info(f"{len(series)} series, {len(dates)} months "
                f"({dates[0]:%Y-%m}..{dates[-1]:%Y-%m}), {len(origins)} origins, models={sorted(want)}")

    out, xgb_cache, diag = [], {}, {"origins_with_missing_ppi": 0}
    for i, T in enumerate(origins):
        ti = dates.get_loc(T)
        Yhist = Yv[: ti + 1]
        # API eligibility: >= 13 real rows ending at T (with lag_1 present -> 14 values)
        ok = (~np.isnan(Yhist[-(API_MIN_ROWS + 1):]).any(axis=0)) & eval_mask
        if not ok.any():
            continue
        H = min(MAX_H, len(dates) - 1 - ti)
        actual = Yv[ti + 1: ti + 1 + H].T                  # (n_series, H)
        preds = {}

        if "naive" in want:
            preds["naive"] = fc_naive(Yhist)
        if "snaive_growth" in want:
            preds["snaive_growth"] = fc_snaive_growth(Yhist)
        if "drift" in want:
            preds["drift"] = fc_drift(Yhist)
        if "ets" in want and i % max(ets_every, 1) == 0:   # ETS is slow: subsample origins
            sub = Yhist[:, ok]
            arr = np.full((len(series), MAX_H), np.nan)
            arr[ok] = fc_ets(sub, dates[: ti + 1], ets_window)
            preds["ets"] = arr

        xgb_names = {"xgb_as_deployed": "leaky", "xgb_leak_fixed": "fixed"}
        need_leaky_model = bool(want & {"xgb_as_deployed", "xgb_leaky_onestep"})
        for name, mode in xgb_names.items():
            if name not in want and not (mode == "leaky" and need_leaky_model):
                continue
            # retrain on a fixed calendar (every N months); always trained on data <= T
            key = (mode, (i // retrain_every) if retrain_every > 0 else i)
            if key not in xgb_cache:
                train_df, med = prepare_training(feats[mode], T)
                xgb_cache = {k: v for k, v in xgb_cache.items() if k[0] != mode}
                xgb_cache[key] = (train_xgb(train_df), med)
                logger.info(f"  [{T:%Y-%m}] trained {name} on {len(train_df):,} rows")
            model, med = xgb_cache[key]
            if name in want:
                eh = exog_cal.iloc[: ti + 1].reset_index()
                if np.isnan(eh["ppi_index"].iloc[-1]):
                    diag["origins_with_missing_ppi"] += 1
                preds[name] = fc_xgb_recursive(model, Yhist, eh)
            if mode == "leaky" and "xgb_leaky_onestep" in want:
                # score the T+1 row the way train.py did: features from the leaky mart
                f = feats["leaky"]
                row = f[f["date"] == dates[ti + 1]] if ti + 1 < len(dates) else f.iloc[0:0]
                arr = np.full((len(series), MAX_H), np.nan)
                if len(row):
                    row = row.copy()
                    row["opr_rate"] = row["opr_rate"].fillna(med["opr_rate"])
                    row[FEATURES] = row[FEATURES].fillna(med)
                    p = model.predict(row[FEATURES].values)
                    pos = series.get_indexer(pd.MultiIndex.from_frame(row[["state", "division"]]))
                    arr[pos[pos >= 0], 0] = p[pos >= 0]
                preds["xgb_leaky_onestep"] = arr

        # naive 1-step in-sample MAE per series -> MASE scale
        scale = np.nanmean(np.abs(np.diff(Yhist, axis=0)), axis=0)

        idx = np.where(ok)[0]
        for name, P in preds.items():
            for h in range(1, H + 1):
                a, p = actual[idx, h - 1], P[idx, h - 1]
                keep = ~np.isnan(a) & ~np.isnan(p)
                if not keep.any():
                    continue
                k = idx[keep]
                out.append(pd.DataFrame({
                    "origin": T, "h": h, "model": name,
                    "state": [series[j][0] for j in k],
                    "division": [series[j][1] for j in k],
                    "y_origin": Yhist[-1, k], "y_true": a[keep], "y_pred": p[keep],
                    "scale": scale[k],
                }))
        if (i + 1) % 12 == 0:
            logger.info(f"  {i + 1}/{len(origins)} origins done ({time.time() - t0:.0f}s)")

    res = pd.concat(out, ignore_index=True)
    logger.info(f"Backtest finished: {len(res):,} scored forecasts in {time.time() - t0:.0f}s; diag={diag}")
    return res, diag


# ═════════════════════════════════════════════════════════════════════════════
# 5. METRICS
# ═════════════════════════════════════════════════════════════════════════════

def score(res: pd.DataFrame, horizons=(1, 3, 6, 12)) -> pd.DataFrame:
    r = res[res["h"].isin(horizons)].copy()
    r["err_pp"] = (r["y_pred"] - r["y_true"]) / r["y_origin"] * 100
    r["scaled"] = np.abs(r["y_pred"] - r["y_true"]) / r["scale"].replace(0, np.nan)
    da, dp = np.sign(r["y_true"] - r["y_origin"]), np.sign(r["y_pred"] - r["y_origin"])
    r["dir_hit"] = np.where(da != 0, (da == dp).astype(float), np.nan)

    rows = []
    for period, (lo, hi) in PERIODS.items():
        m = pd.Series(True, index=r.index)
        if lo:
            m &= r["origin"] >= lo
        if hi:
            m &= r["origin"] <= hi
        sub = r[m]
        if sub.empty:
            continue
        g = sub.groupby(["model", "h"])
        t = pd.DataFrame({
            "n": g.size(),
            "mae_pp": g["err_pp"].apply(lambda e: e.abs().mean()),
            "rmse_pp": g["err_pp"].apply(lambda e: np.sqrt((e ** 2).mean())),
            "mase": g["scaled"].mean(),
            "dir_acc": g["dir_hit"].mean(),
        }).reset_index()
        # rel_mae on matched rows: each model vs naive on the SAME origin/series/h,
        # so models scored on a subset of origins (ets) are still compared fairly
        keys = ["origin", "h", "state", "division"]
        nv = sub.loc[sub["model"] == "naive", keys + ["err_pp"]].rename(columns={"err_pp": "naive_err"})
        mm = sub[keys + ["model", "err_pp"]].merge(nv, on=keys)
        mm["a"], mm["b"] = mm["err_pp"].abs(), mm["naive_err"].abs()
        sums = mm.groupby(["model", "h"])[["a", "b"]].sum()
        rel = sums["a"] / sums["b"]
        t["rel_mae"] = t.set_index(["model", "h"]).index.map(rel)
        t.loc[t["model"] == "naive", "dir_acc"] = np.nan   # naive never predicts a direction
        t["period"] = period
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def report(summary: pd.DataFrame, diag: dict, source: str) -> str:
    order = ["naive", "snaive_growth", "drift", "ets", "xgb_leak_fixed",
             "xgb_as_deployed", "xgb_leaky_onestep"]
    lines = [f"# CPI forecast backtest ({source})", ""]
    if source == "synthetic":
        lines += ["> **Synthetic data.** These numbers test the code, not the model. "
                  "Re-run with `--source bigquery` or `--source dosm`.", ""]
    lines += [
        "**rel_mae** < 1.00 means the model beats the naive forecast "
        "(\"next month = this month\"). **mae_pp** = mean error of the cumulative "
        "% change, in percentage points.", "",
        "`xgb_leaky_onestep` is scored the way train.py scored it (leaky features, "
        "one step). It is not a real forecast — it shows how much the leak flatters the reported metrics.", "",
    ]
    for period in PERIODS:
        s = summary[summary["period"] == period]
        if s.empty:
            continue
        n_orig = s["n"].max()
        lines += [f"## {period.replace('_', ' ').title()}", ""]
        for metric, fmt in [("rel_mae", "{:.2f}"), ("mae_pp", "{:.2f}"),
                            ("mase", "{:.2f}"), ("dir_acc", "{:.0%}")]:
            p = s.pivot(index="model", columns="h", values=metric)
            p = p.reindex([m for m in order if m in p.index])
            lines += [f"**{metric}**", "", "| model | " + " | ".join(f"h={h}" for h in p.columns) + " |",
                      "|---|" + "---:|" * len(p.columns)]
            for m, row in p.iterrows():
                cells = ["—" if pd.isna(v) else fmt.format(v) for v in row]
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines.append("")
    if diag.get("origins_with_missing_ppi"):
        lines += [f"> Note: at {diag['origins_with_missing_ppi']} origin(s) the latest month had no PPI yet; "
                  "the production API feeds 0.0 for it, which this backtest reproduces.", ""]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# 6. LEAKAGE SELF-TEST
#    If any forecaster peeks past the origin, poisoning the future will change
#    its forecast. Proves the backtest itself is honest.
# ═════════════════════════════════════════════════════════════════════════════

def self_test() -> None:
    cpi, ppi, opr = make_synthetic()
    T = pd.Timestamp("2023-06-01")
    kw = dict(origins=[T], retrain_every=0, use_ets=True, ets_window=60,
              models=["naive", "snaive_growth", "drift", "ets", "xgb_as_deployed", "xgb_leak_fixed"])

    clean, _ = run_backtest(cpi, ppi, opr, **kw)

    cpi_p, ppi_p, opr_p = cpi.copy(), ppi.copy(), opr.copy()
    cpi_p.loc[cpi_p["date"] > T, "cpi_index"] *= 50
    ppi_p.loc[ppi_p["date"] > T, "ppi_index"] *= 50
    opr_p.loc[opr_p["date"] > T, "opr_rate"] += 40
    poisoned, _ = run_backtest(cpi_p, ppi_p, opr_p, **kw)

    keys = ["model", "h", "state", "division"]
    m = clean.merge(poisoned, on=keys, suffixes=("", "_p"))
    assert len(m) == len(clean) > 0
    diff = (m["y_pred"] - m["y_pred_p"]).abs().groupby(m["model"]).max()
    logger.info("Max forecast change after poisoning the future, per model:\n" + diff.to_string())
    assert (diff < 1e-9).all(), "LEAK: a forecaster used data after the origin"
    assert (m["y_true_p"] != m["y_true"]).all(), "poisoning did not reach the actuals"
    logger.info("SELF-TEST PASSED: no forecaster uses data after the origin.")


# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["bigquery", "dosm", "synthetic"], default="bigquery")
    ap.add_argument("--project", default=os.getenv("GCP_PROJECT_ID", "modular-rush-495414-i2"))
    ap.add_argument("--dataset", default="dbt_")
    ap.add_argument("--start", default="2018-01-01", help="first forecast origin")
    ap.add_argument("--retrain-every", type=int, default=12, help="months between XGBoost retrains")
    ap.add_argument("--origin-step", type=int, default=1, help="use every Nth month as an origin")
    ap.add_argument("--no-ets", action="store_true")
    ap.add_argument("--ets-window", type=int, default=120, help="months of history ETS sees")
    ap.add_argument("--ets-every", type=int, default=3, help="fit ETS at every Nth origin (it is slow)")
    ap.add_argument("--out", default="backtest_results")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return

    if a.source == "bigquery":
        cpi, ppi, opr = load_bigquery(a.project, a.dataset)
    elif a.source == "dosm":
        cpi, ppi, opr = load_dosm()
    else:
        cpi, ppi, opr = make_synthetic()

    res, diag = run_backtest(cpi, ppi, opr, start=a.start, retrain_every=a.retrain_every,
                             use_ets=not a.no_ets, ets_window=a.ets_window,
                             ets_every=a.ets_every, origin_step=a.origin_step)
    summary = score(res)

    os.makedirs(a.out, exist_ok=True)
    res.to_parquet(os.path.join(a.out, "forecasts.parquet"), index=False)
    summary.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    md = report(summary, diag, a.source)
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    logger.info(f"Wrote {a.out}/forecasts.parquet, summary.csv, report.md")


if __name__ == "__main__":
    sys.exit(main())
