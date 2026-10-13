"""
Monthly batch forecaster: the production replacement for train.py + recursive serving.

Runs after ingest + dbt in the monthly pipeline job. It:
  1. Loads the dbt staging data (or local DOSM files).
  2. Trains the direct growth XGBoost (one model per horizon 1..12) and per-series
     ETS on ALL data up to the latest month, and averages them (ens_ets_direct,
     the best model in docs/backtest_report.md).
  3. Builds 80% forecast ranges from REAL backtest errors: it re-runs the backtest
     over the most recent 36 months and takes the 7.5th/92.5th percentiles of
     log(actual / forecast) per horizon, pooled across categories.
     Why these choices (tested out of sample: ranges built from Aug 2020-Jul 2023
     errors, checked against Aug 2023-2026 outcomes):
       - plain 10th/90th percentiles covered only 75-77% (prices got more
         volatile after the calibration window); 7.5/92.5 covered 81% at
         every horizon.
       - per-category ranges overfit (62-72% coverage; some categories < 50%),
         because category volatility shifts over time.
  4. Writes one small JSON table of every state x category x month forecast to
     GCS. The API just looks rows up; no model runs at request time.

Why precompute: the inputs only change once a month, so computing 208 x 12
forecasts once is cheaper, faster and easier to audit than running models per
request. The API reloads the file when it changes.

Usage:
  python forecast_batch.py                                  # BigQuery -> GCS (pipeline)
  python forecast_batch.py --source files --data-dir data --local-out out.json  # offline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import backtest as bt

logger = logging.getLogger("forecast_batch")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "modular-rush-495414-i2")
GCS_BUCKET = f"{PROJECT_ID}-ml-artifacts"
GCS_LATEST = "forecasts/latest.json"
INTERVAL_LEVEL = 0.80
QUANTILES = (0.075, 0.925)       # gives ~80% coverage out of sample (see docstring)


def forecast_now(cpi, ppi, opr, ets_window=120):
    """Train on everything and forecast 1..12 months from the latest month."""
    exog = bt.build_exog(ppi, opr)
    dates = pd.date_range(cpi["date"].min(), cpi["date"].max(), freq="MS")
    Y = cpi.pivot_table(index="date", columns=["state", "division"], values="cpi_index").reindex(dates)
    series, Yv = Y.columns, Y.values
    exog_cal = exog.set_index("date").reindex(dates)
    T = len(dates) - 1

    ok = ~np.isnan(Yv[-(bt.API_MIN_ROWS + 1):]).any(axis=0)
    ok &= np.array([d != "overall" for _, d in series])
    logger.info(f"Forecasting from {dates[T]:%Y-%m} for {ok.sum()} series")

    f = bt.build_direct_features(Yv, series, dates, exog_cal)
    models = {h: bt.train_direct(f, h, T) for h in range(1, bt.MAX_H + 1)}
    direct = bt.fc_direct(models, f, T, Yv)

    ets = np.full_like(direct, np.nan)
    ets[ok] = bt.fc_ets(Yv[:, ok], dates, ets_window)
    ens = np.sqrt(ets * direct)

    rows = []
    for j in np.where(ok)[0]:
        state, div = series[j]
        for h in range(1, bt.MAX_H + 1):
            rows.append({"state": state, "division": div, "h": h,
                         "base_date": dates[T], "base_index": float(Yv[T, j]),
                         "target_date": dates[T] + pd.DateOffset(months=h),
                         "predicted_index": float(ens[j, h - 1])})
    return pd.DataFrame(rows), models


def calibrate(cpi, ppi, opr, months=36, ets_every=3, ets_window=120):
    """
    Re-run the backtest over the last `months` origins for the production model
    and return pooled 7.5th/92.5th percentiles of log(actual / forecast) by horizon.
    """
    last = cpi["date"].max()
    start = last - pd.DateOffset(months=months)
    origins = list(pd.date_range(start, last - pd.DateOffset(months=1), freq="MS"))
    res, _ = bt.run_backtest(cpi, ppi, opr, origins=origins, retrain_every=12,
                             ets_every=ets_every, ets_window=ets_window,
                             models=["ens_ets_direct", "naive"],
                             direct_horizons=tuple(range(1, bt.MAX_H + 1)))
    ens = res[res["model"] == "ens_ets_direct"].copy()
    ens["log_err"] = np.log(ens["y_true"] / ens["y_pred"])

    qlo, qhi = QUANTILES
    q = ens.groupby("h")["log_err"].agg(
        lo=lambda s: s.quantile(qlo), hi=lambda s: s.quantile(qhi), n="size").reset_index()
    q = q.sort_values("h")
    q["lo"] = q["lo"].cummin()          # ranges must not shrink as the horizon grows
    q["hi"] = q["hi"].cummax()

    # skill on the calibration window, recorded with the forecasts for transparency
    nv = res[res["model"] == "naive"][["origin", "h", "state", "division", "y_true", "y_pred"]]
    m = ens.merge(nv, on=["origin", "h", "state", "division"], suffixes=("", "_naive"))
    skill = (m.assign(a=(m.y_pred - m.y_true).abs(), b=(m.y_pred_naive - m.y_true).abs())
              .groupby("h")[["a", "b"]].sum().pipe(lambda d: d["a"] / d["b"]))
    logger.info("Calibration rel_mae vs naive by horizon: "
                + ", ".join(f"h{h}={v:.2f}" for h, v in skill.items()))
    return q[["h", "lo", "hi", "n"]], skill


def build_table(fc: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    t = fc.merge(q, on="h", how="left")
    t["lower_bound"] = t["predicted_index"] * np.exp(t["lo"])
    t["upper_bound"] = t["predicted_index"] * np.exp(t["hi"])
    t["confidence_band_pct"] = (t["upper_bound"] - t["lower_bound"]) / 2 / t["predicted_index"] * 100
    return t.drop(columns=["lo", "hi", "n"])


def to_payload(table: pd.DataFrame, skill: pd.Series, version: str) -> dict:
    t = table.copy()
    for c in ("base_date", "target_date"):
        t[c] = t[c].dt.strftime("%Y-%m-%d")
    for c in ("base_index", "predicted_index", "lower_bound", "upper_bound"):
        t[c] = t[c].round(3)
    t["confidence_band_pct"] = t["confidence_band_pct"].round(2)
    return {
        "meta": {
            "version": version,
            "model": "ens_ets_direct (geometric mean of AutoETS and direct-growth XGBoost)",
            "base_date": t["base_date"].iloc[0],
            "interval_level": INTERVAL_LEVEL,
            "interval_method": "pooled 7.5th/92.5th percentiles of the last 36 months of backtest "
                               "errors per horizon (81% out-of-sample coverage when tested)",
            "calibration_rel_mae_vs_naive": {int(h): round(float(v), 3) for h, v in skill.items()},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "rows": t.to_dict(orient="records"),
    }


def upload(payload: dict, project_id: str) -> str:
    from google.cloud import storage
    bucket = storage.Client(project=project_id).bucket(f"{project_id}-ml-artifacts")
    body = json.dumps(payload)
    versioned = f"forecasts/forecasts_{payload['meta']['version']}.json"
    bucket.blob(versioned).upload_from_string(body, content_type="application/json")
    bucket.blob(GCS_LATEST).upload_from_string(body, content_type="application/json")
    logger.info(f"Uploaded gs://{project_id}-ml-artifacts/{versioned} and {GCS_LATEST}")
    return versioned


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["bigquery", "files", "dosm"], default="bigquery")
    ap.add_argument("--project", default=PROJECT_ID)
    ap.add_argument("--dataset", default="dbt_")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--calibration-months", type=int, default=36)
    ap.add_argument("--local-out", help="write JSON here instead of uploading to GCS")
    a = ap.parse_args()

    t0 = time.time()
    if a.source == "bigquery":
        cpi, ppi, opr = bt.load_bigquery(a.project, a.dataset)
    elif a.source == "files":
        cpi, ppi, opr = bt.load_files(a.data_dir)
    else:
        cpi, ppi, opr = bt.load_dosm()

    q, skill = calibrate(cpi, ppi, opr, months=a.calibration_months)
    fc, _ = forecast_now(cpi, ppi, opr)
    table = build_table(fc, q)

    bad = table[["predicted_index", "lower_bound", "upper_bound"]].isna().any(axis=1).sum()
    if bad or len(table) == 0:
        raise SystemExit(f"Refusing to publish: {bad} incomplete rows out of {len(table)}")

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = to_payload(table, skill, version)
    if a.local_out:
        with open(a.local_out, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        logger.info(f"Wrote {a.local_out}")
    else:
        upload(payload, a.project)
    logger.info(f"Done: {len(table):,} forecast rows from {payload['meta']['base_date']} "
                f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
