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
  5. Scores every forecast published in earlier months against the actual CPI
     that has since been released, and publishes forecasts/scorecard.json:
     a live, out-of-sample track record (error vs "no change", range coverage)
     that grows every month. Unlike the backtest, nothing here was ever seen by
     the model or by us when the forecast was made.

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
GCS_SCORECARD = "forecasts/scorecard.json"
SCORE_HORIZONS = (1, 3, 6, 12)
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
        # realised change over the past 12 months: the same basis as the official
        # year-on-year headline rate, so the app can compare like with like
        past_12m = (Yv[T, j] / Yv[T - 12, j] - 1) * 100 if T >= 12 else np.nan
        for h in range(1, bt.MAX_H + 1):
            rows.append({"state": state, "division": div, "h": h,
                         "base_date": dates[T], "base_index": float(Yv[T, j]),
                         "past_12m_change_pct": float(past_12m),
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
    t["past_12m_change_pct"] = t["past_12m_change_pct"].round(2)
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


def score_published(published: list, cpi: pd.DataFrame) -> dict:
    """
    Compare previously published forecast payloads with actual CPI.

    For each published base month we keep the newest version (re-runs of the
    same month replace earlier ones), then match every row whose target month
    now has real data. Naive = "no change from base_index", the bar to beat.
    """
    latest_by_base = {}
    for p in published:
        b = p["meta"]["base_date"]
        if b not in latest_by_base or p["meta"]["version"] > latest_by_base[b]["meta"]["version"]:
            latest_by_base[b] = p

    actual = cpi.assign(date=cpi["date"].dt.strftime("%Y-%m-%d")) \
                .set_index(["state", "division", "date"])["cpi_index"]
    recs = []
    for b, p in sorted(latest_by_base.items()):
        for r in p["rows"]:
            key = (r["state"], r["division"], r["target_date"])
            if key not in actual.index:
                continue                    # target month not released yet
            y, base = float(actual[key]), r["base_index"]
            recs.append({
                "base_date": b, "h": r["h"],
                "err_pp": (r["predicted_index"] - y) / base * 100,
                "naive_err_pp": (base - y) / base * 100,
                "inside": r["lower_bound"] <= y <= r["upper_bound"],
            })

    latest_actual = cpi["date"].max().strftime("%Y-%m-%d")
    card = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latest_actual_date": latest_actual,
        "published_months": sorted(latest_by_base),
        "scored_forecasts": len(recs),
        "by_horizon": {},
        "note": "Out-of-sample: each forecast was published before its target month's CPI existed. "
                "rel_mae < 1 means more accurate than assuming no change; coverage_80 should be near 0.80.",
    }
    if not recs:
        return card
    d = pd.DataFrame(recs)
    for h in SCORE_HORIZONS:
        x = d[d["h"] == h]
        if x.empty:
            continue
        mae, nmae = x["err_pp"].abs().mean(), x["naive_err_pp"].abs().mean()
        card["by_horizon"][str(h)] = {
            "n": int(len(x)),
            "base_months": int(x["base_date"].nunique()),
            "mae_pp": round(float(mae), 3),
            "naive_mae_pp": round(float(nmae), 3),
            "rel_mae": round(float(mae / nmae), 3) if nmae > 0 else None,
            "coverage_80": round(float(x["inside"].mean()), 3),
        }
    return card


def load_published_gcs(project_id: str) -> list:
    from google.cloud import storage
    bucket = storage.Client(project=project_id).bucket(f"{project_id}-ml-artifacts")
    out = []
    for blob in bucket.list_blobs(prefix="forecasts/forecasts_"):
        out.append(json.loads(blob.download_as_text()))
    return out


def load_published_dir(path: str) -> list:
    import glob
    return [json.load(open(f, encoding="utf-8"))
            for f in sorted(glob.glob(os.path.join(path, "forecasts_*.json")))]


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
    ap.add_argument("--history-dir", help="offline: folder of earlier forecasts_*.json to score "
                                          "(default: the versioned files in GCS)")
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

    # live track record of everything published so far (never blocks publishing)
    try:
        published = (load_published_dir(a.history_dir) if a.history_dir
                     else [] if a.local_out else load_published_gcs(a.project))
        card = score_published(published, cpi)
        for h, m in card["by_horizon"].items():
            logger.info(f"Scorecard h={h}: rel_mae={m['rel_mae']} coverage={m['coverage_80']} "
                        f"(n={m['n']}, {m['base_months']} published months)")
        if a.local_out:
            out = os.path.splitext(a.local_out)[0] + "_scorecard.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(card, f, indent=1)
            logger.info(f"Wrote {out}")
        else:
            from google.cloud import storage
            storage.Client(project=a.project).bucket(f"{a.project}-ml-artifacts") \
                .blob(GCS_SCORECARD).upload_from_string(json.dumps(card), content_type="application/json")
            logger.info(f"Uploaded gs://{a.project}-ml-artifacts/{GCS_SCORECARD}")
    except Exception as e:
        logger.warning(f"Scorecard step failed (forecasts were still published): {e}")
    logger.info(f"Done: {len(table):,} forecast rows from {payload['meta']['base_date']} "
                f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
