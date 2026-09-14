"""Populate the forecast store by replaying historical issue times.

Live issuance only produces ~36 forecasts a day (15-min cadence across a 9-hour
daylight window), and the ERA5 lane does not return a single score for ~3 days, so
a freshly deployed dashboard would show an empty verification panel for most of a
week. This replays stored issue times through the served model to fill it.

Every row written carries mode='replay'. The rolling scorer keeps that separate
from mode='live', so backfilled numbers can never be presented as a live track
record - they are a backtest, and the UI says so.

Outcomes need no network: dataset_{track}.parquet already holds ghi_t{h}h, which
IS the ERA5 target. They are written to the 'era5' lane directly.

Usage:
    python scripts/backfill_forecast_store.py --track A --days 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import storage  # noqa: E402
from data._client import load_config, log, resolve_path, setup_logging  # noqa: E402
from predict import CatBoostForecaster  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="A")
    ap.add_argument("--days", type=int, default=30,
                    help="most recent N days of the dataset to replay")
    ap.add_argument("--split", default=None,
                    help="restrict to one split (train/val/test); default: any")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config()
    proc = resolve_path(cfg, "processed")

    side = resolve_path(cfg, "checkpoints") / f"catboost_{args.track}_kt.json"
    if not side.exists():
        log.error("%s missing - run `python baselines.py --track %s --save` first",
                  side, args.track)
        return 1
    fc = CatBoostForecaster(side, cfg)

    df = pd.read_parquet(proc / f"dataset_{args.track}.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if args.split:
        df = df[df["split"] == args.split].reset_index(drop=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=args.days)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    if df.empty:
        log.error("no rows in the requested window")
        return 1
    log.info("replaying %d issue times, %s .. %s", len(df),
             df["timestamp"].min(), df["timestamp"].max())

    conn = storage.connect(storage.default_path(cfg))
    n_f = n_o = 0
    for _, r in df.iterrows():
        row = r.to_dict()
        t = row["timestamp"]
        future_cs = np.array([row[f"cs_t{h}h"] for h in fc.horizons], dtype=np.float32)
        pred = fc._predict(row, future_cs)

        for k, h in enumerate(fc.horizons):
            valid = t + pd.Timedelta(hours=h)
            storage.record_forecast(
                conn, issue_time=str(t), valid_time=str(valid), horizon_h=h,
                model_id=fc.model_id, values=pred[k].tolist(),
                median=float(pred[k][fc.quantiles.index(0.5)]),
                clearsky=float(future_cs[k]),
                kt_now=float(row["clearsky_ratio"]), mode="replay")
            n_f += 1
            # ghi_t{h}h IS the ERA5 target - no fetch required
            actual = row[f"ghi_t{h}h"]
            if pd.notna(actual):
                storage.record_outcome(conn, str(valid), "era5", float(actual))
                n_o += 1

    s = storage.summary(conn)
    log.info("wrote %d forecast rows and %d era5 outcomes", n_f, n_o)
    log.info("store now: %d forecasts (%d live, %d replay), outcomes=%s",
             s["forecasts"], s["live"], s["replay"], s["outcomes"])
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
