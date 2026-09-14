"""Open-Meteo ERA5 archive -> hourly CSV, plus pvlib Ineichen clear-sky GHI.

`shortwave_radiation` is the TARGET (GHI, W/m^2). Everything else feeds the
temporal branch. Clear-sky GHI is pure astronomy + climatological turbidity, so
it is always known ahead of time and is safe as a head input at t+1/2/3h.

Chunked by year so a long backfill is resumable; each chunk is written
atomically and skipped on re-run.

Usage:
    python data/fetch_weather.py [--start 2020-01-01] [--end 2026-09-05] [--force]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data._client import (ApiClient, ResumeState, load_config, log, resolve_path,  # noqa: E402
                          setup_logging)


def add_clearsky(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Append `ghi_clearsky` and `clearsky_ratio` (k_t) using pvlib Ineichen.

    Open-Meteo's ERA5 `shortwave_radiation` is the MEAN OVER THE PRECEDING HOUR:
    the row labelled 17:00 covers 16:00-17:00. Clear-sky must therefore be an
    hour-mean over the same window, not a point estimate.

    A point estimate at the hour centre is a good approximation mid-day but breaks
    down near sunrise, where irradiance is strongly convex: at the 08:00 row the
    centre (07:30) sits close to sunrise and badly understates the hour's mean,
    which drove k_t above 1.31 on 78% of 08:00 rows (most pinned at the 1.5 clip).
    Integrating over the hour removes that artefact.
    """
    from data.clearsky import hour_mean_ghi, kt_from_ghi

    df = df.copy()
    df["ghi_clearsky"] = hour_mean_ghi(cfg, df["timestamp"])
    df["clearsky_ratio"] = kt_from_ghi(cfg, df["ghi"], df["ghi_clearsky"])
    return df


def fetch_range(client: ApiClient, cfg: dict, start: str, end: str) -> pd.DataFrame:
    p = {
        "latitude": cfg["site"]["latitude"],
        "longitude": cfg["site"]["longitude"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(cfg["temporal"]["hourly_vars"]),
        "timezone": cfg["site"]["timezone"],
    }
    payload = client.get_json(cfg["weather"]["archive_url"], params=p)
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df = df.rename(columns={"time": "timestamp", "shortwave_radiation": "ghi"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def year_chunks(start: date, end: date):
    cur = start
    while cur <= end:
        stop = min(date(cur.year, 12, 31), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--force", action="store_true", help="ignore resume state")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    raw = resolve_path(cfg, "raw")
    chunk_dir = raw / "era5_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start or cfg["period"]["start"])
    if args.end or cfg["period"]["end"]:
        end = date.fromisoformat(args.end or cfg["period"]["end"])
    else:
        end = date.today() - timedelta(days=cfg["period"]["archive_lag_days"])

    state = ResumeState(raw / "era5_resume.json")
    if args.force:
        state.done.clear()

    client = ApiClient(delay_s=1.0, backoff_s=8.0)
    log.info("ERA5 archive %s -> %s", start, end)

    for c0, c1 in year_chunks(start, end):
        key = f"{c0}_{c1}"
        path = chunk_dir / f"{key}.csv"
        if state.has(key) and path.exists() and not args.force:
            log.info("  %s  (cached)", key)
            continue
        df = fetch_range(client, cfg, c0.isoformat(), c1.isoformat())
        df.to_csv(path, index=False)
        state.add(key, flush=True)
        log.info("  %s  %d rows", key, len(df))

    # ---- assemble
    frames = [pd.read_csv(p, parse_dates=["timestamp"]) for p in sorted(chunk_dir.glob("*.csv"))]
    if not frames:
        log.error("no chunks fetched")
        return 1
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)]
    df = add_clearsky(df, cfg)

    out = raw / "weather_hourly.csv"
    df.to_csv(out, index=False)

    # ---- coverage assertion: a silent gap here is the most damaging failure mode
    expected = int((pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timestamp(start))
                   / pd.Timedelta(hours=1))
    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    missing = full.difference(pd.DatetimeIndex(df["timestamp"]))
    ghi_nan = int(df["ghi"].isna().sum())
    cover = 100.0 * len(df) / expected

    manifest = {
        "source": "open-meteo ERA5 archive",
        "url": cfg["weather"]["archive_url"],
        "start": str(start), "end": str(end),
        "rows": int(len(df)), "expected_rows": expected,
        "coverage_pct": round(cover, 3),
        "missing_hours": int(len(missing)),
        "ghi_nan": ghi_nan,
        "clearsky_model": f"pvlib {cfg.get('clearsky',{}).get('model','ineichen')} hour-mean over preceding hour",
        "vars": cfg["temporal"]["hourly_vars"],
    }
    pd.Series(manifest).to_json(raw / "weather_manifest.json", indent=2)

    log.info("wrote %s", out)
    log.info("rows=%d expected=%d coverage=%.2f%% missing_hours=%d ghi_nan=%d",
             len(df), expected, cover, len(missing), ghi_nan)

    if cover < 98.0:
        log.error("COVERAGE ASSERTION FAILED: %.2f%% < 98%%", cover)
        return 1
    day = df[df["timestamp"].dt.hour.between(cfg["daylight"]["start_hour"],
                                             cfg["daylight"]["end_hour"])]
    log.info("daylight rows=%d  ghi mean=%.1f max=%.1f  k_t median=%.3f",
             len(day), day["ghi"].mean(), day["ghi"].max(), day["clearsky_ratio"].median())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
