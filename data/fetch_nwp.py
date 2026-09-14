"""Open-Meteo Historical-Forecast archive -> NWP forecast fields at the target hours.

WHY THIS FILE IS CAREFUL ABOUT `models=`
----------------------------------------
Verified 2026-09-08: the Historical Forecast API's DEFAULT (`best_match`) returns
values **bit-identical** to the ERA5 archive target, at the same grid cell
(1.3708 N, 103.802 E), on 2024-01-15, 2024-06-15, 2025-03-10 and 2026-08-20.

Feeding that in as "the NWP forecast of the target hour" hands the model its own
answer -> a spectacular, entirely fake skill score. So:

  * every request pins an explicit `models=` (default gfs_seamless),
  * `best_match` is refused outright,
  * `check_leakage()` re-runs the comparison and is asserted in
    tests/test_nwp_leakage.py.

Archive depth measured at this site (2026-09-08):
    gfs_seamless   ~2021-04-01 onward   <- primary, matches Track A
    icon_seamless  ~2023-01-01 onward
    ecmwf_ifs025   ~mid-2024 onward     (most accurate, shortest archive)

KNOWN LIMITATION (stated in README, not silently papered over): this endpoint
serves the archived forecast series for a day, not specifically the run issued at
time t targeting t+1/2/3h. Lead time is therefore approximate rather than a true
issue-time reconstruction. We do NOT substitute reanalysis to paper over it.

Usage:
    python data/fetch_nwp.py [--start 2021-04-01] [--models gfs_seamless,icon_seamless]
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

BEST_MATCH = "best_match"


def _require_explicit_model(model: str, cfg: dict) -> None:
    if cfg["nwp"].get("forbid_best_match", True) and (
        not model or model.strip().lower() == BEST_MATCH
    ):
        raise ValueError(
            "Refusing best_match: verified to return values bit-identical to the "
            "ERA5 target (target leakage). Pin an explicit NWP model."
        )


def fetch_model_range(client: ApiClient, cfg: dict, model: str,
                      start: str, end: str, live: bool = False) -> pd.DataFrame:
    """Fetch one NWP model over a date range. Columns prefixed `nwp_{model}_`."""
    _require_explicit_model(model, cfg)
    url = cfg["nwp"]["live_url"] if live else cfg["nwp"]["archive_url"]
    p = {
        "latitude": cfg["site"]["latitude"],
        "longitude": cfg["site"]["longitude"],
        "hourly": ",".join(cfg["nwp"]["vars"]),
        "timezone": cfg["site"]["timezone"],
        "models": model,
    }
    if live:
        p["forecast_days"] = 2
    else:
        p["start_date"], p["end_date"] = start, end

    h = client.get_json(url, params=p)["hourly"]
    df = pd.DataFrame(h).rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    short = model.replace("_seamless", "").replace("_single", "")
    df = df.rename(columns={v: f"nwp_{short}_{v}" for v in cfg["nwp"]["vars"]})
    return df


def check_leakage(client: ApiClient, cfg: dict, model: str,
                  probe_days: list[str] | None = None) -> dict:
    """Assert this NWP model is NOT a copy of the ERA5 target.

    Returns a diagnostics dict; raises if any probe day comes back identical.
    """
    probe_days = probe_days or ["2024-06-15", "2025-03-10", "2026-08-20"]
    tgt_col = "shortwave_radiation"
    short = model.replace("_seamless", "").replace("_single", "")
    results = []
    for day in probe_days:
        era = client.get_json(cfg["weather"]["archive_url"], params={
            "latitude": cfg["site"]["latitude"], "longitude": cfg["site"]["longitude"],
            "start_date": day, "end_date": day, "hourly": tgt_col,
            "timezone": cfg["site"]["timezone"]})["hourly"][tgt_col]
        nwp_df = fetch_model_range(client, cfg, model, day, day)
        nwp = nwp_df[f"nwp_{short}_{tgt_col}"].tolist()
        a = np.array([x if x is not None else np.nan for x in era], dtype=float)
        b = np.array([x if x is not None else np.nan for x in nwp], dtype=float)
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() == 0:
            results.append({"day": day, "status": "no overlap"})
            continue
        identical = bool(np.array_equal(a[m], b[m]))
        mae = float(np.mean(np.abs(a[m] - b[m])))
        results.append({"day": day, "identical": identical, "mae_wm2": round(mae, 2),
                        "n": int(m.sum())})
        if identical:
            raise AssertionError(
                f"TARGET LEAKAGE: model={model} on {day} is bit-identical to the "
                f"ERA5 target over {int(m.sum())} hours. Do not train on this.")
    return {"model": model, "probes": results}


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
    ap.add_argument("--models", default=None, help="comma-separated; overrides config")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-leakage-check", action="store_true")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    raw = resolve_path(cfg, "raw")
    chunk_dir = raw / "nwp_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    models = ([m.strip() for m in args.models.split(",")] if args.models
              else list(cfg["nwp"]["models"]))
    for m in models:
        _require_explicit_model(m, cfg)

    start = date.fromisoformat(args.start or cfg["dataset"]["tracks"]["A"]["start"])
    end = (date.fromisoformat(args.end) if args.end
           else date.today() - timedelta(days=cfg["period"]["archive_lag_days"]))

    client = ApiClient(delay_s=1.0, backoff_s=8.0)
    state = ResumeState(raw / "nwp_resume.json")
    if args.force:
        state.done.clear()

    # ---- leakage gate: refuse to build a training input that copies the target
    leak_report = []
    if not args.skip_leakage_check:
        for m in models:
            rep = check_leakage(client, cfg, m)
            leak_report.append(rep)
            log.info("leakage check %s: %s", m,
                     ", ".join(f"{p['day']} mae={p.get('mae_wm2')}" for p in rep["probes"]))

    per_model = []
    for m in models:
        short = m.replace("_seamless", "").replace("_single", "")
        for c0, c1 in year_chunks(start, end):
            key = f"{m}_{c0}_{c1}"
            path = chunk_dir / f"{key}.csv"
            if state.has(key) and path.exists() and not args.force:
                continue
            df = fetch_model_range(client, cfg, m, c0.isoformat(), c1.isoformat())
            df.to_csv(path, index=False)
            state.add(key, flush=True)
            log.info("  %s %s..%s  %d rows", m, c0, c1, len(df))

        frames = [pd.read_csv(p, parse_dates=["timestamp"])
                  for p in sorted(chunk_dir.glob(f"{m}_*.csv"))]
        if frames:
            d = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
            per_model.append(d.sort_values("timestamp").reset_index(drop=True))
            log.info("%s: %d rows, non-null swrad %d", m, len(d),
                     int(d[f"nwp_{short}_shortwave_radiation"].notna().sum()))

    if not per_model:
        log.error("no NWP data fetched")
        return 1

    out_df = per_model[0]
    for d in per_model[1:]:
        out_df = out_df.merge(d, on="timestamp", how="outer")
    out_df = out_df.sort_values("timestamp").reset_index(drop=True)
    out = raw / "nwp_hourly.csv"
    out_df.to_csv(out, index=False)

    swrad_cols = [c for c in out_df.columns if c.endswith("shortwave_radiation")]
    first_valid = out_df.loc[out_df[swrad_cols].notna().any(axis=1), "timestamp"].min()
    manifest = {
        "source": "open-meteo historical-forecast",
        "url": cfg["nwp"]["archive_url"],
        "models": models,
        "best_match_forbidden": True,
        "leakage_check": leak_report,
        "start": str(start), "end": str(end),
        "rows": int(len(out_df)),
        "first_valid_forecast": str(first_valid),
        "limitation": ("archived forecast series per day, not the run issued exactly "
                       "at t; lead time approximate"),
    }
    pd.Series(manifest).to_json(raw / "nwp_manifest.json", indent=2)
    log.info("wrote %s  rows=%d  first_valid=%s", out, len(out_df), first_valid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
