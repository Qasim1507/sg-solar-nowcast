"""Recompute the NEA manifest from the parquet files already on disk.

Useful after changing what "complete" means, or to repair a manifest that got out
of step with the files. Purely local - makes no network requests.

    python scripts/rebuild_nea_manifest.py [--endpoint rainfall]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data._client import load_config, log, resolve_path, setup_logging, write_json_atomic  # noqa: E402
from data.fetch_nea import page_offsets, usable_issue_times  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="rainfall")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    raw = resolve_path(cfg, "raw")
    ep_dir = raw / "nea" / args.endpoint
    if not ep_dir.exists():
        log.error("no data at %s", ep_dir)
        return 1

    n_slots = len(page_offsets(cfg)) * cfg["nea"]["page_size"]
    n_issue = (cfg["daylight"]["end_hour"] - max(cfg["horizons"])
               - cfg["daylight"]["start_hour"] + 1)
    manifest, files = {}, sorted(ep_dir.glob("*.parquet"))

    for f in files:
        key = f.stem
        df = pd.read_parquet(f)
        if df.empty:
            manifest[key] = {"frames": 0, "raw_slots_paged": n_slots,
                             "usable_issue_times": 0, "max_issue_times": n_issue,
                             "median_stations": 0, "complete": False}
            continue
        ts = pd.to_datetime(df["timestamp"])
        have = set(ts.unique())
        usable = usable_issue_times(cfg, date.fromisoformat(key), have)
        n_st = int(df.groupby("timestamp")["station_id"].nunique().median())
        manifest[key] = {
            "frames": int(ts.nunique()),
            "raw_slots_paged": n_slots,
            "usable_issue_times": usable,
            "max_issue_times": n_issue,
            "median_stations": n_st,
            "complete": bool(usable >= n_issue and n_st >= cfg["nea"]["min_stations"]),
        }

    out = raw / "nea" / f"manifest_{args.endpoint}.json"
    write_json_atomic(out, manifest)

    # Calendar-level view. Days that 404'd have NO file, so they are absent from
    # the manifest entirely - counting only files would hide them, which is exactly
    # the "silent archive gap" failure this project is meant to guard against.
    keys = sorted(manifest)
    d0, d1 = date.fromisoformat(keys[0]), date.fromisoformat(keys[-1])
    span = (d1 - d0).days + 1
    all_days = {str(d0 + pd.Timedelta(days=i).to_pytimedelta()) for i in range(span)}
    absent = sorted(all_days - set(manifest))

    total = len(manifest)
    complete = sum(1 for v in manifest.values() if v["complete"])
    empty = [k for k, v in manifest.items() if v["frames"] == 0]
    partial = [k for k, v in manifest.items() if 0 < v["usable_issue_times"] < n_issue]
    got = sum(v["usable_issue_times"] for v in manifest.values())
    log.info("%s: %d days on disk", args.endpoint, total)
    log.info("  fully usable %d (%.1f%%) | partial %d | empty %d",
             complete, 100.0 * complete / max(total, 1), len(partial), len(empty))
    log.info("  issue times recovered: %d / %d (%.1f%% of files on disk)",
             got, total * n_issue, 100.0 * got / max(total * n_issue, 1))
    log.info("CALENDAR COVERAGE %s .. %s (%d days)", d0, d1, span)
    log.info("  days with no file at all (404 / never fetched): %d", len(absent))
    log.info("  fully usable days: %d / %d (%.1f%% of calendar)",
             complete, span, 100.0 * complete / max(span, 1))
    log.info("  issue times: %d / %d (%.1f%% of calendar maximum)",
             got, span * n_issue, 100.0 * got / max(span * n_issue, 1))
    if absent:
        log.warning("  ABSENT days (%d): %s%s", len(absent), absent[:12],
                    " ..." if len(absent) > 12 else "")
    if empty:
        log.warning("  empty days (%d): %s%s", len(empty), sorted(empty)[:10],
                    " ..." if len(empty) > 10 else "")
    if partial:
        log.warning("  partial days (%d): %s%s", len(partial), sorted(partial)[:10],
                    " ..." if len(partial) > 10 else "")
    log.info("  wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
