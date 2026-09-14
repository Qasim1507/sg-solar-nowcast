"""data.gov.sg v2 real-time API -> per-day gauge readings (long format) + manifest.

Verified behaviour (2026-09-08), all of which this fetcher depends on:

  * Response shape: data.stations[] (id, name, location.latitude/longitude)
    and data.readings[] (timestamp, data[] of {stationId, value}).
  * Readings are ~5-minutely and **descending** from 23:55 on the requested date.
  * Pagination returns 25 readings/page via data.paginationToken.
  * The token is base64("offset=N") and **crafted offsets work**, so we can seek
    straight to the daylight window instead of walking all ~12 pages/day
    (5 requests/day instead of 12). Falls back to sequential token-walking if a
    crafted offset is ever rejected.
  * Rate limiting arrives as **HTTP 200 with code=24 in the body** - handled in
    ApiClient, not here.
  * Station membership CHANGES over time (51 in Jan-2020, 88 today) and even
    between adjacent frames. Readings are therefore joined to stations by
    stationId per response; no fixed order or count is ever assumed.
  * Partial days exist (2021-01-01, 2023-01-01 both end at 07:55). The manifest
    records actual vs expected frame counts so these surface loudly.

Usage:
    python data/fetch_nea.py --start 2020-01-01 --end 2026-09-05
    python data/fetch_nea.py --endpoints rainfall,wind-speed,wind-direction
"""
from __future__ import annotations

import argparse
import base64
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data._client import (ApiClient, PermanentError, load_config, log, read_json,  # noqa: E402
                          resolve_path, setup_logging, write_json_atomic)

LAST_SLOT_MIN = 23 * 60 + 55   # 23:55 == pagination offset 0
SLOT_MIN = 5


def offset_for_time(hour: int, minute: int = 0) -> int:
    """Pagination offset whose page begins at the given local wall-clock time."""
    return (LAST_SLOT_MIN - (hour * 60 + minute)) // SLOT_MIN


def make_token(offset: int) -> str:
    return base64.b64encode(f"offset={offset}".encode()).decode()


def required_frame_times(cfg: dict, day: date) -> list["pd.Timestamp"]:
    """The 5-minute frames a day must have for its grids to be buildable.

    Grids need t, t-30min and t-60min for every issue time t in the daylight
    window that leaves room for the longest horizon. That is only the :00 and :30
    marks from (start-1h) to (end - max_horizon) - about 15 timestamps, NOT all
    ~100 five-minute slots the pages happen to return.

    Judging a day by raw frame count therefore cries wolf: a day holding 38 of 125
    raw slots can still be 100% usable. Coverage is judged against THIS list.
    """
    lo = cfg["daylight"]["start_hour"]
    hi = cfg["daylight"]["end_hour"] - max(cfg["horizons"])
    offs = sorted(set(cfg["grid"]["frame_offsets_min"]))
    out = set()
    for h in range(lo, hi + 1):
        t = pd.Timestamp(day) + pd.Timedelta(hours=h)
        for off in offs:
            out.add(t - pd.Timedelta(minutes=off))
    return sorted(out)


def usable_issue_times(cfg: dict, day: date, have: set) -> int:
    """How many issue times this day can actually produce grids for."""
    lo = cfg["daylight"]["start_hour"]
    hi = cfg["daylight"]["end_hour"] - max(cfg["horizons"])
    offs = sorted(set(cfg["grid"]["frame_offsets_min"]))
    n = 0
    for h in range(lo, hi + 1):
        t = pd.Timestamp(day) + pd.Timedelta(hours=h)
        if all((t - pd.Timedelta(minutes=o)) in have for o in offs):
            n += 1
    return n


def page_offsets(cfg: dict) -> list[int]:
    """Page-start offsets covering the configured fetch window, newest first."""
    nea = cfg["nea"]
    off_lo = offset_for_time(nea["fetch_end_hour"])      # latest time -> smallest offset
    off_hi = offset_for_time(nea["fetch_start_hour"])    # earliest time -> largest offset
    return list(range(off_lo, off_hi + 1, nea["page_size"]))


def parse_response(payload: dict) -> tuple[pd.DataFrame, dict]:
    """-> (long-format readings df, {station_id: station_meta})."""
    d = payload.get("data") or {}
    stations = {s["id"]: {
        "station_id": s["id"],
        "name": s.get("name"),
        "latitude": (s.get("location") or {}).get("latitude"),
        "longitude": (s.get("location") or {}).get("longitude"),
    } for s in d.get("stations", [])}

    rows = []
    for r in d.get("readings", []):
        ts = r.get("timestamp")
        for item in r.get("data", []):
            rows.append((ts, item.get("stationId"), item.get("value")))
    df = pd.DataFrame(rows, columns=["timestamp", "station_id", "value"])
    return df, stations


def fetch_day(client: ApiClient, cfg: dict, endpoint: str, day: date) -> tuple[pd.DataFrame, dict]:
    url = f"{cfg['nea']['base_url']}/{endpoint}"
    frames, stations = [], {}
    use_seek = cfg["nea"].get("use_offset_seek", True)

    if use_seek:
        for off in page_offsets(cfg):
            params = {"date": day.isoformat()}
            if off > 0:
                params["paginationToken"] = make_token(off)
            try:
                payload = client.get_json(url, params=params)
            except PermanentError as e:
                # no data for this page/day - record it, do not burn retries
                log.info("%s %s offset=%d: %s", endpoint, day, off, str(e)[:80])
                continue
            except RuntimeError as e:
                log.warning("%s %s offset=%d failed (%s); falling back to walk",
                            endpoint, day, off, str(e)[:80])
                use_seek = False
                frames, stations = [], {}
                break
            df, st = parse_response(payload)
            stations.update(st)
            if len(df):
                frames.append(df)

    if not use_seek:
        token, guard = None, 0
        while guard < 40:
            params = {"date": day.isoformat()}
            if token:
                params["paginationToken"] = token
            payload = client.get_json(url, params=params)
            df, st = parse_response(payload)
            stations.update(st)
            if len(df):
                frames.append(df)
            token = (payload.get("data") or {}).get("paginationToken")
            guard += 1
            if not token:
                break

    if not frames:
        return pd.DataFrame(columns=["timestamp", "station_id", "value"]), stations
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["timestamp", "station_id"])
    return out, stations


def clip_to_window(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    local = ts.dt.tz_convert(cfg["site"]["timezone"])
    lo, hi = cfg["nea"]["fetch_start_hour"], cfg["nea"]["fetch_end_hour"]
    keep = (local.dt.hour >= lo) & (local.dt.hour <= hi)
    out = df.loc[keep].copy()
    out["timestamp"] = local[keep].dt.tz_localize(None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--endpoints", default="rainfall",
                    help="comma-separated; default rainfall only (the spatial field)")
    ap.add_argument("--delay", type=float, default=None, help="override request delay (s)")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent days; measured safe at 4, degrades at 8")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    raw = resolve_path(cfg, "raw")
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]

    start = date.fromisoformat(args.start or cfg["period"]["start"])
    end = (date.fromisoformat(args.end) if args.end
           else date.today() - timedelta(days=cfg["period"]["archive_lag_days"]))

    client = ApiClient(
        delay_s=args.delay if args.delay is not None else cfg["nea"]["request_delay_s"],
        backoff_s=cfg["nea"]["rate_limit_backoff_s"],
        json_code_field="code",
    )

    n_slots = len(page_offsets(cfg)) * cfg["nea"]["page_size"]
    n_issue = (cfg["daylight"]["end_hour"] - max(cfg["horizons"])
               - cfg["daylight"]["start_hour"] + 1)
    log.info("NEA %s | %s -> %s | %d pages/day (offset seek=%s)",
             endpoints, start, end, len(page_offsets(cfg)), cfg["nea"]["use_offset_seek"])

    for endpoint in endpoints:
        ep_dir = raw / "nea" / endpoint
        ep_dir.mkdir(parents=True, exist_ok=True)
        man_path = raw / "nea" / f"manifest_{endpoint}.json"
        manifest = read_json(man_path) or {}
        station_meta = read_json(raw / "nea" / f"stations_{endpoint}.json") or {}

        todo, n_skip = [], 0
        day = start
        while day <= end:
            key = day.isoformat()
            if not args.force and (ep_dir / f"{key}.parquet").exists() and key in manifest:
                n_skip += 1
            else:
                todo.append(day)
            day += timedelta(days=1)
        log.info("  %s: %d days to fetch, %d already cached", endpoint, len(todo), n_skip)

        lock = threading.Lock()
        progress = {"n": 0}

        def do_day(d: date):
            """Fetch one day. Each thread gets its own client (Session is not shared)."""
            cl = ApiClient(
                delay_s=args.delay if args.delay is not None else cfg["nea"]["request_delay_s"],
                backoff_s=cfg["nea"]["rate_limit_backoff_s"],
                json_code_field="code",
            )
            key = d.isoformat()
            try:
                df, stations = fetch_day(cl, cfg, endpoint, d)
            except PermanentError as e:
                log.info("  %s %s: no data (%s)", endpoint, key, str(e)[:70])
                with lock:
                    manifest[key] = {"frames": 0, "expected_frames": int(n_slots),
                                     "median_stations": 0, "complete": False,
                                     "permanent_error": str(e)[:120]}
                return
            except Exception as e:                       # keep the run alive
                log.warning("  %s %s FAILED: %s", endpoint, key, str(e)[:110])
                return
            df = clip_to_window(df, cfg)
            if len(df):
                df.to_parquet(ep_dir / f"{key}.parquet", index=False)
                frames_got = int(df["timestamp"].nunique())
                n_st = int(df.groupby("timestamp")["station_id"].nunique().median())
                have = set(pd.to_datetime(df["timestamp"]).unique())
                usable = usable_issue_times(cfg, d, have)
            else:
                frames_got, n_st, usable = 0, 0, 0
            with lock:
                station_meta.update(stations)
                manifest[key] = {
                    "frames": frames_got,
                    "raw_slots_paged": int(n_slots),
                    "usable_issue_times": usable,
                    "max_issue_times": int(n_issue),
                    "median_stations": n_st,
                    # "complete" means: yields every issue time we can actually use.
                    # Judged against required frames, not raw page slots.
                    "complete": bool(usable >= n_issue
                                     and n_st >= cfg["nea"]["min_stations"]),
                }
                progress["n"] += 1
                if progress["n"] % 50 == 0:
                    write_json_atomic(man_path, manifest)
                    write_json_atomic(raw / "nea" / f"stations_{endpoint}.json", station_meta)
                    log.info("  %s  %d/%d done (latest %s frames=%d stations=%d)",
                             endpoint, progress["n"], len(todo), key, frames_got, n_st)

        if todo:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                list(ex.map(do_day, todo))

        write_json_atomic(man_path, manifest)
        write_json_atomic(raw / "nea" / f"stations_{endpoint}.json", station_meta)

        # ---- coverage report: partial/missing days must be loud, never silent
        total = len(manifest)
        complete = sum(1 for v in manifest.values() if v.get("complete"))
        empty = [k for k, v in manifest.items() if v.get("frames", 0) == 0]
        partial = [k for k, v in manifest.items()
                   if 0 < v.get("usable_issue_times", 0) < n_issue]
        got = sum(v.get("usable_issue_times", 0) for v in manifest.values())
        log.info("%s: %d days | fully usable %d (%.1f%%) | partial %d | empty %d | "
                 "stations known %d", endpoint, total, complete,
                 100.0 * complete / max(total, 1), len(partial), len(empty),
                 len(station_meta))
        log.info("  issue times recovered: %d / %d (%.1f%% of theoretical max)",
                 got, total * n_issue, 100.0 * got / max(total * n_issue, 1))
        if partial[:5]:
            log.warning("  partial days (first 5): %s", partial[:5])
        if empty[:5]:
            log.warning("  EMPTY days (first 5): %s", empty[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
