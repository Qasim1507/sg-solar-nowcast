"""FastAPI backend + static dashboard.

Single inference path: this module imports predict.Forecaster and never
reimplements preprocessing. Anything the dashboard shows, the CLI can reproduce.

Design rule for every endpoint: a response ALWAYS carries `as_of` and
`data_age_minutes`. A stale forecast presented as current is the failure mode
this is designed against, so degraded states are explicit fields, not omissions.

Run:
    uvicorn api.main:app --reload
    REPLAY_DATE=2020-04-15 uvicorn api.main:app     # replay a stored day
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data._client import load_config, log, setup_logging  # noqa: E402
from data.geometry import GridSpec  # noqa: E402

setup_logging()
CFG = load_config()
STATIC = Path(__file__).resolve().parent / "static"
CACHE = ROOT / CFG["paths"]["reports"] / "forecast_latest.json"
REPLAY_DATE = os.environ.get("REPLAY_DATE")
REPLAY_MODE = os.environ.get("REPLAY_MODE", "0") == "1" or bool(REPLAY_DATE)

app = FastAPI(title="Singapore GHI Nowcast", version="0.1.0")

_state: dict = {"forecaster": None, "error": None, "last_refresh": None, "store": None}
_lock = threading.Lock()


# --------------------------------------------------------------------- helpers
def sgt_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.utcnow() + timedelta(hours=CFG["site"]["tz_offset_hours"]))


def served_model() -> tuple[Path | None, str]:
    """The model the dashboard serves, as (path, kind).

    CatBoost + k_t is preferred: it beats every deep variant at every horizon on
    Track A with seed sd <= 0.08 (see REPORT.md). The torch checkpoint remains a
    fallback so the app still runs before the tree model has been saved.
    """
    d = ROOT / CFG["paths"]["checkpoints"]
    if not d.exists():
        return None, "none"
    for track in ("A", "B"):
        side = d / f"catboost_{track}_kt.json"
        if side.exists():
            return side, "catboost"
    # Fallback. NOTE: pick the NEWEST, not the lexicographically first - the k_t
    # ablation added A_gated_kt_seed0.pt, which sorts before A_gated_seed0.pt and
    # silently changed which model was served.
    for pattern in ("A_gated_*.pt", "A_*.pt", "B_gated_*.pt", "*.pt"):
        found = sorted(d.glob(pattern), key=lambda q: q.stat().st_mtime, reverse=True)
        if found:
            return found[0], "torch"
    return None, "none"


def latest_checkpoint() -> Path | None:
    return served_model()[0]


def get_forecaster():
    with _lock:
        if _state["forecaster"] is not None:
            return _state["forecaster"]
        ck = latest_checkpoint()
        if ck is None:
            _state["error"] = "no model found - run `python baselines.py --track A --save`"
            return None
        kind = "catboost" if ck.suffix == ".json" else "torch"
        try:
            if kind == "catboost":
                from predict import CatBoostForecaster
                _state["forecaster"] = CatBoostForecaster(ck, CFG)
            else:
                from predict import Forecaster
                _state["forecaster"] = Forecaster(ck, CFG)
            _state["error"] = None
            log.info("serving %s (%s)", ck.name, kind)
        except Exception as e:                      # degraded, not fatal
            _state["error"] = f"failed to load {ck.name}: {e}"
            log.error(_state["error"])
        return _state["forecaster"]


def model_id() -> str:
    f = _state.get("forecaster")
    return getattr(f, "model_id", None) or (latest_checkpoint().stem
                                            if latest_checkpoint() else "unknown")


def get_store():
    """Lazily open the forecast store. Never fatal - the app runs without it."""
    with _lock:
        if _state.get("store") is not None:
            return _state["store"]
        try:
            import storage
            _state["store"] = storage.connect(storage.default_path(CFG, ROOT))
        except Exception as e:
            log.warning("forecast store unavailable: %s", e)
            _state["store"] = None
        return _state["store"]


def _age_min(ts: str | pd.Timestamp | None) -> float | None:
    if ts is None:
        return None
    try:
        return max(0.0, round((sgt_now() - pd.Timestamp(ts)).total_seconds() / 60.0, 1))
    except Exception:
        return None


def compute_forecast(force_replay: bool | None = None) -> dict:
    f = get_forecaster()
    if f is None:
        raise RuntimeError(_state["error"] or "model unavailable")
    replay = REPLAY_MODE if force_replay is None else force_replay
    if replay:
        issue = None
        if REPLAY_DATE:
            proc = ROOT / CFG["paths"]["processed"]
            df = pd.read_parquet(proc / f"dataset_{f.track}.parquet")
            same = df[df["timestamp"].dt.date == pd.Timestamp(REPLAY_DATE).date()]
            if len(same):
                issue = same["timestamp"].iloc[len(same) // 2]
        out = f.replay(issue)
    else:
        out = f.live()
    out["data_age_minutes"] = out.get("data_age_minutes") or {}
    out["data_age_minutes"]["forecast"] = _age_min(out["issue_time"])
    out["generated_at"] = str(sgt_now())
    return out


def refresh_cache() -> dict | None:
    try:
        out = compute_forecast()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(out, indent=2, default=float))
        _state["last_refresh"] = str(sgt_now())
        log.info("forecast refreshed (%s)", out["issue_time"])
        return out
    except Exception as e:
        _state["error"] = str(e)
        log.warning("refresh failed: %s", str(e)[:160])
        return None


def cached_forecast() -> dict | None:
    """Load the cached forecast, discarding one produced by a different model.

    Without this check a cache outlives the checkpoint that made it: after
    retraining (or deleting a checkpoint) the API would keep serving predictions
    attributed to a model that no longer exists. It would be correctly flagged
    stale, but it would still be the wrong model's numbers - so drop it instead.
    """
    if not CACHE.exists():
        return None
    try:
        cached = json.loads(CACHE.read_text())
    except json.JSONDecodeError:
        return None
    ck = latest_checkpoint()
    cached_ck = (cached.get("model") or {}).get("checkpoint")
    if ck is not None and cached_ck and cached_ck != ck.name:
        log.info("discarding cache from %s (current checkpoint is %s)",
                 cached_ck, ck.name)
        return None
    return cached


# ------------------------------------------------------------------ endpoints
@app.get("/api/forecast/latest")
def forecast_latest(refresh: bool = Query(False)):
    out = None
    if refresh:
        out = refresh_cache()
    if out is None:
        out = cached_forecast()
    if out is None:
        out = refresh_cache()
    if out is None:
        return JSONResponse(status_code=503, content={
            "error": _state["error"] or "no forecast available",
            "as_of": str(sgt_now()), "data_age_minutes": {}, "horizons": [],
            "degraded": True,
        })
    age = _age_min(out.get("issue_time"))
    out["data_age_minutes"]["forecast"] = age
    thr = CFG["api"]["staleness_thresholds_min"]
    out["stale"] = bool(age is not None and age > thr.get("nea", 30) * 4)
    out["degraded"] = False
    return out


@app.get("/api/forecast/history")
def forecast_history(days: int = Query(7, ge=1, le=90)):
    """Past forecasts joined to observed outcomes.

    Built by replaying stored issue times through the same model, so the
    comparison is against the same rows the offline evaluation uses.
    """
    f = get_forecaster()
    if f is None:
        return JSONResponse(status_code=503, content={
            "error": _state["error"], "as_of": str(sgt_now()), "items": [],
            "degraded": True})
    proc = ROOT / CFG["paths"]["processed"]
    p = proc / f"dataset_{f.track}.parquet"
    if not p.exists():
        return {"as_of": str(sgt_now()), "items": [], "n": 0, "degraded": True,
                "note": "dataset not built"}
    df = pd.read_parquet(p).sort_values("timestamp")
    end = df["timestamp"].max()
    sub = df[df["timestamp"] > end - pd.Timedelta(days=days)]
    items = []
    for t in sub["timestamp"]:
        try:
            r = f.replay(t)
        except Exception:
            continue
        items.append({
            "issue_time": r["issue_time"],
            "kt_now": r["kt_now"],
            "horizons": [{"h": h["horizon_h"], "valid_time": h["valid_time"],
                          "median": h["median"], "lower": h["lower"], "upper": h["upper"],
                          "actual": h["actual_ghi"], "clearsky": h["clearsky_ghi"]}
                         for h in r["horizons"]],
        })
    return {"as_of": str(sgt_now()), "days": days, "n": len(items), "items": items,
            "target_product": "ERA5 reanalysis", "degraded": False,
            "data_age_minutes": {"dataset": _age_min(end)}}


@app.get("/api/raingrid/latest")
def raingrid_latest():
    """Interpolated grid + raw station values + mask."""
    spec = GridSpec.from_config(CFG)
    out = {
        "as_of": str(sgt_now()),
        "extent": {"lon_min": spec.lon_min, "lon_max": spec.lon_max,
                   "lat_min": spec.lat_min, "lat_max": spec.lat_max},
        "bounds_leaflet": spec.bounds_leaflet(),
        "nx": spec.nx, "ny": spec.ny,
        "cell_km": list(spec.cell_size_km()),
        "units": "mm per 5-minute interval",
        "data_age_minutes": {},
    }
    try:
        from data._client import ApiClient
        f = get_forecaster()
        client = ApiClient(delay_s=0.3, backoff_s=11, json_code_field="code")
        if f is None:
            raise RuntimeError(_state["error"] or "model unavailable")
        grid, meta = f._live_grid(client, sgt_now())
        # row 0 is south; flip so the client can draw north-up directly
        disp = spec.to_display(grid)
        out.update({
            "grid": np.round(disp[0], 4).tolist(),
            "mask": disp[-1].astype(int).tolist(),
            "stations": meta.get("stations", []),
            "motion": {"vx": meta.get("motion_vx"), "vy": meta.get("motion_vy"),
                       "valid": meta.get("motion_valid")},
            "n_stations_reporting": meta.get("n_stations_reporting"),
            "latest_reading": meta.get("latest_reading"),
            "newest_frame_skipped": meta.get("newest_frame_skipped"),
            "wet_fraction": meta.get("wet_frac"),
            "degraded": False,
        })
        out["data_age_minutes"]["nea"] = meta.get("age_minutes")
    except Exception as e:
        out.update({"degraded": True, "error": str(e)[:200], "grid": [], "mask": [],
                    "stations": [], "motion": {"vx": 0, "vy": 0, "valid": 0}})
    return out


@app.get("/api/raingrid/series")
def raingrid_series(minutes: int = Query(180, ge=15, le=360)):
    """A sequence of 5-minute rain fields for the map time slider.

    Scrubbing the last few hours is the most direct way to see whether the spatial
    modality carries usable signal at all - the fields are sparse enough that a
    static snapshot can look identically empty whether or not anything is moving.
    """
    spec = GridSpec.from_config(CFG)
    out = {"as_of": str(sgt_now()), "minutes": minutes,
           "bounds_leaflet": spec.bounds_leaflet(),
           "extent": {"lon_min": spec.lon_min, "lon_max": spec.lon_max,
                      "lat_min": spec.lat_min, "lat_max": spec.lat_max},
           "nx": spec.nx, "ny": spec.ny, "units": "mm per 5-minute interval",
           "frames": [], "data_age_minutes": {}}
    try:
        from data._client import ApiClient
        from data.build_grids import Interpolator
        from data.fetch_nea import parse_response
        g = CFG["grid"]
        client = ApiClient(delay_s=0.3, backoff_s=11, json_code_field="code")
        now = sgt_now()
        payload = client.get_json(f"{CFG['nea']['base_url']}/rainfall",
                                  params={"date": now.strftime("%Y-%m-%d")})
        df, meta = parse_response(payload)
        if df.empty:
            raise RuntimeError("no gauge readings for today")
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True) \
            .dt.tz_convert(CFG["site"]["timezone"]).dt.tz_localize(None)

        counts = df.groupby("timestamp")["station_id"].nunique()
        floor = max(CFG["nea"]["min_stations"], int(0.5 * counts.max()))
        good = counts[counts >= floor].index.sort_values()
        good = [t for t in good if (now - t).total_seconds() / 60.0 <= minutes]

        interp = Interpolator(spec, g["method"], g["idw_power"], g["idw_max_km"])
        for ts in good:
            grp = df[df["timestamp"] == ts]
            field, mask, n_used = interp(grp["station_id"].tolist(),
                                         grp["value"].tolist(), meta)
            disp = spec.to_display(field)
            out["frames"].append({
                "timestamp": str(ts),
                "age_minutes": round((now - ts).total_seconds() / 60.0, 1),
                "n_stations": int(n_used),
                "wet_fraction": float((field > 0.01).mean()),
                "max_mm": float(field.max()),
                "grid": np.round(disp, 4).tolist(),
            })
        if out["frames"]:
            out["mask"] = spec.to_display(mask).astype(int).tolist()
            out["data_age_minutes"]["nea"] = out["frames"][-1]["age_minutes"]
        out["n"] = len(out["frames"])
        out["degraded"] = not out["frames"]
    except Exception as e:
        out.update({"degraded": True, "error": str(e)[:200], "frames": [], "n": 0,
                    "mask": []})
    return out


@app.get("/api/stations")
def stations():
    p = ROOT / CFG["paths"]["raw"] / "nea" / "stations_rainfall.json"
    known = json.loads(p.read_text()) if p.exists() else {}
    live_vals, age, err = {}, None, None
    try:
        from data._client import ApiClient
        client = ApiClient(delay_s=0.3, backoff_s=11, json_code_field="code")
        from data.fetch_nea import parse_response
        payload = client.get_json(f"{CFG['nea']['base_url']}/rainfall",
                                  params={"date": sgt_now().strftime("%Y-%m-%d")})
        df, meta = parse_response(payload)
        known.update({k: v for k, v in meta.items()})
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
            counts = df.groupby("timestamp")["station_id"].nunique()
            floor = max(CFG["nea"]["min_stations"], int(0.5 * counts.max()))
            usable = counts[counts >= floor]
            ts = usable.index.max() if not usable.empty else counts.idxmax()
            cur = df[df["timestamp"] == ts]
            live_vals = dict(zip(cur["station_id"], cur["value"]))
            age = _age_min(pd.Timestamp(ts).tz_convert(CFG["site"]["timezone"]).tz_localize(None))
    except Exception as e:
        err = str(e)[:200]

    items = []
    for sid, m in sorted(known.items()):
        items.append({
            "id": sid, "name": m.get("name"),
            "lat": m.get("latitude"), "lon": m.get("longitude"),
            "value": (float(live_vals[sid]) if sid in live_vals else None),
            "online": sid in live_vals,
        })
    return {"as_of": str(sgt_now()), "n_known": len(items),
            "n_online": sum(1 for i in items if i["online"]),
            "stations": items, "data_age_minutes": {"nea": age},
            "degraded": bool(err), "error": err,
            "note": ("Station membership changes over time (51 gauges in Jan-2020, "
                     "88 today); readings are joined by stationId per response.")}


@app.get("/api/verification")
def verification():
    """Rolling skill, coverage and stratified calibration from the last evaluation."""
    reports = ROOT / CFG["paths"]["reports"]
    out = {"as_of": str(sgt_now()), "degraded": True, "tracks": {},
           "data_age_minutes": {}}
    found = False
    for track in ("A", "B"):
        p = reports / f"evaluation_{track}.json"
        if not p.exists():
            continue
        found = True
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out["tracks"][track] = {
            "baselines": data.get("baselines", {}).get("models", {}),
            "models": data.get("models", []),
            "n_test": data.get("baselines", {}).get("n_test"),
        }
        out["data_age_minutes"][f"evaluation_{track}"] = _age_min(
            datetime.fromtimestamp(p.stat().st_mtime))
    out["degraded"] = not found
    out["nominal_interval"] = round(max(CFG["quantiles"]) - min(CFG["quantiles"]), 4)
    out["note"] = ("Verification uses the same ERA5 product the model was trained on. "
                   "It is a consistency check, not independent ground truth.")
    return out


@app.get("/api/model/info")
def model_info():
    f = get_forecaster()
    proc = ROOT / CFG["paths"]["processed"]
    limits = (ROOT / "README.md")
    known_limits = []
    if limits.exists():
        text = limits.read_text()
        if "## Known limitations" in text:
            sec = text.split("## Known limitations", 1)[1]
            sec = sec.split("\n## ", 1)[0]
            known_limits = [ln.strip("- ").strip() for ln in sec.splitlines()
                            if ln.strip().startswith("-")]
    if f is None:
        return {"as_of": str(sgt_now()), "degraded": True,
                "error": _state["error"], "known_limitations": known_limits,
                "data_age_minutes": {}}
    stats_p = proc / f"train_stats_{f.track}.json"
    stats = json.loads(stats_p.read_text()) if stats_p.exists() else {}
    # info() is the shape-agnostic accessor: a torch checkpoint dict for the neural
    # model, the sidecar for CatBoost. Keys absent from one simply come back None.
    meta = f.info()
    return {
        "as_of": str(sgt_now()), "degraded": False,
        "checkpoint": f.checkpoint_name, "variant": f.variant, "track": f.track,
        "seed": meta.get("seed"), "trained_at": meta.get("trained_at"),
        "n_train": meta.get("n_train"), "n_val": meta.get("n_val"),
        "n_test": meta.get("n_test"),
        "params": meta.get("params"),
        "best_val_pinball": meta.get("best_val_pinball"),
        "quantiles": f.quantiles,
        "nominal_interval": round(max(f.quantiles) - min(f.quantiles), 4),
        "horizons": f.horizons,
        "features": getattr(f, "features", stats.get("features", [])),
        "nwp_cols": stats.get("nwp_cols", []),
        "nwp_models": CFG["nwp"]["models"],
        "target_product": "ERA5 reanalysis (Open-Meteo archive)",
        "daylight_window": [CFG["daylight"]["start_hour"], CFG["daylight"]["end_hour"]],
        "known_limitations": known_limits,
        "data_age_minutes": {"checkpoint": _age_min(meta.get("trained_at"))},
    }


@app.get("/api/health")
def health():
    """Data freshness per source, with explicit staleness warnings."""
    thr = CFG["api"]["staleness_thresholds_min"]
    raw = ROOT / CFG["paths"]["raw"]
    sources = {}

    wx = raw / "weather_hourly.csv"
    if wx.exists():
        try:
            last = pd.read_csv(wx, usecols=["timestamp"]).iloc[-1, 0]
            sources["era5"] = {"last": str(last), "age_minutes": _age_min(last),
                               "threshold": thr["era5"]}
        except Exception as e:
            sources["era5"] = {"error": str(e)[:120]}
    else:
        sources["era5"] = {"error": "not fetched"}

    nwp = raw / "nwp_hourly.csv"
    if nwp.exists():
        sources["nwp"] = {"last": str(datetime.fromtimestamp(nwp.stat().st_mtime)),
                          "age_minutes": _age_min(datetime.fromtimestamp(nwp.stat().st_mtime)),
                          "threshold": thr["nwp"], "models": CFG["nwp"]["models"]}
    else:
        sources["nwp"] = {"error": "not fetched"}

    man = raw / "nea" / "manifest_rainfall.json"
    if man.exists():
        try:
            m = json.loads(man.read_text())
            days = sorted(m)
            complete = sum(1 for v in m.values() if v.get("complete"))
            sources["nea"] = {
                "last_day": days[-1] if days else None,
                "days": len(days), "complete_days": complete,
                "partial_days": sum(1 for v in m.values()
                                    if 0 < v.get("frames", 0) < 0.9 * v.get("expected_frames", 1)),
                "empty_days": sum(1 for v in m.values() if v.get("frames", 0) == 0),
                "threshold": thr["nea"],
            }
        except Exception as e:
            sources["nea"] = {"error": str(e)[:120]}
    else:
        sources["nea"] = {"error": "not fetched"}

    warnings = []
    for name, s in sources.items():
        if "error" in s:
            warnings.append(f"{name}: {s['error']}")
        elif s.get("age_minutes") is not None and s.get("threshold") is not None:
            if s["age_minutes"] > s["threshold"]:
                warnings.append(
                    f"{name} is stale: {s['age_minutes']:.0f} min > {s['threshold']} min")
    ck = latest_checkpoint()
    return {
        "as_of": str(sgt_now()), "sources": sources, "warnings": warnings,
        "healthy": not warnings,
        "replay_mode": REPLAY_MODE, "replay_date": REPLAY_DATE,
        "checkpoint": ck.name if ck else None,
        "model_error": _state["error"],
        "last_refresh": _state["last_refresh"],
        "data_age_minutes": {k: v.get("age_minutes") for k, v in sources.items()},
    }


# ------------------------------------------------------------------- frontend
@app.get("/")
def index():
    f = STATIC / "index.html"
    if not f.exists():
        raise HTTPException(404, "frontend not built")
    return FileResponse(f)


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ------------------------------------------------------------------ scheduler
# ------------------------------------------------------- continuous operation
def in_daylight(t: pd.Timestamp | None = None) -> bool:
    """Issue times run 08:00-17:00 SGT.

    Night targets were excluded from training explicitly (they are ~0 W/m2 and
    trivially easy), so issuing after dark would pad the verification sample with
    rows the model was never fitted for and flatter every metric.
    """
    t = t or sgt_now()
    return CFG["daylight"]["start_hour"] <= t.hour <= CFG["daylight"]["end_hour"]


def issue_forecast() -> dict | None:
    """Scheduler job: refresh the cache AND append the forecast to the store."""
    if not in_daylight():
        log.debug("outside daylight window - not issuing")
        return None
    out = refresh_cache()
    if out is None:
        return None
    store = get_store()
    if store is None:
        return out
    try:
        import storage
        n = storage.record_package(store, out, model_id(),
                                   mode="replay" if out.get("mode") == "replay" else "live")
        log.info("stored %d forecast rows for %s", n, out["issue_time"])
    except Exception as e:                          # degraded, never fatal
        log.warning("could not store forecast: %s", str(e)[:160])
    return out


def _fetch_ghi(url: str, start: str, end: str, model: str | None = None) -> list[tuple[str, float]]:
    """Observed GHI for a date range.

    `model` MUST be pinned for the provisional lane. Open-Meteo's default
    (best_match) returns values bit-identical to the ERA5 archive - verified in
    data/fetch_nwp.py and guarded by tests/test_nwp_leakage.py - so an unpinned
    request makes the provisional lane a relabelled copy of the final one rather
    than an independent early check.
    """
    from data._client import ApiClient
    site = CFG["site"]
    client = ApiClient(delay_s=1.0, backoff_s=8.0)
    params = {
        "latitude": site["latitude"], "longitude": site["longitude"],
        "hourly": "shortwave_radiation", "timezone": site["timezone"],
        "start_date": start, "end_date": end,
    }
    if model:
        params["models"] = model
    h = client.get_json(url, params=params)["hourly"]
    return list(zip(h["time"], h["shortwave_radiation"]))


def collect_outcomes(source: str) -> int:
    """Join arriving truth to forecasts already issued. Returns rows written.

    'analysis' is the PROVISIONAL lane (~1h behind); 'era5' is the real target and
    only becomes available after `period.archive_lag_days`.
    """
    store = get_store()
    if store is None:
        return 0
    try:
        import storage
        now = sgt_now()
        if source == "era5":
            cutoff = now - timedelta(days=CFG["period"]["archive_lag_days"])
            url, model = CFG["weather"]["archive_url"], None
        else:
            cutoff = now - timedelta(hours=1)
            # pinned, NOT best_match - see _fetch_ghi
            url = CFG["nwp"]["live_url"]
            model = CFG["nwp"]["models"][0]

        due = storage.missing_outcomes(store, source, before=str(cutoff))
        if not due:
            return 0
        start = pd.Timestamp(min(due)).strftime("%Y-%m-%d")
        end = pd.Timestamp(max(due)).strftime("%Y-%m-%d")
        wanted = set(due)
        # Open-Meteo returns "2026-08-15T15:00"; the store keys on
        # str(pd.Timestamp) = "2026-08-15 15:00:00". Normalise both sides or every
        # match silently fails and the lane stays empty forever.
        rows = [(str(pd.Timestamp(t)), g) for t, g in _fetch_ghi(url, start, end, model)]
        rows = [(t, g) for t, g in rows if t in wanted]
        n = storage.record_outcomes(store, rows, source)
        log.info("outcomes[%s]: %d of %d pending valid times filled", source, n, len(due))
        return n
    except Exception as e:                          # degraded, never fatal
        log.warning("outcome collection (%s) failed: %s", source, str(e)[:160])
        return 0


@app.get("/api/verification/rolling")
def verification_rolling(window: str = Query("7d"),
                         source: str = Query("era5"),
                         horizon: int = Query(1, ge=1, le=24)):
    """How the SERVED model is doing on forecasts it actually issued.

    Distinct from /api/verification, which serves the static offline evaluate.py
    report. Never mixes the two outcome lanes - `provisional` says which is which.
    """
    import verification

    store = get_store()
    base = {"as_of": str(sgt_now()), "window": window, "source": source}
    if store is None:
        return {**base, "degraded": True, "error": "forecast store unavailable",
                "horizons": {}, "timeline": [], "data_age_minutes": {}}
    if window not in verification.WINDOWS:
        raise HTTPException(422, f"window must be one of {sorted(verification.WINDOWS)}")
    if source not in ("era5", "analysis"):
        raise HTTPException(422, "source must be 'era5' or 'analysis'")
    try:
        import storage
        mid = model_id()
        rep = verification.score_window(store, CFG, window=window, source=source,
                                        model_id=mid, now=sgt_now())
        rep["timeline"] = verification.timeline(store, window=window, source=source,
                                                model_id=mid, horizon=horizon,
                                                now=sgt_now())
        rep["store"] = storage.summary(store)
        rep["as_of"] = str(sgt_now())
        rep["degraded"] = rep["n_total"] == 0
        if rep["degraded"]:
            rep["error"] = ("no forecasts scored yet in this window - outcomes arrive "
                            "~1h behind for analysis, ~2-3 days for ERA5")
        rep["data_age_minutes"] = {"last_issue": _age_min(rep["store"].get("last_issue"))}
        return rep
    except Exception as e:
        log.warning("rolling verification failed: %s", str(e)[:200])
        return {**base, "degraded": True, "error": str(e)[:200],
                "horizons": {}, "timeline": [], "data_age_minutes": {}}


@app.on_event("startup")
def _startup():
    if os.environ.get("NO_SCHEDULER") == "1":
        return
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        sched = BackgroundScheduler(daemon=True)
        every = CFG["api"]["refresh_minutes"]
        now = datetime.now()
        # NOTE: do NOT pass next_run_time=None - APScheduler reads that as "paused"
        # and the job never fires. The original refresh_cache job had exactly that
        # bug, so nothing was ever refreshed on a schedule. An explicit start time
        # both fixes it and makes the first run happen at boot instead of one full
        # interval later.
        sched.add_job(issue_forecast, "interval", minutes=every,
                      next_run_time=now + timedelta(seconds=5),
                      id="issue_forecast", max_instances=1, coalesce=True)
        # provisional truth lands within the hour; ERA5 takes days, so once daily
        sched.add_job(lambda: collect_outcomes("analysis"), "interval", minutes=60,
                      next_run_time=now + timedelta(seconds=30),
                      id="outcomes_analysis", max_instances=1, coalesce=True)
        sched.add_job(lambda: collect_outcomes("era5"), "interval", hours=24,
                      next_run_time=now + timedelta(seconds=60),
                      id="outcomes_era5", max_instances=1, coalesce=True)
        sched.start()
        _state["scheduler"] = sched
        log.info("scheduler started: issue every %d min, outcomes hourly/daily; "
                 "next runs %s", every,
                 {j.id: str(j.next_run_time) for j in sched.get_jobs()})
    except Exception as e:
        log.warning("scheduler unavailable: %s", e)
