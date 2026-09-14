"""Gauge point readings -> (C, ny, nx) rasters + mask channel + motion vectors.

Channels (C = len(frame_offsets_min) + 1):
    0..n-1  rain rate at t, t-30min, t-60min   (mm per 5-min slot)
    n       MASK: 1 where a reporting gauge lies within `idw_max_km`, else 0

The mask exists so the CNN can tell "no rain" from "no data". Without it, an
interpolated-from-nothing cell over the sea is indistinguishable from a genuine
dry cell, and the model learns to trust regions that carry no information.

Station membership is read from EACH response's station list and joined by
stationId (verified: 51 gauges in Jan-2020, 88 today, and it varies between
adjacent frames), so the interpolation weights are rebuilt whenever the
reporting set changes - never assumed fixed.

Motion vectors come from cross-correlating consecutive frames. Singapore rain
fields are very sparse (a typical midday frame has 0-1 wet gauges out of 60+),
so when there is not enough signal we emit (0, 0) with motion_valid = 0 rather
than an arbitrary displacement.

Usage:
    python data/build_grids.py [--start 2020-01-01] [--end 2026-09-05]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data._client import load_config, log, read_json, resolve_path, setup_logging  # noqa: E402
from data.geometry import GridSpec, haversine_km  # noqa: E402


class Interpolator:
    """IDW / linear interpolation onto a fixed grid, cached per station set."""

    def __init__(self, spec: GridSpec, method: str, power: float, max_km: float):
        self.spec = spec
        self.method = method
        self.power = power
        self.max_km = max_km
        spec.self_test()                      # geometry guard, every run
        self.LON, self.LAT = spec.mesh()      # (ny, nx)
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def _weights(self, key: tuple, lons: np.ndarray, lats: np.ndarray):
        """-> (weights (ny,nx,S) normalised, mask (ny,nx))."""
        if key in self._cache:
            return self._cache[key]
        # (ny, nx, S) distances
        d = haversine_km(self.LON[..., None], self.LAT[..., None],
                         lons[None, None, :], lats[None, None, :])
        mask = (d.min(axis=-1) <= self.max_km).astype(np.float32)
        d = np.maximum(d, 1e-3)
        w = 1.0 / np.power(d, self.power)
        w /= w.sum(axis=-1, keepdims=True)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = (w.astype(np.float32), mask)
        return self._cache[key]

    def __call__(self, station_ids, values, meta: dict):
        """Interpolate one frame. Returns (field (ny,nx), mask (ny,nx), n_used)."""
        lons, lats, vals = [], [], []
        for sid, v in zip(station_ids, values):
            m = meta.get(sid)
            if m is None or v is None or not np.isfinite(v):
                continue
            if m["latitude"] is None or m["longitude"] is None:
                continue
            lons.append(m["longitude"]); lats.append(m["latitude"]); vals.append(float(v))
        if not lons:
            z = np.zeros((self.spec.ny, self.spec.nx), np.float32)
            return z, z.copy(), 0
        lons = np.asarray(lons); lats = np.asarray(lats)
        vals = np.asarray(vals, dtype=np.float32)
        key = tuple(sorted(zip(np.round(lons, 5), np.round(lats, 5))))
        w, mask = self._weights(key, lons, lats)
        field = (w * vals[None, None, :]).sum(axis=-1).astype(np.float32)
        return field, mask, len(vals)


def estimate_motion(prev: np.ndarray, curr: np.ndarray, max_shift: int = 8,
                    min_signal: float = 1e-3) -> tuple[float, float, int]:
    """Displacement (dx, dy) in cells from prev -> curr by cross-correlation.

    Returns (dx, dy, valid). valid=0 when the field carries too little rain to
    support a displacement estimate, in which case (0, 0) is returned.
    """
    if prev.sum() < min_signal or curr.sum() < min_signal:
        return 0.0, 0.0, 0
    a = prev - prev.mean()
    b = curr - curr.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0, 0.0, 0
    best, bdx, bdy = -np.inf, 0, 0
    ny, nx = prev.shape
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            ys, ye = max(0, dy), min(ny, ny + dy)
            xs, xe = max(0, dx), min(nx, nx + dx)
            if ye - ys < ny // 2 or xe - xs < nx // 2:
                continue
            pa = a[ys - dy:ye - dy, xs - dx:xe - dx]
            pb = b[ys:ye, xs:xe]
            denom = np.linalg.norm(pa) * np.linalg.norm(pb)
            if denom < 1e-6:
                continue
            c = float((pa * pb).sum() / denom)
            if c > best:
                best, bdx, bdy = c, dx, dy
    if best < 0.2:                      # correlation too weak to trust
        return 0.0, 0.0, 0
    return float(bdx), float(bdy), 1


def load_day(raw: Path, endpoint: str, day: date) -> pd.DataFrame | None:
    p = raw / "nea" / endpoint / f"{day.isoformat()}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def target_times(cfg: dict, day: date) -> list[pd.Timestamp]:
    """Hourly issue times t for which a full sample can exist on this day."""
    lo = cfg["daylight"]["start_hour"]
    hi = cfg["daylight"]["end_hour"] - max(cfg["horizons"])
    return [pd.Timestamp(day) + pd.Timedelta(hours=h) for h in range(lo, hi + 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--endpoint", default="rainfall")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    raw = resolve_path(cfg, "raw")
    grids = resolve_path(cfg, "grids")
    spec = GridSpec.from_config(cfg)
    g = cfg["grid"]
    interp = Interpolator(spec, g["method"], g["idw_power"], g["idw_max_km"])
    offsets = g["frame_offsets_min"]
    n_ch = len(offsets) + 1

    meta = read_json(raw / "nea" / f"stations_{args.endpoint}.json") or {}
    if not meta:
        log.error("no station metadata; run data/fetch_nea.py first")
        return 1
    log.info("grid %dx%d  cell %.2fx%.2f km  |  %d known stations  |  %d channels",
             spec.nx, spec.ny, *spec.cell_size_km(), len(meta), n_ch)

    start = date.fromisoformat(args.start or cfg["period"]["start"])
    end = (date.fromisoformat(args.end) if args.end
           else date.today() - timedelta(days=cfg["period"]["archive_lag_days"]))

    man_path = grids / "manifest.csv"
    existing = set()
    if man_path.exists() and not args.force:
        try:
            existing = set(pd.read_csv(man_path)["date"].astype(str))
        except Exception:
            existing = set()

    rows, day, n_days = [], start, 0
    if man_path.exists() and not args.force:
        rows = pd.read_csv(man_path).to_dict("records")

    while day <= end:
        key = day.isoformat()
        if key in existing:
            day += timedelta(days=1)
            continue
        df = load_day(raw, args.endpoint, day)
        if df is None or df.empty:
            day += timedelta(days=1)
            continue

        # frame lookup: exact 5-min slot -> {station_id: value}
        by_ts = {ts: grp for ts, grp in df.groupby("timestamp")}
        times = target_times(cfg, day)
        stack, recs = [], []
        for t in times:
            chans, n_report, ok = [], [], True
            for off in offsets:
                ts = t - pd.Timedelta(minutes=off)
                grp = by_ts.get(ts)
                if grp is None:
                    ok = False
                    break
                field, mask, n_used = interp(grp["station_id"].tolist(),
                                             grp["value"].tolist(), meta)
                chans.append(field)
                n_report.append(n_used)
            if not ok or not chans:
                continue
            arr = np.stack(chans + [mask], axis=0).astype(np.float32)
            dx, dy, valid = estimate_motion(chans[1], chans[0]) if len(chans) > 1 else (0., 0., 0)
            recs.append({
                "timestamp": t.isoformat(), "date": key, "index": len(stack),
                "n_stations_reporting": int(min(n_report)),
                "rain_mean": float(chans[0].mean()), "rain_max": float(chans[0].max()),
                "wet_frac": float((chans[0] > 0.01).mean()),
                "rain_std": float(chans[0].std()),
                "mask_frac": float(mask.mean()),
                "motion_vx": dx, "motion_vy": dy, "motion_valid": valid,
            })
            stack.append(arr)

        if stack:
            out = grids / f"{key}.npy"
            np.save(out, np.stack(stack, axis=0))
            for r in recs:
                r["path"] = out.name
                r["method"] = g["method"]
            rows.extend(recs)
        n_days += 1
        if n_days % 100 == 0:
            pd.DataFrame(rows).to_csv(man_path, index=False)
            log.info("  %s  frames=%d  total=%d", key, len(stack), len(rows))
        day += timedelta(days=1)

    if not rows:
        log.error("no grids built - is the NEA backfill populated?")
        return 1
    man = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    man.to_csv(man_path, index=False)

    # ---- coverage assertions
    n = len(man)
    low = int((man["n_stations_reporting"] < cfg["nea"]["min_stations"]).sum())
    log.info("grids: %d frames over %d days -> %s", n, man["date"].nunique(), man_path)
    log.info("  stations/frame: min=%d median=%d max=%d  (below floor: %d)",
             man["n_stations_reporting"].min(), int(man["n_stations_reporting"].median()),
             man["n_stations_reporting"].max(), low)
    log.info("  mask coverage: %.1f%% of cells | wet fraction mean %.3f | motion valid %.1f%%",
             100 * man["mask_frac"].mean(), man["wet_frac"].mean(),
             100 * man["motion_valid"].mean())
    if low:
        log.warning("  %d frames below the %d-station floor - they are kept but flagged",
                    low, cfg["nea"]["min_stations"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
