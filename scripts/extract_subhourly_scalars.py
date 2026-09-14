"""Extract rain statistics from the t-30 and t-60 raster channels.

build_grids.py computes its scalars from `chans[0]` (the frame at t) ONLY, so the
t-30 and t-60 frames are stored but never summarised - the tabular models see no
sub-hourly rain history at all. This reads the rasters already on disk and emits
the missing statistics plus their tendencies; nothing is refetched.

Output: artifacts/processed/subhourly_scalars.parquet, keyed by timestamp.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data._client import load_config, log, resolve_path, setup_logging  # noqa: E402

# channel 0 = t (already summarised by build_grids), 1 = t-30, 2 = t-60, 3 = mask
LAG_CHANNELS = {1: "m30", 2: "m60"}


def frame_stats(ch: np.ndarray, tag: str) -> dict:
    return {
        f"rain_mean_{tag}": float(ch.mean()),
        f"rain_max_{tag}": float(ch.max()),
        f"wet_frac_{tag}": float((ch > 0.01).mean()),
        f"rain_std_{tag}": float(ch.std()),
    }


def main() -> int:
    setup_logging()
    cfg = load_config(None)
    grids = resolve_path(cfg, "grids")
    proc = resolve_path(cfg, "processed")

    man = pd.read_csv(grids / "manifest.csv", parse_dates=["timestamp"])
    log.info("manifest: %d frames across %d day-files", len(man), man["path"].nunique())

    out_rows = []
    for path, grp in man.groupby("path", sort=True):
        arr = np.load(grids / path, mmap_mode="r")
        for ts, idx in zip(grp["timestamp"], grp["index"]):
            frame = arr[int(idx)]
            rec = {"timestamp": ts}
            for ch_i, tag in LAG_CHANNELS.items():
                rec.update(frame_stats(np.asarray(frame[ch_i]), tag))
            out_rows.append(rec)

    sub = pd.DataFrame(out_rows).sort_values("timestamp").reset_index(drop=True)

    # tendencies: how the field is CHANGING inside the hour, which is the part a
    # t+1h forecast can use and an hourly NWP field cannot express
    base = man[["timestamp", "rain_mean", "rain_max", "wet_frac"]].copy()
    sub = sub.merge(base, on="timestamp", how="left")
    sub["rain_mean_d30"] = sub["rain_mean"] - sub["rain_mean_m30"]
    sub["rain_mean_d60"] = sub["rain_mean_m30"] - sub["rain_mean_m60"]
    sub["wet_frac_d30"] = sub["wet_frac"] - sub["wet_frac_m30"]
    sub["wet_frac_d60"] = sub["wet_frac_m30"] - sub["wet_frac_m60"]
    sub["rain_max_d30"] = sub["rain_max"] - sub["rain_max_m30"]
    sub = sub.drop(columns=["rain_mean", "rain_max", "wet_frac"])

    out = proc / "subhourly_scalars.parquet"
    sub.to_parquet(out, index=False)
    cols = [c for c in sub.columns if c != "timestamp"]
    log.info("wrote %s  rows=%d  new features=%d", out, len(sub), len(cols))
    nz = {c: float((sub[c].abs() > 1e-9).mean()) for c in cols}
    for c, f in sorted(nz.items(), key=lambda kv: -kv[1]):
        log.info("  %-18s non-zero on %5.1f%% of frames", c, 100 * f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
