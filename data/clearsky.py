"""Clear-sky irradiance - computed ONE way, for training and for serving.

This module exists because the two paths had drifted apart. `data/fetch_weather.py`
integrated clear-sky over the preceding hour, while `predict.py` sampled a single
point at t-30 under a comment asserting the two matched. They did not, and the
difference is largest exactly where it hurts: near sunrise, where irradiance is
strongly convex and a point estimate at the hour centre understates the hour mean.

Open-Meteo's ERA5 `shortwave_radiation` is the MEAN OVER THE PRECEDING HOUR - the
row labelled 17:00 covers 16:00-17:00 - so clear-sky must be an hour-mean over the
same window. Anything that needs clear-sky calls `hour_mean_ghi`; nothing computes
it locally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _location(cfg: dict):
    import pvlib

    site = cfg["site"]
    return pvlib.location.Location(
        latitude=site["latitude"], longitude=site["longitude"],
        altitude=site.get("altitude_m", 0), tz=site["timezone"],
    )


def hour_mean_ghi(cfg: dict, timestamps) -> np.ndarray:
    """Mean clear-sky GHI over the hour PRECEDING each timestamp, in W/m2.

    Midpoint rule over `clearsky.integration_window_min` at
    `clearsky.integration_step_min` resolution: offsets are
    t-60+step/2, ..., t-step/2 for the default 60/5.
    """
    cscfg = cfg.get("clearsky", {})
    site = cfg["site"]
    step = int(cscfg.get("integration_step_min", 5))
    window = int(cscfg.get("integration_window_min", 60))
    model = cscfg.get("model", "ineichen")

    idx = pd.DatetimeIndex(timestamps)
    base = idx.tz_localize(site["timezone"]) if idx.tz is None else idx

    loc = _location(cfg)
    offsets = [-window + step * (k + 0.5) for k in range(window // step)]
    acc = np.zeros(len(base), dtype=float)
    for off in offsets:
        acc += loc.get_clearsky(base + pd.Timedelta(minutes=off), model=model)["ghi"].to_numpy()
    return acc / len(offsets)


def kt_from_ghi(cfg: dict, ghi, clearsky) -> np.ndarray:
    """Clear-sky index with the configured floor and cap, as used everywhere."""
    cscfg = cfg.get("clearsky", {})
    floor = float(cscfg.get("min_ghi_for_ratio", 5.0))
    cap = float(cscfg.get("max_ratio", 1.5))
    ghi = np.asarray(ghi, dtype=float)
    cs = np.asarray(clearsky, dtype=float)
    kt = np.clip(ghi / np.clip(cs, floor, None), 0.0, cap)
    return np.where(cs <= 0, 0.0, kt)
