"""Rolling verification: score forecasts that were actually issued, as truth lands.

`evaluate.py` answers "how did this model do on the held-out test split", once,
offline. This answers "how is the served model doing right now", continuously, over
a moving window - which needs the forecasts to have been stored when they were made
(storage.py) and the outcomes to be joined in as they arrive.

All scoring goes through metrics.py, the same functions evaluate.py calls. Nothing
here reimplements a metric.

THE TWO LANES ARE NEVER MERGED. 'era5' is the real target and lags ~2 days;
'analysis' is available within the hour but is a model product the forecaster also
consumes, so it flatters. Every result carries `provisional` and the caller is
expected to show it.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

import metrics
import storage

WINDOWS = {"24h": 1, "7d": 7, "30d": 30}


def window_start(window: str, now: pd.Timestamp) -> str:
    days = WINDOWS.get(window)
    if days is None:
        raise ValueError(f"window must be one of {sorted(WINDOWS)}, got {window!r}")
    return str(now - timedelta(days=days))


def score_window(conn, cfg: dict, *, window: str = "7d", source: str = "era5",
                 model_id: str | None = None, mode: str | None = None,
                 now: pd.Timestamp | None = None) -> dict:
    """Per-horizon metrics over one lane and one window.

    Returns `horizons: {}` with n=0 rather than raising when nothing has been
    scored yet - a cold store is the normal state on day one, not an error.
    """
    now = now or pd.Timestamp.now()
    since = window_start(window, now)
    rows = storage.scored_rows(conn, source, since, model_id=model_id, mode=mode)

    quantiles = cfg["quantiles"]
    bins = [tuple(b) for b in cfg["eval"]["cloud_bins"]]
    out_h: dict[str, dict] = {}

    by_h: dict[int, list] = {}
    for r in rows:
        by_h.setdefault(int(r["horizon_h"]), []).append(r)

    for h in sorted(by_h):
        rs = by_h[h]
        pred = np.array([r["quantiles"] for r in rs], dtype=float)
        y = np.array([r["actual"] for r in rs], dtype=float)
        cs = np.array([r["clearsky"] if r["clearsky"] is not None else np.nan
                       for r in rs], dtype=float)
        kt = np.array([r["kt_now"] if r["kt_now"] is not None else np.nan
                       for r in rs], dtype=float)

        ok = np.isfinite(y) & np.isfinite(pred).all(axis=1)
        if not ok.any():
            continue
        pred, y, cs, kt = pred[ok], y[ok], cs[ok], kt[ok]
        # a missing clear-sky would poison the k_t-space dispersion metric
        cs = np.where(np.isfinite(cs), cs, np.nan)

        rep = metrics.full_report(pred, y, cs, kt, quantiles, bins)
        rep["n_live"] = sum(1 for r, keep in zip(rs, ok) if keep and r["mode"] == "live")
        rep["n_replay"] = int(rep["n"]) - rep["n_live"]
        out_h[f"t+{h}h"] = rep

    return {
        "window": window,
        "since": since,
        "source": source,
        "provisional": source == "analysis",
        "source_label": ("Open-Meteo analysis - PROVISIONAL, not ERA5"
                         if source == "analysis" else "ERA5 reanalysis (final)"),
        "model_id": model_id,
        "mode": mode,
        "horizons": out_h,
        "n_total": sum(v["n"] for v in out_h.values()),
    }


def timeline(conn, *, window: str = "7d", source: str = "era5",
             model_id: str | None = None, horizon: int = 1,
             now: pd.Timestamp | None = None, limit: int = 500) -> list[dict]:
    """Issued median vs actual, oldest first, for the chart."""
    now = now or pd.Timestamp.now()
    rows = storage.scored_rows(conn, source, window_start(window, now), model_id=model_id)
    pts = [{
        "valid_time": r["valid_time"],
        "issue_time": r["issue_time"],
        "median": r["median"],
        "lower": r["quantiles"][0],
        "upper": r["quantiles"][-1],
        "actual": r["actual"],
        "mode": r["mode"],
    } for r in rows if int(r["horizon_h"]) == horizon]
    return pts[-limit:]
