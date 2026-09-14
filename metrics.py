"""Forecast metrics: point, probabilistic, and stratified calibration.

DISPERSION IS MEASURED IN k_t SPACE, NOT W/m^2
----------------------------------------------
The raw spread/reality ratio in W/m^2 is confounded by clear-sky magnitude
changing through the day, which masks exactly the failure it is meant to expose.
Everything below converts to clear-sky index before computing dispersion.

`spread/reality` near 100% is the goal. ~50% means the model is hedging to the
mean - the prior project measured 52-63% on cloudy rows while aggregate coverage
looked fine (91.6% overall vs 81.4% on cloudy days).

A NOTE ON THE NOMINAL COVERAGE LEVEL
------------------------------------
With QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9], the central band [q0.10, q0.90] is
an **80%** prediction interval, not a 90% one. Scoring it against a 90% target
would make a perfectly calibrated model look broken by ~10 points. The nominal
level is therefore DERIVED from the quantile levels (hi_q - lo_q) everywhere
below, and reported next to the observed coverage under that name. To obtain a
true 90% interval, add 0.05 and 0.95 to `quantiles` in config.yaml.
"""
from __future__ import annotations

import numpy as np


def _safe_kt(ghi: np.ndarray, cs: np.ndarray) -> np.ndarray:
    cs = np.where(cs > 5.0, cs, np.nan)
    return ghi / cs


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def bias(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred - y))


def pinball(pred_q: np.ndarray, y: np.ndarray, quantiles) -> float:
    """pred_q (N, Q), y (N,). Mean pinball loss across quantiles."""
    qs = np.asarray(quantiles)
    e = y[:, None] - pred_q
    return float(np.mean(np.maximum(qs * e, (qs - 1.0) * e)))


def crps_from_quantiles(pred_q: np.ndarray, y: np.ndarray, quantiles) -> float:
    """CRPS approximated from a quantile set.

    For a finite quantile set the pinball loss averaged over quantiles is a
    consistent discrete approximation of CRPS (up to a factor of 2 depending on
    convention); we use the standard 2 * mean-pinball form so the number is
    comparable to CRPS reported elsewhere.
    """
    return 2.0 * pinball(pred_q, y, quantiles)


def coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def skill_score(model_err: float, ref_err: float) -> float:
    """Fraction of the reference model's error removed. 1 = perfect, 0 = no gain."""
    if ref_err <= 0:
        return float("nan")
    return float(1.0 - model_err / ref_err)


def dispersion_kt(pred_q: np.ndarray, y: np.ndarray, cs: np.ndarray,
                  quantiles, lo_q: float | None = None, hi_q: float | None = None) -> dict:
    """spread/reality in CLEAR-SKY INDEX space.

    Predicted sigma is inferred from the central interval assuming approximate
    normality (z = 1.2816 for the 10-90 band), which makes it comparable with the
    observed standard deviation of the outcome.
    """
    qs = list(quantiles)
    lo_q = min(qs) if lo_q is None else lo_q
    hi_q = max(qs) if hi_q is None else hi_q
    i_lo, i_hi = qs.index(lo_q), qs.index(hi_q)
    # width of the central band in sigma, for the ACTUAL quantile levels used
    from scipy.stats import norm
    z = float(norm.ppf(hi_q) - norm.ppf(lo_q))
    cs_safe = np.where(cs > 5.0, cs, np.nan)
    width_kt = (pred_q[:, i_hi] - pred_q[:, i_lo]) / cs_safe
    pred_sigma = np.nanmean(width_kt) / z
    actual_sigma = float(np.nanstd(_safe_kt(y, cs)))
    ratio = float(pred_sigma / actual_sigma) if actual_sigma > 1e-9 else float("nan")
    return {"pred_sigma_kt": float(pred_sigma), "actual_sigma_kt": actual_sigma,
            "spread_over_reality": ratio}


def nominal_level(quantiles, lo_q: float | None = None, hi_q: float | None = None) -> float:
    """Nominal coverage of the central band, derived from the quantile levels."""
    qs = sorted(quantiles)
    lo = qs[0] if lo_q is None else lo_q
    hi = qs[-1] if hi_q is None else hi_q
    return float(hi - lo)


def band_indices(quantiles) -> tuple[int, int, float]:
    """(index of lowest q, index of highest q, nominal level of that band)."""
    qs = list(quantiles)
    lo_q, hi_q = min(qs), max(qs)
    return qs.index(lo_q), qs.index(hi_q), float(hi_q - lo_q)


def stratified_report(pred_q: np.ndarray, y: np.ndarray, cs: np.ndarray,
                      kt_now: np.ndarray, quantiles, cloud_bins,
                      target: float | None = None) -> list[dict]:
    """Coverage, dispersion and bias split by CURRENT cloud regime.

    Aggregate coverage hid a failure confined to cloudy conditions in the prior
    project; this split is what surfaces it.
    """
    qs = list(quantiles)
    i_lo, i_hi, nominal = band_indices(qs)
    if target is None:
        target = nominal
    i_med = qs.index(0.5)
    rows = []
    for lo, hi, label in cloud_bins:
        m = (kt_now >= lo) & (kt_now < hi)
        n = int(m.sum())
        if n == 0:
            rows.append({"regime": label, "n": 0})
            continue
        d = dispersion_kt(pred_q[m], y[m], cs[m], quantiles)
        rows.append({
            "regime": label,
            "n": n,
            "coverage": coverage(pred_q[m, i_lo], pred_q[m, i_hi], y[m]),
            "coverage_target": target,
            "nominal_level": nominal,
            "spread_over_reality": d["spread_over_reality"],
            "pred_sigma_kt": d["pred_sigma_kt"],
            "actual_sigma_kt": d["actual_sigma_kt"],
            "mae": mae(pred_q[m, i_med], y[m]),
            "bias": bias(pred_q[m, i_med], y[m]),
            "interval_width": interval_width(pred_q[m, i_lo], pred_q[m, i_hi]),
        })
    return rows


def full_report(pred_q: np.ndarray, y: np.ndarray, cs: np.ndarray,
                kt_now: np.ndarray, quantiles, cloud_bins,
                ref_mae: float | None = None) -> dict:
    """Everything for one horizon. pred_q (N, Q)."""
    qs = list(quantiles)
    med = pred_q[:, qs.index(0.5)]
    i_lo, i_hi, nominal = band_indices(qs)
    out = {
        "n": int(len(y)),
        "nominal_level": nominal,
        "mae": mae(med, y),
        "rmse": rmse(med, y),
        "bias": bias(med, y),
        "pinball": pinball(pred_q, y, qs),
        "crps": crps_from_quantiles(pred_q, y, qs),
        "coverage": coverage(pred_q[:, i_lo], pred_q[:, i_hi], y),
        "interval_width": interval_width(pred_q[:, i_lo], pred_q[:, i_hi]),
        "stratified": stratified_report(pred_q, y, cs, kt_now, qs, cloud_bins),
    }
    out.update({f"dispersion_{k}": v for k, v in
                dispersion_kt(pred_q, y, cs, qs).items()})
    if ref_mae is not None:
        out["skill_vs_ref"] = skill_score(out["mae"], ref_mae)
    return out
