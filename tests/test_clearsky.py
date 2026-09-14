"""Clear-sky alignment tests.

Open-Meteo's ERA5 `shortwave_radiation` is the mean of the PRECEDING hour: the row
labelled 17:00 covers 16:00-17:00 and is centred at 16:30. Evaluating pvlib
clear-sky at the label - or worse, at label+30min - misaligns the clear-sky index
by up to a full hour.

That bug was live in this repo and did real damage: median k_t ran from 0.19 at
08:00 to 1.46 at 17:00 (a clear-sky index must not trend with time of day), and
the 1.25x clear-sky clamp bound on 68.7% of t+3h targets issued at 14:00, which
collapsed the predictive interval to a single point.

The physical invariant these tests pin: on genuinely clear days, k_t must be flat
and near 1.0 across the day.
"""
import numpy as np
import pandas as pd
import pytest

pvlib = pytest.importorskip("pvlib")

ROOT_CSV = "artifacts/raw/weather_hourly.csv"


def _clearsky(cfg, index, offset_min):
    site = cfg["site"]
    loc = pvlib.location.Location(site["latitude"], site["longitude"],
                                  altitude=site.get("altitude_m", 0), tz=site["timezone"])
    idx = pd.DatetimeIndex(index).tz_localize(site["timezone"])
    return loc.get_clearsky(idx + pd.Timedelta(minutes=offset_min),
                            model="ineichen")["ghi"].to_numpy()


def test_config_uses_preceding_hour_centre(cfg):
    """The offset must be negative - ERA5 radiation is a preceding-hour mean."""
    off = cfg["clearsky"]["eval_offset_min"]
    assert off == -30, f"clear-sky offset is {off}; ERA5 needs -30 (preceding-hour centre)"


def _load(root, cfg):
    p = root / ROOT_CSV
    if not p.exists():
        pytest.skip("fetch the weather archive first")
    wx = pd.read_csv(p, parse_dates=["timestamp"])
    wx = wx[wx["timestamp"].dt.year >= 2024].reset_index(drop=True)
    if len(wx) < 2000:
        pytest.skip("not enough data")
    return wx


def test_configured_offset_beats_the_alternatives(root, cfg):
    """The configured offset must maximise correlation with solar geometry."""
    wx = _load(root, cfg)
    g = wx["ghi"].to_numpy()
    scores = {}
    for off in (-60, -30, 0, 30, 60):
        cs = _clearsky(cfg, wx["timestamp"], off)
        m = (cs > 5) | (g > 5)
        scores[off] = float(np.corrcoef(g[m], cs[m])[0, 1])
    best = max(scores, key=scores.get)
    assert best == cfg["clearsky"]["eval_offset_min"], (
        f"offset {best} correlates better ({scores[best]:.4f}) than the configured "
        f"{cfg['clearsky']['eval_offset_min']} ({scores[cfg['clearsky']['eval_offset_min']]:.4f})")


def test_kt_is_flat_and_near_one_on_clear_days(root, cfg):
    """The physical invariant: clear days give k_t ~ 1.0 at every hour."""
    wx = _load(root, cfg)
    cs = _clearsky(cfg, wx["timestamp"], cfg["clearsky"]["eval_offset_min"])
    d = pd.DataFrame({"ts": wx["timestamp"], "hour": wx["timestamp"].dt.hour,
                      "ghi": wx["ghi"].to_numpy(), "cs": cs})
    # 09:00-17:00; 08:00 is excluded because the sun is too low for a stable ratio
    d = d[d["hour"].between(9, 17) & (d["cs"] > 50)]
    d["kt"] = d["ghi"] / d["cs"]
    clearest = d.groupby(d["ts"].dt.date)["kt"].mean().nlargest(30).index
    clear = d[d["ts"].dt.date.isin(clearest)]
    med = clear.groupby("hour")["kt"].median()

    assert med.between(0.85, 1.20).all(), (
        f"k_t on clear days is not near 1.0 at every hour:\n{med.round(3).to_dict()}")
    drift = abs(med.iloc[-1] - med.iloc[0])
    assert drift < 0.30, (
        f"k_t drifts {drift:.2f} from 09:00 to 17:00 on clear days - clear-sky is "
        f"misaligned in time:\n{med.round(3).to_dict()}")


def test_kt_does_not_trend_across_the_day_overall(root, cfg):
    """Median k_t across ALL days must not climb monotonically through the day."""
    wx = _load(root, cfg)
    cs = _clearsky(cfg, wx["timestamp"], cfg["clearsky"]["eval_offset_min"])
    d = pd.DataFrame({"hour": wx["timestamp"].dt.hour, "ghi": wx["ghi"].to_numpy(), "cs": cs})
    d = d[d["hour"].between(9, 17) & (d["cs"] > 50)]
    med = (d["ghi"] / d["cs"]).groupby(d["hour"]).median()
    assert med.max() < 1.15, f"median k_t exceeds 1.15 at some hour:\n{med.round(3).to_dict()}"
    assert abs(med.iloc[-1] - med.iloc[0]) < 0.35, (
        f"median k_t drifts across the day:\n{med.round(3).to_dict()}")


def test_the_old_plus_30_offset_would_fail(root, cfg):
    """Guard the specific regression: +30 min must be detectably wrong."""
    wx = _load(root, cfg)
    cs = _clearsky(cfg, wx["timestamp"], +30)
    d = pd.DataFrame({"hour": wx["timestamp"].dt.hour, "ghi": wx["ghi"].to_numpy(), "cs": cs})
    d = d[d["hour"].between(9, 17) & (d["cs"] > 50)]
    med = (d["ghi"] / d["cs"]).groupby(d["hour"]).median()
    assert med.iloc[-1] > 1.15, (
        "the known-bad +30min offset no longer produces its signature late-day "
        "k_t > 1.15; this test needs revisiting")


def test_train_serve_parity(cfg):
    """Serving clear-sky must equal training clear-sky, exactly.

    Regression guard for a real bug: predict.py::clearsky_series took a single
    point at eval_offset_min while data/fetch_weather.py integrated over the
    preceding hour. Live k_t was therefore inflated near sunrise, and once the
    served model emits k_t scaled by clear-sky the error enters twice.
    """
    import predict
    from data.clearsky import hour_mean_ghi

    idx = pd.date_range("2024-06-15 08:00", "2024-06-15 17:00", freq="h")
    serve = predict.clearsky_series(cfg, idx).to_numpy()
    train = hour_mean_ghi(cfg, idx)
    assert np.allclose(serve, train), "serving clear-sky has drifted from training"


def test_hour_mean_exceeds_point_estimate_at_sunrise(cfg):
    """Why the integration matters: the 08:00 hour is where they disagree most."""
    from data.clearsky import hour_mean_ghi

    idx = pd.date_range("2024-06-15 08:00", "2024-06-15 17:00", freq="h")
    integrated = hour_mean_ghi(cfg, idx)
    point = _clearsky(cfg, idx, cfg["clearsky"]["eval_offset_min"])
    rel = (integrated - point) / np.maximum(point, 1.0)
    assert rel[0] > rel[-1], (
        "the point estimate should understate the sunrise hour more than the "
        f"afternoon; got {rel[0]:.3f} at 08:00 vs {rel[-1]:.3f} at 17:00")
