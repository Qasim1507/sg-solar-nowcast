"""NWP inputs must not be a copy of the target.

Verified 2026-09-08: the Historical Forecast API's default (`best_match`) returns
values BIT-IDENTICAL to the ERA5 archive target at this site, on 2024-01-15,
2024-06-15, 2025-03-10 and 2026-08-20. Training on that would hand the model its
own answer and produce a spectacular but entirely fake skill score.

These tests are the regression guard. The offline ones always run; the networked
one is opt-in via RUN_NETWORK_TESTS=1.
"""
import os

import numpy as np
import pandas as pd
import pytest

from data.fetch_nwp import _require_explicit_model

NETWORK = os.environ.get("RUN_NETWORK_TESTS") == "1"


def test_best_match_is_refused(cfg):
    for bad in ("best_match", "BEST_MATCH", "", None):
        with pytest.raises(ValueError, match="best_match|Refusing"):
            _require_explicit_model(bad, cfg)


def test_explicit_models_are_accepted(cfg):
    for good in ("gfs_seamless", "icon_seamless", "ecmwf_ifs025"):
        _require_explicit_model(good, cfg)


def test_config_pins_explicit_models(cfg):
    assert cfg["nwp"]["forbid_best_match"] is True
    assert cfg["nwp"]["models"], "no NWP model pinned"
    assert "best_match" not in [m.lower() for m in cfg["nwp"]["models"]]


def test_identical_series_is_detected():
    """The detector itself: an identical column must be flagged."""
    a = np.array([30.0, 106.0, 238.0, 417.0, 570.0])
    assert np.array_equal(a, a.copy()), "sanity"
    b = a.copy()
    b[2] += 1.0
    assert not np.array_equal(a, b)


@pytest.mark.skipif(not NETWORK, reason="set RUN_NETWORK_TESTS=1 to hit the API")
def test_live_nwp_is_not_the_target(cfg):
    from data._client import ApiClient
    from data.fetch_nwp import check_leakage
    client = ApiClient(delay_s=1.0, backoff_s=8.0)
    for model in cfg["nwp"]["models"]:
        rep = check_leakage(client, cfg, model, probe_days=["2025-03-10"])
        for probe in rep["probes"]:
            assert probe.get("identical") is False
            assert probe.get("mae_wm2", 0) > 1.0, "NWP is suspiciously close to the target"


def test_fetched_nwp_differs_from_target(root):
    """If both files exist, the joined NWP column must not equal the target."""
    wx_p = root / "artifacts/raw/weather_hourly.csv"
    nwp_p = root / "artifacts/raw/nwp_hourly.csv"
    if not (wx_p.exists() and nwp_p.exists()):
        pytest.skip("fetch weather and nwp first")
    wx = pd.read_csv(wx_p, parse_dates=["timestamp"])[["timestamp", "ghi"]]
    nwp = pd.read_csv(nwp_p, parse_dates=["timestamp"])
    swrad = [c for c in nwp.columns if c.endswith("shortwave_radiation")]
    assert swrad, "no NWP shortwave_radiation column"
    m = wx.merge(nwp, on="timestamp", how="inner").dropna(subset=["ghi"] + swrad)
    assert len(m) > 500, f"only {len(m)} overlapping rows"
    for c in swrad:
        assert not np.array_equal(m["ghi"].to_numpy(), m[c].to_numpy()), (
            f"{c} is bit-identical to the ERA5 target - this is best_match leakage")
        mae = float(np.mean(np.abs(m["ghi"] - m[c])))
        assert mae > 1.0, f"{c} MAE vs target is {mae:.3f} W/m2 - implausibly low"
