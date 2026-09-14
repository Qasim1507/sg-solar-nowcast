"""Target-alignment and lag tests.

These are the tests that must fail loudly if someone reintroduces
`df["ghi"].shift(-h)` on a daylight-filtered frame.

Measured contamination in the prior project from exactly that bug:
10.0% / 20.0% / 30.0% of t+1/2/3h targets, worst at the most important horizon,
plus ghi_lag1 stale by up to 15 hours on 10% of rows.
"""
import numpy as np
import pandas as pd
import pytest

from data.build_dataset import assert_alignment, build_targets_and_lags


def reference_series(df):
    """Full hourly GHI indexed by timestamp - what the real pipeline passes in."""
    return df.set_index("timestamp")["ghi"]


def make_daylight_frame(days=6, lo=8, hi=17):
    """Hourly daylight-only rows across several days - i.e. WITH overnight gaps."""
    rows = []
    for d in range(days):
        for h in range(lo, hi + 1):
            ts = pd.Timestamp("2024-03-01") + pd.Timedelta(days=d, hours=h)
            rows.append({
                "timestamp": ts,
                # value encodes absolute hour so any mis-pairing is arithmetic
                "ghi": float(d * 100 + h),
                "ghi_clearsky": 1000.0,
                "clearsky_ratio": 0.5,
            })
    return pd.DataFrame(rows)


def test_time_indexed_targets_are_exactly_h_hours_ahead():
    df = make_daylight_frame()
    out = build_targets_and_lags(df, [1, 2, 3])
    for h in (1, 2, 3):
        col = out[f"ghi_t{h}h"]
        valid = out[col.notna()]
        # value encodes d*100+h, so a correct t+h target differs by exactly h
        diff = valid[f"ghi_t{h}h"] - valid["ghi"]
        assert (diff == h).all(), f"t+{h}h target is not {h} hours ahead"


def test_overnight_gap_becomes_nan_not_tomorrow_morning():
    """16:00 must NOT be paired with the next morning's 08:00."""
    df = make_daylight_frame()
    out = build_targets_and_lags(df, [1, 2, 3])
    late = out[out["timestamp"].dt.hour == 17]
    assert late["ghi_t1h"].isna().all(), "17:00 got a t+1h target across the night"
    for h in (1, 2, 3):
        end = out[out["timestamp"].dt.hour > 17 - h]
        assert end[f"ghi_t{h}h"].isna().all(), f"t+{h}h leaked across the overnight gap"


def test_positional_shift_would_be_wrong_and_is_detected():
    """Demonstrate the bug, and prove assert_alignment() catches it."""
    df = make_daylight_frame()
    good = build_targets_and_lags(df, [1, 2, 3])

    bad = good.copy()
    bad["ghi_t3h"] = bad["ghi"].shift(-3)          # the prior project's bug

    contaminated = (bad["ghi_t3h"] - bad["ghi"] != 3) & bad["ghi_t3h"].notna()
    frac = contaminated.mean()
    assert frac > 0.15, (
        f"positional shift should contaminate ~30% of t+3h rows, saw {frac:.1%}")

    ref = reference_series(good)
    with pytest.raises(AssertionError):
        assert_alignment(bad.dropna(subset=["ghi_t1h", "ghi_t2h", "ghi_t3h"]),
                         [1, 2, 3], reference=ref)


def test_lags_are_time_indexed():
    df = make_daylight_frame()
    out = build_targets_and_lags(df, [1, 2, 3])
    valid = out[out["ghi_lag1"].notna()]
    assert (valid["ghi"] - valid["ghi_lag1"] == 1).all(), "ghi_lag1 is not one hour back"
    # first daylight hour of each day has no in-frame predecessor
    first = out[out["timestamp"].dt.hour == 8]
    assert first["ghi_lag1"].isna().all(), "08:00 got a lag1 from yesterday evening"


def test_stale_lag_is_detected():
    df = make_daylight_frame()
    good = build_targets_and_lags(df, [1, 2, 3])
    bad = good.copy()
    bad["ghi_lag1"] = bad["ghi"].shift(1)          # positional -> 15h stale at day breaks
    with pytest.raises(AssertionError):
        assert_alignment(bad.dropna(subset=["ghi_t1h", "ghi_t2h", "ghi_t3h", "ghi_lag1"]),
                         [1, 2, 3], reference=reference_series(good))


def test_assert_alignment_accepts_correct_frame():
    df = make_daylight_frame()
    out = build_targets_and_lags(df, [1, 2, 3])
    ref = reference_series(out)
    kept = out.dropna(subset=["ghi_t1h", "ghi_t2h", "ghi_t3h", "ghi_lag1"]).reset_index(drop=True)
    assert_alignment(kept, [1, 2, 3], reference=ref)
    assert len(kept) > 0


def test_dst_free_tz_naive_timestamps_are_unique():
    df = make_daylight_frame()
    assert df["timestamp"].is_unique
    assert df["timestamp"].is_monotonic_increasing


def test_verifier_refuses_vacuous_pass():
    """A frame it cannot actually check must raise, not quietly succeed.

    An earlier version compared only where both sides were non-NaN, which skipped
    exactly the day-boundary rows where positional shifts go wrong.
    """
    df = make_daylight_frame()
    out = build_targets_and_lags(df, [1, 2, 3])
    kept = out.dropna(subset=["ghi_t1h", "ghi_t2h", "ghi_t3h", "ghi_lag1"]).reset_index(drop=True)
    empty_ref = pd.Series(dtype=float, index=pd.DatetimeIndex([], name="timestamp"))
    with pytest.raises(AssertionError):
        assert_alignment(kept, [1, 2, 3], reference=empty_ref)
