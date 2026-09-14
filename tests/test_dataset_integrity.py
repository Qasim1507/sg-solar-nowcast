"""Dataset-level constraints: night targets, split integrity, stats provenance."""
import json

import numpy as np
import pandas as pd
import pytest

from data.build_dataset import build_targets_and_lags


def test_night_targets_are_excluded():
    """A 17:00 issue time must not get a ~0 W/m2 target at 20:00.

    Targets are built on the FULL hourly frame so the time-reindex is correct,
    which means a nocturnal target is a real number, not NaN. Near-zero targets
    are trivially easy and inflate every metric, so the target HOUR must be
    filtered explicitly rather than left to whatever the grid join drops.
    """
    lo, hi, horizons = 8, 17, [1, 2, 3]
    rows = []
    for d in range(4):
        for h in range(0, 24):                     # full day incl. night
            ts = pd.Timestamp("2024-03-01") + pd.Timedelta(days=d, hours=h)
            ghi = 0.0 if (h < lo or h > hi) else 500.0
            rows.append({"timestamp": ts, "ghi": ghi, "ghi_clearsky": 900.0,
                         "clearsky_ratio": 0.55})
    wx = build_targets_and_lags(pd.DataFrame(rows), horizons)

    day = wx[wx["timestamp"].dt.hour.between(lo, hi)].copy()
    # without the daylight-target filter, night targets survive as real zeros
    assert (day["ghi_t3h"] == 0.0).any(), "fixture should contain night targets"

    keep = pd.Series(True, index=day.index)
    for h in horizons:
        keep &= (day["timestamp"] + pd.Timedelta(hours=h)).dt.hour.between(lo, hi)
    filtered = day[keep]
    assert not (filtered["ghi_t3h"] == 0.0).any(), "night target survived the filter"
    assert (filtered["timestamp"].dt.hour <= hi - max(horizons)).all()
    assert len(filtered) == 4 * (hi - max(horizons) - lo + 1)


def test_split_is_chronological_and_disjoint(root, cfg):
    p = root / "artifacts/processed/dataset_B.parquet"
    if not p.exists():
        pytest.skip("build the dataset first")
    df = pd.read_parquet(p)
    tr = df[df["split"] == "train"]["timestamp"]
    va = df[df["split"] == "val"]["timestamp"]
    te = df[df["split"] == "test"]["timestamp"]
    assert tr.max() <= va.min(), "train overlaps val in time"
    assert va.max() <= te.min(), "val overlaps test in time"
    assert set(tr).isdisjoint(set(te)), "train and test share timestamps"
    assert df["timestamp"].is_unique


def test_train_stats_come_from_training_split_only(root):
    ds = root / "artifacts/processed/dataset_B.parquet"
    st = root / "artifacts/processed/train_stats_B.json"
    if not (ds.exists() and st.exists()):
        pytest.skip("build the dataset first")
    df = pd.read_parquet(ds)
    stats = json.loads(st.read_text())
    tr = df[df["split"] == "train"]
    for f in stats["features"][:4]:
        assert np.isclose(stats["mean"][f], tr[f].mean(), rtol=1e-6), (
            f"{f} mean does not match the training split - stats leaked from val/test")
    assert stats["n_train"] == len(tr)


def test_no_target_is_nan(root, cfg):
    p = root / "artifacts/processed/dataset_B.parquet"
    if not p.exists():
        pytest.skip("build the dataset first")
    df = pd.read_parquet(p)
    for h in cfg["horizons"]:
        assert df[f"ghi_t{h}h"].notna().all()
        assert df[f"cs_t{h}h"].notna().all()


def test_every_row_has_a_grid_reference(root):
    p = root / "artifacts/processed/dataset_B.parquet"
    if not p.exists():
        pytest.skip("build the dataset first")
    df = pd.read_parquet(p)
    assert df["path"].notna().all(), "rows without a rain grid survived into the splits"
    assert df["index"].notna().all()


def test_test_split_meets_floor_on_full_archive(root, cfg):
    """The assertion that would have caught the 18-sample test collapse."""
    p = root / "artifacts/processed/dataset_A.parquet"
    if not p.exists():
        pytest.skip("track A not built yet (needs the full archive)")
    df = pd.read_parquet(p)
    n_test = int((df["split"] == "test").sum())
    assert n_test > cfg["dataset"]["min_test_samples"], (
        f"Test set collapsed to {n_test} samples")
