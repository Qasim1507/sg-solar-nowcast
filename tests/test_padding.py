"""Lookback padding tests.

Zero-padding BEFORE normalisation sent inputs in at -12.6 sigma (temperature) and
-6.7 sigma (humidity) in the prior project. Padding must use the training mean,
which normalises to exactly 0, and a window that is mostly synthetic must raise
rather than be served.
"""
import numpy as np
import pandas as pd
import pytest

from data.dataset import NowcastDataset, PaddingError, normalise


def build_ds(cfg, n_hours=6, strict=True, tmp_path=None):
    """A short daylight run so early rows have incomplete lookbacks."""
    feats = cfg["temporal"]["features"]
    rows = []
    for k in range(n_hours):
        ts = pd.Timestamp("2024-03-01 08:00") + pd.Timedelta(hours=k)
        r = {"timestamp": ts, "path": "none.npy", "index": 0,
             "clearsky_ratio": 0.6, "cloud_cover": 50.0,
             "motion_vx": 0.0, "motion_vy": 0.0, "motion_valid": 0,
             "rain_mean": 0.0, "rain_max": 0.0, "wet_frac": 0.0, "rain_std": 0.0,
             "mask_frac": 0.9}
        for f in feats:
            r.setdefault(f, float(k + 1))
        for h in cfg["horizons"]:
            r[f"ghi_t{h}h"] = 500.0
            r[f"cs_t{h}h"] = 900.0
        rows.append(r)
    df = pd.DataFrame(rows)

    gs = ["rain_mean", "rain_max", "wet_frac", "rain_std", "mask_frac",
          "motion_vx", "motion_vy", "motion_valid"]
    stats = {
        "features": feats,
        "mean": {f: float(df[f].mean()) for f in feats},
        "std": {f: float(max(df[f].std(), 1e-6)) for f in feats},
        "grid_scalars": gs,
        "grid_mean": {c: float(df[c].mean()) for c in gs},
        "grid_std": {c: float(max(df[c].std(), 1e-6)) for c in gs},
        "nwp_cols": [],
    }
    return NowcastDataset(df, stats, cfg, tmp_path or ".", variant="temporal",
                          strict_padding=strict)


def test_padding_normalises_to_exactly_zero(cfg, tmp_path):
    ds = build_ds(cfg, strict=False, tmp_path=tmp_path)
    win, n_real = ds.build_window(0)          # only 1 real step, rest padded
    assert n_real == 1
    pad = win[:-1]
    assert np.allclose(pad, 0.0), (
        f"padding is not 0 in normalised space (max |v|={np.abs(pad).max():.3f})")


def test_zero_padding_would_be_far_off_distribution(cfg, tmp_path):
    """Show what the prior bug did: raw zeros land many sigma from the mean."""
    ds = build_ds(cfg, strict=False, tmp_path=tmp_path)
    f = cfg["temporal"]["features"][2]        # temperature_2m
    mean, std = ds.stats["mean"][f], ds.stats["std"][f]
    sigma_if_zero = abs((0.0 - mean) / std)
    assert sigma_if_zero > 1.0, "fixture should make raw-zero padding clearly off-distribution"
    win, _ = ds.build_window(0)
    assert np.abs(win[:-1]).max() < 1e-6, "mean-padding should sit at 0 sigma"


def test_raises_when_window_is_mostly_synthetic(cfg, tmp_path):
    ds = build_ds(cfg, strict=True, tmp_path=tmp_path)
    # lookback is 24; a 6-row frame can never fill half of it
    with pytest.raises(PaddingError):
        ds.build_window(0)


def test_does_not_raise_when_enough_real_data(cfg, tmp_path):
    need = int(np.ceil(cfg["temporal"]["lookback"] * cfg["dataset"]["min_real_fraction"]))
    ds = build_ds(cfg, n_hours=need + 2, strict=True, tmp_path=tmp_path)
    win, n_real = ds.build_window(need + 1)
    assert n_real >= need
    assert win.shape == (cfg["temporal"]["lookback"], len(cfg["temporal"]["features"]))


def test_window_is_time_ordered_and_ends_at_t(cfg, tmp_path):
    ds = build_ds(cfg, n_hours=20, strict=False, tmp_path=tmp_path)
    i = 19
    win, n_real = ds.build_window(i)
    # feature values were set to k+1, so the last real step must equal row i's value
    f_idx = 2
    expected = ds.X[i, f_idx]
    assert np.isclose(win[-1, f_idx], expected), "window does not end at time t"


def test_window_skips_a_gap_without_inventing_data(cfg, tmp_path):
    """A missing hour must be skipped, not filled with a synthetic value.

    Semantics changed when the lookback moved to the daylight observation series:
    one step back is now the previous available DAYLIGHT observation, so a hole in
    the record shortens the wall-clock span rather than truncating the window. What
    must still hold is that no fabricated timestamp enters, the sequence stays
    ordered, and nothing after t is admitted.
    """
    ds = build_ds(cfg, n_hours=20, strict=False, tmp_path=tmp_path)
    gap = pd.Timestamp("2024-03-01 15:00")
    keep = ds.df[ds.df["timestamp"] != gap].reset_index(drop=True)
    ds.df = keep
    ds.history = keep
    ds.X = normalise(keep, ds.feats, ds.stats["mean"], ds.stats["std"])
    ds.ts_to_row = {t: i for i, t in enumerate(keep["timestamp"])}
    ds.ts_to_pos = ds.ts_to_row

    i = len(keep) - 1
    t0 = keep["timestamp"].iloc[i]
    _, n_real = ds.build_window(i)
    assert n_real == min(len(keep), ds.lookback)

    pos = ds.ts_to_pos[t0]
    span = ds.history["timestamp"].iloc[max(0, pos - ds.lookback + 1):pos + 1]
    assert gap not in set(span), "the missing hour was invented"
    assert span.is_monotonic_increasing
    assert (span <= t0).all()


# --------------------------------------------------------------- real dataset
def test_real_lookback_windows_are_actually_full(root, cfg):
    """The gap the other padding tests missed: assert on the REAL dataset.

    Those tests exercise build_window against a synthetic frame, so they passed
    while every real training window was 83% mean-padding - the sample frame holds
    only issue times (08:00-14:00), so walking back hour-by-hour through it capped
    the lookback at 7 steps and averaged 4 of 24. The lookback must come from the
    full daylight observation series instead.
    """
    import numpy as np

    from data.dataset import make_datasets

    if not (root / "artifacts/processed/dataset_A.parquet").exists():
        pytest.skip("build the dataset first")
    L = cfg["temporal"]["lookback"]
    ds = make_datasets(cfg, "A", variant="temporal")
    for split in ("train", "test"):
        d = ds[split]
        n = np.array([d.build_window(i)[1] for i in range(0, len(d), 31)])
        frac = n.mean() / L
        assert frac > 0.95, (
            f"{split} lookback windows are only {frac:.1%} real data; the BiLSTM is "
            f"mostly reading padding")
        assert (n / L >= cfg["dataset"]["min_real_fraction"]).mean() > 0.99


def test_lookback_never_reaches_past_the_issue_time(root, cfg):
    """Stepping backwards must not admit any observation after t."""
    from data.dataset import make_datasets

    if not (root / "artifacts/processed/dataset_A.parquet").exists():
        pytest.skip("build the dataset first")
    L = cfg["temporal"]["lookback"]
    ds = make_datasets(cfg, "A", variant="temporal")["test"]
    for i in range(0, len(ds), 97):
        t0 = ds.df["timestamp"].iloc[i]
        pos = ds.ts_to_pos.get(t0)
        if pos is None:
            continue
        span = ds.history["timestamp"].iloc[max(0, pos - L + 1):pos + 1]
        assert (span <= t0).all(), f"window for {t0} contains a future observation"
        assert span.is_monotonic_increasing


def test_lookback_spans_multiple_days(root, cfg):
    """24 DAYLIGHT steps must cross the overnight gap, not stop at one day."""
    from data.dataset import make_datasets

    if not (root / "artifacts/processed/dataset_A.parquet").exists():
        pytest.skip("build the dataset first")
    L = cfg["temporal"]["lookback"]
    ds = make_datasets(cfg, "A", variant="temporal")["test"]
    i = len(ds) // 2
    pos = ds.ts_to_pos[ds.df["timestamp"].iloc[i]]
    span = ds.history["timestamp"].iloc[max(0, pos - L + 1):pos + 1]
    assert span.dt.date.nunique() >= 2, "lookback did not cross the overnight gap"
