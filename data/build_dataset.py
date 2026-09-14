"""Time-indexed join -> daylight filter -> chronological splits -> train_stats.json.

THE ALIGNMENT BUG THIS FILE EXISTS TO PREVENT
---------------------------------------------
The frame is daylight-filtered, so consecutive ROWS are not consecutive HOURS:
row 16:00 is followed by row 08:00 the next morning. `df["ghi"].shift(-h)` therefore
pairs 16:00 with tomorrow-08:00. Measured contamination in the prior project:
10.0% / 20.0% / 30.0% of t+1/2/3h targets - worst at the horizon that matters most.
The same bug staled `ghi_lag1` on 10% of rows by up to 15 hours.

Every target and lag here is built by reindexing a TIMESTAMP index by an explicit
`timestamp +/- Timedelta(hours=h)`. `reindex` yields NaN across the overnight gap;
those rows are dropped. `assert_alignment()` re-derives the offsets from the
timestamps themselves and raises if any pair is not exactly h hours apart.

OTHER CONSTRAINTS ENFORCED HERE
  * Rows lacking a valid rain grid are dropped BEFORE the split, not after. In the
    prior project the unusable tail landed entirely in test and collapsed it to 18
    samples, silently invalidating every deep-model result.
  * train_stats.json is computed from the TRAINING SPLIT ONLY.
  * Test split size is asserted > `dataset.min_test_samples`.

Usage:
    python data/build_dataset.py [--track A|B|both]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data._client import load_config, log, resolve_path, setup_logging  # noqa: E402

GRID_SCALARS = ["rain_mean", "rain_max", "wet_frac", "rain_std", "mask_frac",
                "motion_vx", "motion_vy", "motion_valid"]


def build_targets_and_lags(df: pd.DataFrame, horizons: list[int],
                           lags: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Time-indexed targets and lags. Never positional shift."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df.set_index("timestamp")

    for h in horizons:
        f = ts[["ghi", "ghi_clearsky"]].reindex(df["timestamp"] + pd.Timedelta(hours=h))
        df[f"ghi_t{h}h"] = f["ghi"].to_numpy()
        df[f"cs_t{h}h"] = f["ghi_clearsky"].to_numpy()

    for L in lags:
        df[f"ghi_lag{L}"] = ts["ghi"].reindex(df["timestamp"] - pd.Timedelta(hours=L)).to_numpy()
        df[f"kt_lag{L}"] = ts["clearsky_ratio"].reindex(
            df["timestamp"] - pd.Timedelta(hours=L)).to_numpy()
    return df


def assert_alignment(df: pd.DataFrame, horizons: list[int],
                     reference: pd.Series | None = None,
                     min_verified: float = 0.5) -> None:
    """Prove every target/lag is an exact TIME offset, not a positional one.

    `reference` is the full hourly GHI series (all hours, including night). It is
    what makes verification possible: the rows where a positional shift actually
    goes wrong are the day-boundary rows, and those are precisely the rows whose
    target hour is missing from a daylight-filtered frame. An earlier version of
    this function compared only where both sides were non-NaN, which silently
    skipped exactly those rows and passed a frame built with `shift()`.

    Two rules are enforced:
      1. where the reference has t+h, the stored target must equal it;
      2. a NON-NaN target whose hour is ABSENT from the reference is an error -
         it can only have come from positional slicing across a gap.
    A minimum verified fraction guards against vacuous passes.
    """
    ref = reference if reference is not None else df.set_index("timestamp")["ghi"]
    ref = ref[~ref.index.duplicated()]
    ref_index = set(ref.index)

    def _check(offset: pd.Timedelta, stored: pd.Series, label: str) -> None:
        want = df["timestamp"] + offset
        present = want.isin(ref_index)
        stored_ok = stored.notna()

        orphan = stored_ok & ~present
        if orphan.any():
            bad = df.loc[orphan, "timestamp"].head(3).tolist()
            raise AssertionError(
                f"{label}: {int(orphan.sum())} rows carry a value whose hour is absent "
                f"from the reference (positional shift across a gap). e.g. {bad}")

        verify = present & stored_ok
        if verify.sum() == 0:
            raise AssertionError(f"{label}: nothing verifiable - refusing a vacuous pass")
        got = ref.reindex(want[verify]).to_numpy(dtype=float)
        have = stored[verify].to_numpy(dtype=float)
        if not np.allclose(got, have, rtol=0, atol=1e-6, equal_nan=True):
            n_bad = int((~np.isclose(got, have, rtol=0, atol=1e-6, equal_nan=True)).sum())
            raise AssertionError(
                f"{label}: {n_bad}/{len(have)} values do not match the value at that time")
        frac = verify.sum() / max(stored_ok.sum(), 1)
        if frac < min_verified:
            raise AssertionError(
                f"{label}: only {frac:.0%} of non-null values could be verified "
                f"(need >= {min_verified:.0%})")

    for h in horizons:
        _check(pd.Timedelta(hours=h), df[f"ghi_t{h}h"], f"t+{h}h target")
    if "ghi_lag1" in df.columns:
        _check(pd.Timedelta(hours=-1), df["ghi_lag1"], "ghi_lag1")


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    month = df["timestamp"].dt.month
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * month / 12.0)
    return df


def chronological_split(df: pd.DataFrame, ratios: list[float]) -> pd.DataFrame:
    n = len(df)
    i_tr = int(n * ratios[0])
    i_va = int(n * (ratios[0] + ratios[1]))
    split = np.array(["test"] * n, dtype=object)
    split[:i_tr] = "train"
    split[i_tr:i_va] = "val"
    df = df.copy()
    df["split"] = split
    return df


def build_track(cfg: dict, track: str) -> pd.DataFrame | None:
    raw = resolve_path(cfg, "raw")
    grids = resolve_path(cfg, "grids")
    proc = resolve_path(cfg, "processed")
    tcfg = cfg["dataset"]["tracks"][track]
    horizons = cfg["horizons"]

    wx = pd.read_csv(raw / "weather_hourly.csv", parse_dates=["timestamp"])
    wx = wx.sort_values("timestamp").reset_index(drop=True)

    # ---- targets/lags on the FULL hourly frame (before daylight filtering), so
    # the overnight gap is expressed as NaN rather than silently bridged.
    wx = build_targets_and_lags(wx, horizons)
    wx = add_calendar(wx)

    # ---- daylight filter
    lo, hi = cfg["daylight"]["start_hour"], cfg["daylight"]["end_hour"]
    df = wx[wx["timestamp"].dt.hour.between(lo, hi)].copy()
    n_daylight = len(df)

    # Every horizon's target must itself be a real DAYLIGHT hour.
    #
    # Targets are built on the full hourly frame (so the reindex is correct), which
    # means t=17:00 gets a real - but nocturnal - t+3h value of ~0 W/m2 rather than
    # NaN. Training on those would reintroduce exactly the night rows the daylight
    # filter exists to remove, and they inflate every metric because near-zero
    # targets are trivially easy. So the target HOUR is filtered explicitly here,
    # not left to whatever the grid join happens to drop.
    need = [f"ghi_t{h}h" for h in horizons] + [f"cs_t{h}h" for h in horizons] + ["ghi_lag1"]
    in_daylight = pd.Series(True, index=df.index)
    for h in horizons:
        th = (df["timestamp"] + pd.Timedelta(hours=h)).dt
        in_daylight &= th.hour.between(lo, hi)
    n_before = len(df)
    df = df[in_daylight].dropna(subset=need).reset_index(drop=True)
    n_after_targets = len(df)
    log.info("  dropped %d rows whose t+%dh target fell outside daylight",
             n_before - int(in_daylight.sum()), max(horizons))

    assert_alignment(df, horizons, reference=wx.set_index('timestamp')['ghi'])

    # ---- grid join (rows without a usable rain grid are dropped BEFORE splitting)
    man_path = grids / "manifest.csv"
    if not man_path.exists():
        log.error("no grid manifest; run data/build_grids.py first")
        return None
    man = pd.read_csv(man_path, parse_dates=["timestamp"])
    man = man.drop_duplicates("timestamp")
    df = df.merge(man[["timestamp", "path", "index", "n_stations_reporting"] + GRID_SCALARS],
                  on="timestamp", how="inner")
    n_after_grid = len(df)

    # ---- NWP join
    nwp_path = raw / "nwp_hourly.csv"
    nwp_cols: list[str] = []
    if nwp_path.exists():
        nwp = pd.read_csv(nwp_path, parse_dates=["timestamp"])
        base = [c for c in nwp.columns if c != "timestamp"]
        nidx = nwp.set_index("timestamp")
        for h in horizons:
            fut = nidx[base].reindex(df["timestamp"] + pd.Timedelta(hours=h))
            for c in base:
                col = f"{c}_t{h}h"
                df[col] = fut[c].to_numpy()
                nwp_cols.append(col)
        df["nwp_ok"] = df[nwp_cols].notna().all(axis=1).astype(int)
    else:
        df["nwp_ok"] = 0

    if tcfg["require_nwp"]:
        df = df[df["nwp_ok"] == 1].reset_index(drop=True)
    df = df[df["timestamp"] >= pd.Timestamp(tcfg["start"])].reset_index(drop=True)
    n_final = len(df)

    if n_final == 0:
        log.error("track %s: no rows survived", track)
        return None

    # A NaN in any model feature becomes a silently dropped batch at train time.
    # Catch it here, where it is attributable, rather than as a mysterious loss of
    # samples later. (A 20 W/m2 clear-sky floor once voided k_t on 6% of rows -
    # every one of them an 08:00 issue time.)
    feat_nan = {c: int(df[c].isna().sum()) for c in cfg["temporal"]["features"]
                if df[c].isna().any()}
    if feat_nan:
        raise AssertionError(
            f"NaN in model features would be dropped during training: {feat_nan}")

    df = chronological_split(df, cfg["dataset"]["split"])

    # ---- train_stats from the TRAINING SPLIT ONLY
    feats = cfg["temporal"]["features"]
    tr = df[df["split"] == "train"]
    stats = {
        "features": feats,
        "mean": {f: float(tr[f].mean()) for f in feats},
        "std": {f: float(max(tr[f].std(), 1e-6)) for f in feats},
        "grid_scalars": GRID_SCALARS,
        "grid_mean": {c: float(tr[c].mean()) for c in GRID_SCALARS},
        "grid_std": {c: float(max(tr[c].std(), 1e-6)) for c in GRID_SCALARS},
        "nwp_cols": nwp_cols,
        "nwp_mean": {c: float(tr[c].mean()) for c in nwp_cols} if nwp_cols else {},
        "nwp_std": {c: float(max(tr[c].std(), 1e-6)) for c in nwp_cols} if nwp_cols else {},
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "track": track,
        "train_end": str(tr["timestamp"].max()),
    }

    # ---- observation history for lookback windows
    #
    # The sample frame contains only ISSUE TIMES (08:00-14:00, 7 rows/day), because
    # a sample needs all three horizons inside daylight. Building the lookback from
    # that frame caps it at 7 consecutive hours and, in practice, averaged 4 - so a
    # 24-step window was 83% mean-padding and the BiLSTM never saw a real sequence.
    #
    # The lookback must instead come from the full DAYLIGHT observation series
    # (08:00-17:00, every day), stepping back one daylight hour at a time and across
    # the overnight gap. Using rows from before a split boundary is correct, not
    # leakage: at inference time t those observations genuinely exist. Leakage would
    # be using anything AFTER t, which reindex-by-timestamp cannot do.
    hist_cols = ["timestamp"] + list(dict.fromkeys(feats + ["ghi", "ghi_clearsky"]))
    hist = wx[wx["timestamp"].dt.hour.between(lo, hi)][hist_cols].copy()
    hist = hist.sort_values("timestamp").reset_index(drop=True)
    hist = hist[hist["timestamp"] <= df["timestamp"].max()]
    hist_nan = {c: int(hist[c].isna().sum()) for c in feats if hist[c].isna().any()}
    if hist_nan:
        # ghi_lag1 is NaN at the first daylight hour of the record only
        hist = hist.dropna(subset=feats).reset_index(drop=True)
    hist.to_parquet(proc / f"history_{track}.parquet", index=False)
    log.info("  history rows (08:00-%02d:00 daylight series): %d", hi, len(hist))

    out = proc / f"dataset_{track}.parquet"
    df.to_parquet(out, index=False)
    with open(proc / f"train_stats_{track}.json", "w") as f:
        json.dump(stats, f, indent=2)

    counts = df["split"].value_counts()
    log.info("track %s  %s -> %s", track, df["timestamp"].min().date(), df["timestamp"].max().date())
    log.info("  daylight rows           %d", n_daylight)
    log.info("  after target/lag filter %d  (-%d: night targets + overnight gap)",
             n_after_targets, n_daylight - n_after_targets)
    log.info("  after grid join         %d  (-%d without a rain grid)",
             n_after_grid, n_after_targets - n_after_grid)
    log.info("  after nwp/period filter %d", n_final)
    log.info("  SPLIT  train=%d  val=%d  test=%d",
             counts.get("train", 0), counts.get("val", 0), counts.get("test", 0))
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        if len(sub):
            log.info("    %-5s %s .. %s", s, sub["timestamp"].min().date(),
                     sub["timestamp"].max().date())

    n_test = int(counts.get("test", 0))
    floor = cfg["dataset"]["min_test_samples"]
    if n_test <= floor:
        raise AssertionError(f"Test set collapsed to {n_test} samples (need > {floor})")
    log.info("  wrote %s", out)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--track", default="both", choices=["A", "B", "both"])
    ap.add_argument("--smoke", action="store_true",
                    help="lower the test-size floor to smoke-test the pipeline on a "
                         "partial archive. Results are NOT valid for reporting.")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.smoke:
        cfg["dataset"]["min_test_samples"] = 1
        log.warning("SMOKE MODE: test-size floor lowered. Results are NOT valid "
                    "for reporting - rerun without --smoke on the full archive.")
    tracks = ["A", "B"] if args.track == "both" else [args.track]
    rc = 0
    for t in tracks:
        try:
            if build_track(cfg, t) is None:
                rc = 1
        except AssertionError as e:
            log.error("track %s FAILED: %s", t, e)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
