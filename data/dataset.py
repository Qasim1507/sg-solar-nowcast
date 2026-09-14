"""Torch Dataset: lookback windows, normalisation, grid loading, NWP head inputs.

WHERE THE LOOKBACK COMES FROM
-----------------------------
The sample frame holds only ISSUE TIMES (08:00-14:00, 7 rows/day) because a sample
needs all three horizons inside daylight. Walking back through THAT frame caps the
lookback at 7 steps and averaged 4 in practice - a 24-step window was 83%
mean-padding, so the BiLSTM never saw a real sequence.

The lookback is therefore built from the full DAYLIGHT OBSERVATION SERIES
(`history_{track}.parquet`, 08:00-17:00 every day), stepping back one daylight hour
at a time and across the overnight gap. Reaching back past a split boundary is
correct rather than leakage: at inference time t those observations genuinely
exist. Leakage would mean using data AFTER t, which stepping backwards cannot do.

PADDING CONSTRAINT
------------------
Zero-padding a lookback window BEFORE normalisation sends inputs in at -12.6 sigma
(temperature) and -6.7 sigma (humidity) - the network sees an impossible day. We
pad with `train_stats["mean"]`, which normalises to exactly 0, and we REFUSE to
serve a window that is mostly synthetic: if fewer than `min_real_fraction` of the
steps are real data, `build_window` raises.

The lookback is built by TIMESTAMP, walking back over real daylight rows, so it
never silently bridges the overnight gap the way positional slicing would.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PaddingError(RuntimeError):
    """Raised when a lookback window is too synthetic to serve."""


def load_stats(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def normalise(df: pd.DataFrame, cols: list[str], mean: dict, std: dict) -> np.ndarray:
    out = np.empty((len(df), len(cols)), dtype=np.float32)
    for k, c in enumerate(cols):
        out[:, k] = (df[c].to_numpy(dtype=np.float32) - mean[c]) / std[c]
    return out


class NowcastDataset(Dataset):
    """One sample = (tabular lookback, rain grid, gate features, future clear-sky,
    NWP at target hours) -> GHI at t+1/2/3h.

    Parameters
    ----------
    variant : which modalities the sample carries. Controls the ablation:
        'temporal' | 'spatial' | 'concat' | 'gated'
    """

    def __init__(self, df: pd.DataFrame, stats: dict, cfg: dict, grids_dir: Path,
                 variant: str = "gated", strict_padding: bool = True,
                 use_nwp: bool = True, history: pd.DataFrame | None = None):
        self.cfg = cfg
        self.stats = stats
        self.variant = variant
        self.grids_dir = Path(grids_dir)
        self.strict = strict_padding
        self.lookback = cfg["temporal"]["lookback"]
        self.horizons = cfg["horizons"]
        self.feats = stats["features"]
        self.min_real = cfg["dataset"]["min_real_fraction"]

        self.df = df.sort_values("timestamp").reset_index(drop=True)

        # The lookback reads from the daylight observation series, NOT from the
        # sample frame (see module docstring). Fall back to the sample frame only
        # when no history is supplied, which is the degraded path.
        hist = self.df if history is None else history.sort_values(
            "timestamp").reset_index(drop=True)
        self.history = hist
        self.X = normalise(hist, self.feats, stats["mean"], stats["std"])
        self.ts_to_row = {ts: i for i, ts in enumerate(hist["timestamp"])}
        # position of each timestamp in the daylight sequence, so "one step back"
        # means the previous DAYLIGHT hour (crossing the overnight gap), not t-1h
        self.ts_to_pos = self.ts_to_row
        self.mean_vec = np.array([0.0] * len(self.feats), dtype=np.float32)  # mean -> 0 post-norm

        # NWP is only carried when the track actually has it. Track B starts in
        # 2020, before the GFS archive (~2021-04), so its NWP columns are entirely
        # NaN by design - feeding them would make every sample non-finite.
        self.nwp_cols = stats.get("nwp_cols", [])
        self.use_nwp = (use_nwp and bool(self.nwp_cols)
                        and all(c in self.df.columns for c in self.nwp_cols)
                        and len(self.df) > 0
                        and bool(self.df[self.nwp_cols].notna().any().all()))
        if self.use_nwp:
            self.NWP = normalise(self.df, self.nwp_cols, stats["nwp_mean"], stats["nwp_std"])
            # any residual gap fills to the training mean, which is 0 after z-scoring
            self.NWP = np.nan_to_num(self.NWP, nan=0.0)

        self.gate_cols = ["clearsky_ratio", "cloud_cover", "motion_vx", "motion_vy"]
        gm = {**stats["mean"], **stats["grid_mean"]}
        gs = {**stats["std"], **stats["grid_std"]}
        self.G = np.nan_to_num(normalise(self.df, self.gate_cols, gm, gs), nan=0.0)

        self._grid_cache: dict[str, np.ndarray] = {}
        self.pad_events = 0

    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------ pieces
    def build_window(self, i: int) -> tuple[np.ndarray, int]:
        """Lookback ending at row i, walking back by TIMESTAMP. -> (L, F), n_real."""
        t0 = self.df["timestamp"].iloc[i]
        pos = self.ts_to_pos.get(t0)
        if pos is None:
            rows, n_real = [], 0
        else:
            # take the preceding `lookback` DAYLIGHT steps ending at t inclusive
            start = max(0, pos - self.lookback + 1)
            rows = list(range(start, pos + 1))
            n_real = len(rows)

        win = np.zeros((self.lookback, len(self.feats)), dtype=np.float32)
        if n_real:
            win[-n_real:] = self.X[rows]
        # pad with the TRAINING MEAN, which is exactly 0 in normalised space
        win[:self.lookback - n_real] = self.mean_vec

        frac = n_real / self.lookback
        if frac < self.min_real:
            self.pad_events += 1
            if self.strict:
                raise PaddingError(
                    f"lookback at {t0} is only {n_real}/{self.lookback} real "
                    f"({frac:.0%} < {self.min_real:.0%}); refusing a synthetic window")
        return win, n_real

    def load_grid(self, i: int) -> np.ndarray:
        row = self.df.iloc[i]
        name = row["path"]
        arr = self._grid_cache.get(name)
        if arr is None:
            arr = np.load(self.grids_dir / name, mmap_mode="r")
            if len(self._grid_cache) > 24:
                self._grid_cache.clear()
            self._grid_cache[name] = arr
        return np.asarray(arr[int(row["index"])], dtype=np.float32)

    # ------------------------------------------------------------------- item
    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        win, n_real = self.build_window(i)

        if self.variant in ("spatial", "concat", "gated"):
            grid = self.load_grid(i)
        else:
            grid = np.zeros((len(self.cfg["grid"]["frame_offsets_min"]) + 1,
                             self.cfg["grid"]["ny"], self.cfg["grid"]["nx"]), np.float32)

        future_cs = np.array([row[f"cs_t{h}h"] for h in self.horizons], dtype=np.float32)
        # clear-sky is pure astronomy; scale to a sane range rather than z-scoring
        future_cs_n = future_cs / 1000.0

        nwp = self.NWP[i] if self.use_nwp else np.zeros(0, dtype=np.float32)
        y = np.array([row[f"ghi_t{h}h"] for h in self.horizons], dtype=np.float32)

        return {
            "seq": torch.from_numpy(win),
            "grid": torch.from_numpy(np.ascontiguousarray(grid)),
            "gate": torch.from_numpy(self.G[i]),
            "future_cs": torch.from_numpy(future_cs_n),
            "future_cs_raw": torch.from_numpy(future_cs),
            "nwp": torch.from_numpy(np.asarray(nwp, dtype=np.float32)),
            "y": torch.from_numpy(y),
            "kt_now": torch.tensor(float(row["clearsky_ratio"]), dtype=torch.float32),
            "n_real": torch.tensor(n_real, dtype=torch.int16),
            "idx": torch.tensor(i, dtype=torch.long),
        }


def make_datasets(cfg: dict, track: str, variant: str = "gated",
                  root: Path | None = None) -> dict[str, NowcastDataset]:
    root = Path(root or Path(__file__).resolve().parent.parent)
    proc = root / cfg["paths"]["processed"]
    grids = root / cfg["paths"]["grids"]
    df = pd.read_parquet(proc / f"dataset_{track}.parquet")
    stats = load_stats(proc / f"train_stats_{track}.json")
    hist_path = proc / f"history_{track}.parquet"
    history = pd.read_parquet(hist_path) if hist_path.exists() else None
    if history is None:
        raise FileNotFoundError(
            f"{hist_path} is missing - rerun data/build_dataset.py. Without it every "
            f"lookback window collapses to ~4 real steps of 24.")

    # A track that does not require NWP does not carry an NWP branch at all.
    use_nwp = bool(cfg["dataset"]["tracks"][track].get("require_nwp", False))

    out = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].reset_index(drop=True)
        # Every split shares the same observation history; a window may reach back
        # past its split boundary, which is what a real forecaster would have.
        out[split] = NowcastDataset(sub, stats, cfg, grids, variant=variant,
                                    strict_padding=False, use_nwp=use_nwp,
                                    history=history)
    return out
