"""Single inference code path, shared by the CLI and the API.

There is exactly one implementation of preprocessing here; api/main.py imports
this module rather than reimplementing anything. That is what keeps the dashboard
from quietly disagreeing with the offline evaluation.

TWO MODES
---------
replay  Reconstructs an issue time from the STORED processed dataset, i.e. through
        the exact pipeline that produced the training rows. This is the honest
        path and the one to demo in a defence - no network, works at night.

live    Fetches current inputs. Note the product mismatch, which the dashboard
        must display: the model is TRAINED on ERA5 reanalysis GHI, but ERA5 runs
        ~2 days behind, so live recent observations come from the Open-Meteo
        forecast API's analysis (`past_days`) instead. That is train/serve skew.
        It is disclosed, not hidden.

The clear-sky envelope at t+1/2/3h is pure astronomy and is always available.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data._client import ApiClient, load_config, log, resolve_path, setup_logging  # noqa: E402
from data.build_dataset import GRID_SCALARS, add_calendar  # noqa: E402
from data.build_grids import Interpolator, estimate_motion  # noqa: E402
from data.geometry import GridSpec  # noqa: E402

# torch is imported LAZILY, only by Forecaster (the neural model). The served model
# is CatBoost, ~2 MB; importing torch at module scope would force every deployment
# and every scheduled CI run to install ~800 MB of wheels it never uses.
torch = None
MultimodalNowcaster = None


def _require_torch():
    global torch, MultimodalNowcaster
    if torch is None:
        import torch as _torch

        from models.fusion import MultimodalNowcaster as _Nowcaster
        torch = _torch
        MultimodalNowcaster = _Nowcaster
    return torch


def clearsky_series(cfg: dict, index: pd.DatetimeIndex) -> pd.Series:
    """Clear-sky GHI for serving - the SAME hour mean training uses.

    This used to take a single point at `eval_offset_min` (t-30) while training
    integrated over [t-60, t], so live k_t was inflated near sunrise. Both paths
    now call data/clearsky.py; tests/test_clearsky.py::test_train_serve_parity
    pins them together.
    """
    from data.clearsky import hour_mean_ghi
    return pd.Series(hour_mean_ghi(cfg, index), index=index)


class Forecaster:
    """Loads one checkpoint and produces quantile forecasts + diagnostics."""

    def __init__(self, checkpoint: str | Path, cfg: dict | None = None,
                 device: str | None = None):
        _require_torch()
        self.root = Path(__file__).resolve().parent
        ck_path = Path(checkpoint)
        if not ck_path.is_absolute():
            ck_path = self.root / (cfg or load_config())["paths"]["checkpoints"] / ck_path
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available()
                                              else "cpu"))
        ck = torch.load(ck_path, map_location=self.device, weights_only=False)
        self.cfg = cfg or ck.get("config") or load_config()
        self.ck = ck
        self.checkpoint_name = ck_path.name
        self.quantiles = ck["quantiles"]
        self.horizons = ck["horizons"]
        self.variant = ck["variant"]
        self.track = ck["track"]

        self.model = MultimodalNowcaster(
            n_features=ck["n_features"], n_grid_channels=ck["n_grid_channels"],
            n_horizons=len(self.horizons), n_quantiles=len(self.quantiles),
            n_nwp=ck["n_nwp"], variant=ck["variant"], cfg=self.cfg,
        ).to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

        proc = self.root / self.cfg["paths"]["processed"]
        with open(proc / f"train_stats_{self.track}.json") as f:
            self.stats = json.load(f)
        self.features = self.stats["features"]
        self.lookback = self.cfg["temporal"]["lookback"]

    def info(self) -> dict:
        """Metadata, in the shape api/main.py reports. Mirrored by CatBoostForecaster."""
        return self.ck

    # ---------------------------------------------------------------- helpers
    def _normalise(self, df: pd.DataFrame, cols, mean, std) -> np.ndarray:
        out = np.empty((len(df), len(cols)), dtype=np.float32)
        for k, c in enumerate(cols):
            out[:, k] = (df[c].to_numpy(dtype=np.float32) - mean[c]) / std[c]
        return np.nan_to_num(out, nan=0.0)

    def _window(self, hist: pd.DataFrame, t: pd.Timestamp) -> tuple[np.ndarray, int]:
        """Lookback over the preceding DAYLIGHT steps, padded with the training mean.

        Must match data/dataset.NowcastDataset.build_window exactly or we create
        train/serve skew. `hist` is the daylight observation series, so one step
        back is the previous daylight hour - crossing the overnight gap - not t-1h.
        Stepping backwards from t cannot pull in anything after t.
        """
        hist = hist.sort_values("timestamp").reset_index(drop=True)
        X = self._normalise(hist, self.features, self.stats["mean"], self.stats["std"])
        pos = {ts: i for i, ts in enumerate(hist["timestamp"])}.get(t)
        if pos is None:
            # t is not in the series (live mode, partial hour): take everything before it
            prior = hist.index[hist["timestamp"] <= t]
            pos = int(prior[-1]) if len(prior) else None
        rows = ([] if pos is None
                else list(range(max(0, pos - self.lookback + 1), pos + 1)))
        win = np.zeros((self.lookback, len(self.features)), np.float32)
        if rows:
            win[-len(rows):] = X[rows]
        return win, len(rows)

    def _run(self, seq, grid, gate, future_cs, nwp) -> tuple[np.ndarray, float | None]:
        with torch.no_grad():
            return self._run_inner(seq, grid, gate, future_cs, nwp)

    def _run_inner(self, seq, grid, gate, future_cs, nwp) -> tuple[np.ndarray, float | None]:
        batch = {
            "seq": torch.from_numpy(seq[None]).to(self.device),
            "grid": torch.from_numpy(np.ascontiguousarray(grid[None])).to(self.device),
            "gate": torch.from_numpy(gate[None]).to(self.device),
            "future_cs": torch.from_numpy((future_cs / 1000.0).astype(np.float32)[None]).to(self.device),
            "nwp": torch.from_numpy(nwp.astype(np.float32)[None]).to(self.device),
        }
        pred = self.model(batch)
        hi = torch.from_numpy(future_cs.astype(np.float32)[None]).to(self.device).unsqueeze(-1) * 1.25
        pred = torch.clamp(pred, min=0.0).minimum(hi)
        alpha = getattr(self.model, "last_alpha", None)
        return pred[0].cpu().numpy(), (float(alpha.flatten()[0]) if alpha is not None else None)

    # ------------------------------------------------------------------ modes
    def replay(self, issue_time: str | pd.Timestamp | None = None) -> dict:
        """Forecast from the stored dataset - the exact training pipeline."""
        proc = self.root / self.cfg["paths"]["processed"]
        grids_dir = self.root / self.cfg["paths"]["grids"]
        df = pd.read_parquet(proc / f"dataset_{self.track}.parquet")
        df = df.sort_values("timestamp").reset_index(drop=True)
        hist_p = proc / f"history_{self.track}.parquet"
        if not hist_p.exists():
            raise FileNotFoundError(f"{hist_p} missing - rerun data/build_dataset.py")
        history = pd.read_parquet(hist_p)

        if issue_time is None:
            row = df.iloc[-1]
        else:
            t = pd.Timestamp(issue_time)
            m = df["timestamp"] == t
            if not m.any():
                raise ValueError(f"{t} is not an issue time in track {self.track}")
            row = df[m].iloc[0]
        t = row["timestamp"]

        seq, n_real = self._window(history, t)
        grid = np.asarray(np.load(grids_dir / row["path"], mmap_mode="r")[int(row["index"])],
                          dtype=np.float32)
        gate_cols = ["clearsky_ratio", "cloud_cover", "motion_vx", "motion_vy"]
        gm = {**self.stats["mean"], **self.stats["grid_mean"]}
        gs = {**self.stats["std"], **self.stats["grid_std"]}
        gate = self._normalise(pd.DataFrame([row]), gate_cols, gm, gs)[0]

        future_cs = np.array([row[f"cs_t{h}h"] for h in self.horizons], dtype=np.float32)
        nwp_cols = self.stats.get("nwp_cols", [])
        use_nwp = bool(self.ck["n_nwp"])
        nwp = (self._normalise(pd.DataFrame([row]), nwp_cols,
                               self.stats["nwp_mean"], self.stats["nwp_std"])[0]
               if use_nwp else np.zeros(0, np.float32))

        pred, alpha = self._run(seq, grid, gate, future_cs, nwp)
        actual = [float(row[f"ghi_t{h}h"]) for h in self.horizons]
        return self._package(t, pred, future_cs, alpha, n_real, row, grid,
                             mode="replay", actual=actual)

    def live(self) -> dict:
        """Forecast from current data. Discloses the ERA5 -> analysis substitution."""
        g = gather_live_inputs(self.cfg, self.horizons, want_nwp=bool(self.ck["n_nwp"]))
        hist, t, row = g["hist"], g["t"], g["row"]

        seq, n_real = self._window(hist, t)
        gate_cols = ["clearsky_ratio", "cloud_cover", "motion_vx", "motion_vy"]
        gm = {**self.stats["mean"], **self.stats["grid_mean"]}
        gs = {**self.stats["std"], **self.stats["grid_std"]}
        gate = self._normalise(pd.DataFrame([row]), gate_cols, gm, gs)[0]

        nwp = (self._normalise(g["nwp_row"], self.stats["nwp_cols"],
                               self.stats["nwp_mean"], self.stats["nwp_std"])[0]
               if g["nwp_row"] is not None else np.zeros(0, np.float32))

        pred, alpha = self._run(seq, g["grid"], gate, g["future_cs"], nwp)
        out = self._package(t, pred, g["future_cs"], alpha, n_real, row, g["grid"],
                            mode="live")
        out["observed_history"] = g["observed_history"]
        out["data_age_minutes"]["nea"] = g["grid_meta"].get("age_minutes")
        out["data_age_minutes"]["nwp"] = g["nwp_age"]
        return out

    def _live_grid(self, client: ApiClient, t: pd.Timestamp):
        return live_grid(self.cfg, client, t)


def live_grid(cfg: dict, client: ApiClient, t: pd.Timestamp):
    """Build the current (C, ny, nx) rain raster from live gauge readings."""
    spec = GridSpec.from_config(cfg)
    g = cfg["grid"]
    interp = Interpolator(spec, g["method"], g["idw_power"], g["idw_max_km"])
    url = f"{cfg['nea']['base_url']}/rainfall"
    payload = client.get_json(url, params={"date": t.strftime("%Y-%m-%d")})
    from data.fetch_nea import parse_response
    df, meta = parse_response(payload)
    if df.empty:
        z = np.zeros((len(g["frame_offsets_min"]) + 1, g["ny"], g["nx"]), np.float32)
        return z, {c: 0.0 for c in GRID_SCALARS} | {"age_minutes": None}
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True) \
        .dt.tz_convert(cfg["site"]["timezone"]).dt.tz_localize(None)

    # The newest 5-minute frame is usually PARTIAL - gauge reports trickle in over
    # a minute or two, so max(timestamp) can carry 3 of 88 stations. Take the most
    # recent frame that actually has adequate coverage, and report its true age.
    counts = df.groupby("timestamp")["station_id"].nunique()
    floor = max(cfg["nea"]["min_stations"], int(0.5 * counts.max()))
    usable = counts[counts >= floor]
    if usable.empty:
        latest = counts.idxmax()
    else:
        latest = usable.index.max()

    chans, n_rep = [], []
    for off in g["frame_offsets_min"]:
        ts = latest - pd.Timedelta(minutes=off)
        grp = df[df["timestamp"] == ts]
        if grp.empty:
            grp = df[df["timestamp"] == latest]
        field, mask, n_used = interp(grp["station_id"].tolist(),
                                     grp["value"].tolist(), meta)
        chans.append(field)
        n_rep.append(n_used)
    arr = np.stack(chans + [mask], axis=0).astype(np.float32)
    dx, dy, valid = estimate_motion(chans[1], chans[0]) if len(chans) > 1 else (0., 0., 0)
    scalars = {
        "rain_mean": float(chans[0].mean()), "rain_max": float(chans[0].max()),
        "wet_frac": float((chans[0] > 0.01).mean()), "rain_std": float(chans[0].std()),
        "mask_frac": float(mask.mean()), "motion_vx": dx, "motion_vy": dy,
        "motion_valid": valid, "n_stations_reporting": int(min(n_rep)),
        "age_minutes": max(0.0, round(
            (pd.Timestamp(datetime.utcnow() + timedelta(hours=cfg["site"]["tz_offset_hours"]))
             - latest).total_seconds() / 60.0, 1)),
        "latest_reading": str(latest),
        "newest_frame_skipped": str(counts.index.max()) if latest != counts.index.max() else None,
        "stations": [{"id": sid, "name": m.get("name"),
                      "lat": m.get("latitude"), "lon": m.get("longitude"),
                      "value": float(v)}
                     for sid, v in zip(df[df["timestamp"] == latest]["station_id"],
                                       df[df["timestamp"] == latest]["value"])
                     if (m := meta.get(sid)) is not None],
    }
    return arr, scalars

    # ---------------------------------------------------------------- packaging
    def _model_block(self) -> dict:
        return {
            "checkpoint": self.checkpoint_name, "variant": self.variant,
            "track": self.track, "seed": self.ck.get("seed"),
            "trained_at": self.ck.get("trained_at"),
            "n_train": self.ck.get("n_train"), "n_test": self.ck.get("n_test"),
        }

    def _package(self, t, pred, future_cs, alpha, n_real, row, grid,
                 mode: str, actual=None) -> dict:
        return build_package(
            cfg=self.cfg, t=t, pred=pred, future_cs=future_cs,
            quantiles=self.quantiles, horizons=self.horizons, lookback=self.lookback,
            alpha=alpha, n_real=n_real, row=row, mode=mode,
            model_block=self._model_block(), actual=actual)


class CatBoostForecaster:
    """Serves the tree model: CatBoost + k_t, one MultiQuantile model per horizon.

    This is the model the dashboard serves. On Track A it beats every deep variant
    at every horizon (t+1h 18.26, t+2h 25.22, t+3h 27.28 pinball, 3-seed means)
    with seed sd <= 0.08.

    It emits the SAME package shape as Forecaster - see build_package - so the
    frontend and tests/test_api.py cannot tell the two apart structurally.

    Trees take raw features, so none of the z-scoring the neural path needs
    applies here; the only transform is k_t -> W/m2 on the way out.
    """

    def __init__(self, sidecar: str | Path, cfg: dict | None = None):
        from catboost import CatBoostRegressor

        self.root = Path(__file__).resolve().parent
        self.cfg = cfg or load_config()
        side_path = Path(sidecar)
        if not side_path.is_absolute():
            side_path = self.root / self.cfg["paths"]["checkpoints"] / side_path
        with open(side_path) as f:
            self.meta = json.load(f)

        self.checkpoint_name = side_path.name
        self.model_id = self.meta["model_id"]
        self.track = self.meta["track"]
        self.quantiles = self.meta["quantiles"]
        self.horizons = self.meta["horizons"]
        self.features = self.meta["feature_cols"]
        self.target_space = self.meta.get("target_space", "kt")
        self.variant = "catboost"
        self.lookback = self.cfg["temporal"]["lookback"]

        self.models = {}
        for h in self.horizons:
            m = CatBoostRegressor()
            m.load_model(str(side_path.parent / f"{self.model_id}_t{h}h.cbm"))
            self.models[h] = m

    def info(self) -> dict:
        return self.meta

    def _live_grid(self, client: ApiClient, t: pd.Timestamp):
        return live_grid(self.cfg, client, t)

    # ---------------------------------------------------------------- helpers
    def _features(self, row: dict) -> np.ndarray:
        """Feature vector in the sidecar's contractual order. Missing -> NaN.

        CatBoost handles NaN natively (nan_mode defaults to Min), so a gap degrades
        one feature rather than failing the whole forecast.
        """
        return np.array([[float(row.get(c, np.nan)) if row.get(c) is not None else np.nan
                          for c in self.features]], dtype=float)

    def _predict(self, row: dict, future_cs: np.ndarray) -> np.ndarray:
        """-> (n_horizons, n_quantiles) in W/m2, sorted and physically clipped."""
        X = self._features(row)
        out = np.empty((len(self.horizons), len(self.quantiles)), dtype=float)
        for k, h in enumerate(self.horizons):
            q = np.asarray(self.models[h].predict(X), dtype=float).ravel()
            if self.target_space == "kt":
                q = q * float(future_cs[k])
            out[k] = q
        out = np.sort(out, axis=1)
        # same physics clamp the offline scorer applies (baselines._clip_physical)
        return np.clip(out, 0.0, future_cs[:, None] * 1.25)

    def _model_block(self) -> dict:
        return {
            "checkpoint": self.checkpoint_name, "variant": "catboost",
            "track": self.track, "seed": self.meta.get("seed"),
            "trained_at": self.meta.get("trained_at"),
            "n_train": self.meta.get("n_train"), "n_test": None,
        }

    def _package(self, t, pred, future_cs, n_real, row, mode, actual=None) -> dict:
        return build_package(
            cfg=self.cfg, t=t, pred=pred, future_cs=future_cs,
            quantiles=self.quantiles, horizons=self.horizons, lookback=self.lookback,
            alpha=None, n_real=n_real, row=row, mode=mode,
            model_block=self._model_block(), actual=actual)

    # ------------------------------------------------------------------ modes
    def replay(self, issue_time: str | pd.Timestamp | None = None) -> dict:
        """Forecast from the stored dataset - the exact training pipeline."""
        proc = self.root / self.cfg["paths"]["processed"]
        df = pd.read_parquet(proc / f"dataset_{self.track}.parquet")
        df = df.sort_values("timestamp").reset_index(drop=True)
        if issue_time is None:
            row_s = df.iloc[-1]
        else:
            t = pd.Timestamp(issue_time)
            m = df["timestamp"] == t
            if not m.any():
                raise ValueError(f"{t} is not an issue time in track {self.track}")
            row_s = df[m].iloc[0]

        t = row_s["timestamp"]
        row = row_s.to_dict()
        future_cs = np.array([row[f"cs_t{h}h"] for h in self.horizons], dtype=np.float32)
        pred = self._predict(row, future_cs)
        actual = [float(row[f"ghi_t{h}h"]) for h in self.horizons]
        return self._package(t, pred, future_cs, self.lookback, row, "replay", actual)

    def live(self) -> dict:
        """Forecast from current data. Discloses the ERA5 -> analysis substitution."""
        g = gather_live_inputs(self.cfg, self.horizons, want_nwp=True)
        row = dict(g["row"])
        if g["nwp_row"] is not None:
            row.update(g["nwp_row"].iloc[0].to_dict())

        pred = self._predict(row, g["future_cs"])
        n_real = int(min(len(g["hist"]), self.lookback))
        out = self._package(g["t"], pred, g["future_cs"], n_real, row, "live")
        out["observed_history"] = g["observed_history"]
        out["data_age_minutes"]["nea"] = g["grid_meta"].get("age_minutes")
        out["data_age_minutes"]["nwp"] = g["nwp_age"]
        return out


def build_package(*, cfg, t, pred, future_cs, quantiles, horizons, lookback,
                  alpha, n_real, row, mode, model_block, actual=None) -> dict:
    """The response contract shared by every served model.

    Kept model-agnostic so a tree model and the neural model return byte-identical
    shapes - the frontend and tests/test_api.py depend on this.
    """
    kt_now = float(row.get("clearsky_ratio", np.nan))
    lo, hi = min(quantiles), max(quantiles)
    hz_out = []
    for k, h in enumerate(horizons):
        q = {str(qq): float(pred[k, i]) for i, qq in enumerate(quantiles)}
        hz_out.append({
            "horizon_h": h,
            "valid_time": str(t + pd.Timedelta(hours=h)),
            "quantiles": q,
            "median": float(pred[k, quantiles.index(0.5)]),
            "lower": float(pred[k, 0]), "upper": float(pred[k, -1]),
            "clearsky_ghi": float(future_cs[k]),
            "actual_ghi": (float(actual[k]) if actual else None),
        })
    daylight_ok = cfg["daylight"]["start_hour"] <= t.hour <= cfg["daylight"]["end_hour"]
    return {
        "as_of": str(t),
        "issue_time": str(t),
        "timezone": "SGT (UTC+8)",
        "mode": mode,
        "nominal_interval": round(hi - lo, 4),
        "quantiles": quantiles,
        "horizons": hz_out,
        "kt_now": None if not np.isfinite(kt_now) else kt_now,
        "bimodal_warning": bool(np.isfinite(kt_now) and kt_now < 0.4),
        "outside_training_window": not daylight_ok,
        "diagnostics": {
            "gate_alpha": alpha,
            "lookback_real_steps": int(n_real),
            "lookback_required": lookback,
            "lookback_real_fraction": round(n_real / lookback, 3),
            "n_stations_reporting": int(row.get("n_stations_reporting", 0) or 0),
            "motion_vx": float(row.get("motion_vx", 0) or 0),
            "motion_vy": float(row.get("motion_vy", 0) or 0),
            "motion_valid": int(row.get("motion_valid", 0) or 0),
            "grid_wet_fraction": float(row.get("wet_frac", 0) or 0),
            "grid_mask_fraction": float(row.get("mask_frac", 0) or 0),
        },
        "model": model_block,
        "target_product": "ERA5 reanalysis (Open-Meteo archive)",
        "observation_product": ("ERA5 reanalysis" if mode == "replay"
                                else "Open-Meteo forecast-API analysis (past_days) - "
                                     "NOT ERA5; ERA5 lags ~2 days"),
        "data_age_minutes": {},
    }


def gather_live_inputs(cfg: dict, horizons: list[int], want_nwp: bool = True) -> dict:
    """Fetch everything a live forecast needs, for ANY model family.

    Extracted so the neural and tree paths cannot drift apart the way clear-sky
    did (see data/clearsky.py). Returns raw, UN-normalised values; each model
    applies whatever transform it needs.
    """
    from data.clearsky import kt_from_ghi

    client = ApiClient(delay_s=0.5, backoff_s=8.0)
    site = cfg["site"]

    # --- recent observations (NOT ERA5: see module docstring)
    obs = client.get_json(cfg["nwp"]["live_url"], params={
        "latitude": site["latitude"], "longitude": site["longitude"],
        "hourly": ",".join(cfg["temporal"]["hourly_vars"]),
        "timezone": site["timezone"], "past_days": 3, "forecast_days": 2})["hourly"]
    hist = pd.DataFrame(obs).rename(columns={"time": "timestamp",
                                             "shortwave_radiation": "ghi"})
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    now = pd.Timestamp(datetime.utcnow() + timedelta(hours=site["tz_offset_hours"]))
    t = hist.loc[hist["timestamp"] <= now.floor("h"), "timestamp"].max()

    hist["ghi_clearsky"] = clearsky_series(cfg, pd.DatetimeIndex(hist["timestamp"])).to_numpy()
    hist["clearsky_ratio"] = kt_from_ghi(cfg, hist["ghi"], hist["ghi_clearsky"])
    hist = add_calendar(hist)

    # Lags come off the FULL hourly frame by timestamp, exactly as
    # data/build_dataset.py::build_targets_and_lags does - never a positional
    # shift, and never after the daylight filter, which would skip the overnight
    # gap and silently change what "one hour ago" means.
    idx = hist.set_index("timestamp")
    for L in (1, 2, 3):
        back = hist["timestamp"] - pd.Timedelta(hours=L)
        hist[f"ghi_lag{L}"] = idx["ghi"].reindex(back).to_numpy()
        hist[f"kt_lag{L}"] = idx["clearsky_ratio"].reindex(back).to_numpy()

    # Restrict to the daylight window so that "one step back" means the same
    # thing here as it does in training.
    lo, hi = cfg["daylight"]["start_hour"], cfg["daylight"]["end_hour"]
    hist = hist[(hist["timestamp"] <= t)
                & hist["timestamp"].dt.hour.between(lo, hi)].reset_index(drop=True)

    grid, grid_meta = live_grid(cfg, client, t)
    row = hist[hist["timestamp"] == t].iloc[0].to_dict()
    row.update(grid_meta)

    future_times = pd.DatetimeIndex([t + pd.Timedelta(hours=h) for h in horizons])
    future_cs = clearsky_series(cfg, future_times).to_numpy().astype(np.float32)

    # --- NWP at the target hours, from the SAME pinned model as training
    nwp_row, nwp_age = None, None
    if want_nwp:
        from data.fetch_nwp import fetch_model_range
        frames = [fetch_model_range(client, cfg, m, "", "", live=True)
                  for m in cfg["nwp"]["models"]]
        n = frames[0]
        for f in frames[1:]:
            n = n.merge(f, on="timestamp", how="outer")
        nidx = n.set_index("timestamp")
        vals, cols = [], []
        base = [c for c in n.columns if c != "timestamp"]
        for h in horizons:
            r = nidx[base].reindex([t + pd.Timedelta(hours=h)])
            for c in base:
                vals.append(float(r[c].iloc[0]) if pd.notna(r[c].iloc[0]) else np.nan)
                cols.append(f"{c}_t{h}h")
        nwp_row = pd.DataFrame([dict(zip(cols, vals))])
        nwp_age = 0

    return {
        "hist": hist, "t": t, "row": row, "grid": grid, "grid_meta": grid_meta,
        "future_cs": future_cs, "nwp_row": nwp_row, "nwp_age": nwp_age,
        "observed_history": [
            {"timestamp": str(ts), "ghi": (None if pd.isna(v) else float(v))}
            for ts, v in zip(hist["timestamp"], hist["ghi"])][-24:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", default="replay", choices=["replay", "live"])
    ap.add_argument("--issue-time", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    f = Forecaster(args.checkpoint, cfg)
    out = f.live() if args.mode == "live" else f.replay(args.issue_time)
    text = json.dumps(out, indent=2, default=float)
    if args.out:
        Path(args.out).write_text(text)
        log.info("wrote %s", args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
