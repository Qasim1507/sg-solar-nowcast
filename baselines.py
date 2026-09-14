"""Baselines, built and reported BEFORE any deep model.

  1. Persistence          ghi(t+h) = ghi(t)
  2. Smart persistence    hold the clear-sky index: k_t(t) * clearsky(t+h)
  3. LightGBM             quantile regression on tabular features + grid scalars

Smart persistence is the honest benchmark for irradiance nowcasting - plain
persistence is easy to beat simply by knowing the sun moves. Skill scores are
reported against smart persistence.

The deterministic baselines are given a probabilistic form so they can be scored
on pinball/CRPS/coverage alongside the model: their quantiles come from the
EMPIRICAL DISTRIBUTION OF TRAINING RESIDUALS in k_t space, which is both standard
practice and a genuinely strong competitor. Residual quantiles are fitted on the
training split only.

In the prior project the tabular baselines beat every deep variant. If that
happens again it is the finding, and it is reported as such.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# NOTE ON IMPORT ORDER - do not "tidy" this.
# On macOS, LightGBM and PyTorch each bring their own libomp. If torch is imported
# FIRST and LightGBM then spawns its OpenMP pool, the process segfaults (SIGSEGV,
# exit 139) the moment .fit() runs. Importing lightgbm before torch avoids it, and
# n_jobs=1 below is the second, independent guard.
import lightgbm as lgb  # noqa: E402  (must precede torch)
from catboost import CatBoostRegressor  # noqa: E402  (same libomp caveat as lightgbm)
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
from data._client import load_config, log, resolve_path, setup_logging  # noqa: E402

GRID_SCALARS = ["rain_mean", "rain_max", "wet_frac", "rain_std", "mask_frac",
                "motion_vx", "motion_vy", "motion_valid"]

# Statistics of the t-30 / t-60 raster channels plus their tendencies. build_grids
# summarises only the frame at t, so without these the tabular models carry no
# sub-hourly rain history. Produced by scripts/extract_subhourly_scalars.py.
SUBHOURLY_SCALARS = [
    "rain_mean_m30", "rain_max_m30", "wet_frac_m30", "rain_std_m30",
    "rain_mean_m60", "rain_max_m60", "wet_frac_m60", "rain_std_m60",
    "rain_mean_d30", "rain_mean_d60", "wet_frac_d30", "wet_frac_d60",
    "rain_max_d30",
]


def _kt_from_ghi(ghi: np.ndarray, cs: np.ndarray, cfg: dict) -> np.ndarray:
    """GHI -> clear-sky index, using the SAME floor/cap as data/fetch_weather.py.

    Training on k_t instead of raw GHI removes the deterministic solar-geometry
    envelope from the target, which the learner would otherwise have to re-derive
    from sin/cos hour+month. It also flattens the conditional spread: in W/m2 the
    residual scale varies by an order of magnitude between 08:00 and midday, which
    is what makes the outer quantiles unstable.
    """
    cscfg = cfg["clearsky"]
    floor = float(cscfg.get("min_ghi_for_ratio", 5.0))
    cap = float(cscfg.get("max_ratio", 1.5))
    kt = ghi / np.clip(cs, floor, None)
    return np.clip(kt, 0.0, cap)


def _clip_physical(pred: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """GHI cannot be negative, and a clear-sky clamp is physics, not tuning."""
    return np.clip(pred, 0.0, cs[:, None] * 1.25 if pred.ndim == 2 else cs * 1.25)


def persistence(train: pd.DataFrame, test: pd.DataFrame, h: int, quantiles) -> np.ndarray:
    """ghi(t+h) = ghi(t), with quantiles from training residuals."""
    res = train["ghi"].to_numpy() - train[f"ghi_t{h}h"].to_numpy()
    qs = np.quantile(res[np.isfinite(res)], [1 - q for q in quantiles])
    point = test["ghi"].to_numpy()[:, None]
    pred = point - qs[None, :]
    pred = np.sort(pred, axis=1)
    return _clip_physical(pred, test[f"cs_t{h}h"].to_numpy())


def smart_persistence(train: pd.DataFrame, test: pd.DataFrame, h: int,
                      quantiles) -> np.ndarray:
    """Hold k_t constant. Quantiles from the training distribution of k_t change."""
    kt_now_tr = train["clearsky_ratio"].to_numpy()
    kt_fut_tr = train[f"ghi_t{h}h"].to_numpy() / np.where(
        train[f"cs_t{h}h"].to_numpy() > 5, train[f"cs_t{h}h"].to_numpy(), np.nan)
    d = kt_fut_tr - kt_now_tr
    d = d[np.isfinite(d)]
    dq = np.quantile(d, quantiles)                  # additive k_t change quantiles

    kt_now = test["clearsky_ratio"].to_numpy()[:, None]
    cs_fut = test[f"cs_t{h}h"].to_numpy()[:, None]
    pred = (kt_now + dq[None, :]) * cs_fut
    pred = np.sort(pred, axis=1)
    return _clip_physical(pred, test[f"cs_t{h}h"].to_numpy())


def smart_persistence_point(test: pd.DataFrame, h: int) -> np.ndarray:
    return test["clearsky_ratio"].to_numpy() * test[f"cs_t{h}h"].to_numpy()


def _targets(train, val, test, h: int, cfg: dict, space: str):
    """Return (ytr, yva, cs_te) for the requested target space ("ghi" or "kt")."""
    if space == "ghi":
        return train[f"ghi_t{h}h"].to_numpy(), val[f"ghi_t{h}h"].to_numpy(), None
    if space != "kt":
        raise ValueError(f"space must be 'ghi' or 'kt', got {space!r}")
    ytr = _kt_from_ghi(train[f"ghi_t{h}h"].to_numpy(), train[f"cs_t{h}h"].to_numpy(), cfg)
    yva = _kt_from_ghi(val[f"ghi_t{h}h"].to_numpy(), val[f"cs_t{h}h"].to_numpy(), cfg)
    return ytr, yva, test[f"cs_t{h}h"].to_numpy()


def _to_ghi(pred: np.ndarray, cs_te, test, h: int) -> np.ndarray:
    """Map predictions back to W/m2 if they were made in k_t space, then clip."""
    if cs_te is not None:
        pred = pred * cs_te[:, None]
    pred = np.sort(pred, axis=1)
    return _clip_physical(pred, test[f"cs_t{h}h"].to_numpy())


def lightgbm_quantiles(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                       h: int, quantiles, feature_cols: list[str],
                       seed: int = 0, cfg: dict | None = None,
                       space: str = "ghi") -> np.ndarray:
    """One independent fit per quantile; crossing repaired by a post-hoc sort."""
    ytr, yva, cs_te = _targets(train, val, test, h, cfg, space)
    Xtr, Xva, Xte = train[feature_cols], val[feature_cols], test[feature_cols]
    preds = []
    for q in quantiles:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=600, learning_rate=0.05,
            num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=1,
        )
        m.fit(Xtr, ytr, eval_X=Xva, eval_y=yva,
              callbacks=[lgb.early_stopping(50, verbose=False)])
        preds.append(m.predict(Xte))
    return _to_ghi(np.stack(preds, axis=1), cs_te, test, h)


def fit_catboost(train: pd.DataFrame, val: pd.DataFrame, h: int, quantiles,
                 feature_cols: list[str], seed: int = 0, cfg: dict | None = None,
                 space: str = "ghi") -> CatBoostRegressor:
    """One MultiQuantile model for horizon h, emitting every quantile.

    Unlike the LightGBM path above (5 independent fits + a post-hoc sort), the
    quantiles here share a tree structure, so the outer levels borrow strength
    from the median and cannot cross by construction.
    """
    ytr, yva, _ = _targets(train, val, train, h, cfg, space)
    alphas = ",".join(str(q) for q in quantiles)
    m = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={alphas}",
        iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        random_seed=seed, verbose=0, thread_count=1, allow_writing_files=False,
    )
    m.fit(train[feature_cols], ytr, eval_set=(val[feature_cols], yva),
          early_stopping_rounds=50, verbose=0)
    return m


def catboost_quantiles(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                       h: int, quantiles, feature_cols: list[str],
                       seed: int = 0, cfg: dict | None = None,
                       space: str = "ghi") -> np.ndarray:
    _, _, cs_te = _targets(train, val, test, h, cfg, space)
    m = fit_catboost(train, val, h, quantiles, feature_cols, seed, cfg, space)
    return _to_ghi(np.asarray(m.predict(test[feature_cols]), dtype=float), cs_te, test, h)


def save_catboost(cfg: dict, track: str, seed: int = 0) -> Path:
    """Persist the served model: CatBoost + k_t, one .cbm per horizon.

    The API needs a model on disk; baselines.py otherwise fits and discards. The
    sidecar records `feature_cols` IN ORDER - that ordering is contractual for
    serving exactly as train_stats_{track}.json is for the deep model.
    """
    import time

    proc = resolve_path(cfg, "processed")
    ck_dir = resolve_path(cfg, "checkpoints")
    ck_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(proc / f"dataset_{track}.parquet")
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    quantiles = cfg["quantiles"]
    feats = feature_columns(cfg, df, use_grid=True)

    for h in cfg["horizons"]:
        m = fit_catboost(train, val, h, quantiles, feats, seed, cfg, "kt")
        m.save_model(str(ck_dir / f"catboost_{track}_kt_t{h}h.cbm"))
        log.info("  saved catboost_%s_kt_t%dh.cbm (%d trees)", track, h, m.tree_count_)

    side = ck_dir / f"catboost_{track}_kt.json"
    with open(side, "w") as f:
        json.dump({
            "model_id": f"catboost_{track}_kt",
            "kind": "catboost", "track": track, "target_space": "kt",
            "feature_cols": feats, "quantiles": quantiles,
            "horizons": cfg["horizons"], "seed": seed,
            "n_train": int(len(train)), "n_val": int(len(val)),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)
    log.info("wrote %s  (%d features)", side, len(feats))
    return side


# The tree line-up: (name, learner, feature set, target space). Both GHI-space
# LightGBM entries are kept deliberately - they are the incumbents REPORT.md is
# written against, so the k_t gain stays reproducible side by side.
TREE_SPECS = (
    ("lightgbm_tabular",     "lgbm",     "tab",  "ghi"),
    ("lightgbm_tab+grid",    "lgbm",     "grid", "ghi"),
    ("lgbm_kt_tab+grid",     "lgbm",     "grid", "kt"),
    ("catboost_tab+grid",    "catboost", "grid", "ghi"),
    ("catboost_kt_tab+grid", "catboost", "grid", "kt"),
)

# Opt-in via --subhourly. Same learner and target space as the two k_t entries
# above, one extra feature block, so the delta isolates what sub-hourly rain
# history is worth. MEASURED 2026-09-14, Track A seed 0: it is worth nothing.
# t+1h 18.24 -> 18.20 (lgbm, inside the +/-0.05 seed sd) and 18.23 -> 18.39
# (catboost, WORSE); both degrade at t+3h. Rain gauges measure rain, irradiance
# is governed by cloud, and the tendency features are non-zero on only ~29% of
# frames - 13 mostly-empty columns act as noise dimensions. Kept so the negative
# result stays reproducible, off by default so it costs nothing.
SUBHOURLY_SPECS = (
    ("lgbm_kt_+subhourly",     "lgbm",     "sub",  "kt"),
    ("catboost_kt_+subhourly", "catboost", "sub",  "kt"),
)

LEARNERS = {"lgbm": lightgbm_quantiles, "catboost": catboost_quantiles}


def feature_columns(cfg: dict, df: pd.DataFrame, use_grid: bool = True,
                    use_subhourly: bool = False) -> list[str]:
    cols = list(cfg["temporal"]["features"])
    cols += [c for c in ("ghi_lag2", "ghi_lag3", "kt_lag1", "kt_lag2") if c in df.columns]
    if use_grid:
        cols += [c for c in GRID_SCALARS if c in df.columns]
    if use_subhourly:
        cols += [c for c in SUBHOURLY_SCALARS if c in df.columns]
    nwp = [c for c in df.columns if c.startswith("nwp_")]
    cols += nwp
    return [c for c in dict.fromkeys(cols) if c in df.columns]


def run(cfg: dict, track: str, seed: int = 0, subhourly: bool = False) -> dict:
    proc = resolve_path(cfg, "processed")
    reports = resolve_path(cfg, "reports")
    df = pd.read_parquet(proc / f"dataset_{track}.parquet")

    sub_path = proc / "subhourly_scalars.parquet"
    if subhourly and sub_path.exists():
        n_before = len(df)
        sub = pd.read_parquet(sub_path)
        df = df.merge(sub, on="timestamp", how="left")
        assert len(df) == n_before, "sub-hourly join changed the row count"
        log.info("joined %d sub-hourly features (%.1f%% complete)",
                 len(SUBHOURLY_SCALARS),
                 100 * df[SUBHOURLY_SCALARS].notna().all(axis=1).mean())

    quantiles = cfg["quantiles"]
    bins = [tuple(b) for b in cfg["eval"]["cloud_bins"]]

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    log.info("track %s | train=%d val=%d test=%d", track, len(train), len(val), len(test))

    feat_grid = feature_columns(cfg, df, use_grid=True)
    feat_tab = feature_columns(cfg, df, use_grid=False)
    feat_sub = feature_columns(cfg, df, use_grid=True, use_subhourly=True)
    log.info("features: %d tabular, %d +grid, %d +sub-hourly",
             len(feat_tab), len(feat_grid), len(feat_sub))

    out: dict = {"track": track, "seed": seed, "n_test": int(len(test)),
                 "quantiles": quantiles, "horizons": cfg["horizons"], "models": {}}

    for h in cfg["horizons"]:
        y = test[f"ghi_t{h}h"].to_numpy()
        cs = test[f"cs_t{h}h"].to_numpy()
        kt_now = test["clearsky_ratio"].to_numpy()

        feats = {"tab": feat_tab, "grid": feat_grid, "sub": feat_sub}
        preds = {
            "persistence": persistence(train, test, h, quantiles),
            "smart_persistence": smart_persistence(train, test, h, quantiles),
        }
        specs = TREE_SPECS + (SUBHOURLY_SPECS if subhourly else ())
        for name, learner, fset, space in specs:
            preds[name] = LEARNERS[learner](train, val, test, h, quantiles,
                                            feats[fset], seed, cfg, space)
        # reference for skill: smart persistence POINT forecast
        ref_mae = metrics.mae(smart_persistence_point(test, h), y)

        for name, pq in preds.items():
            rep = metrics.full_report(pq, y, cs, kt_now, quantiles, bins, ref_mae=ref_mae)
            out["models"].setdefault(name, {})[f"t+{h}h"] = rep
        out.setdefault("reference", {})[f"t+{h}h"] = {"smart_persistence_point_mae": ref_mae}

    # seed 0 keeps the canonical filename so existing report references still resolve
    fname = f"baselines_{track}.json" if seed == 0 else f"baselines_{track}_seed{seed}.json"
    with open(reports / fname, "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def print_table(out: dict) -> None:
    q = out["quantiles"]
    nominal = max(q) - min(q)
    print()
    print(f"BASELINES - track {out['track']}   (test n={out['n_test']}, "
          f"central band = {nominal:.0%} nominal)")
    print("=" * 110)
    hdr = f"{'model':<22}{'horizon':<9}{'n':>7}{'MAE':>9}{'RMSE':>9}{'pinball':>9}" \
          f"{'CRPS':>9}{'cover':>8}{'skill':>9}{'sprd/real':>11}"
    print(hdr)
    print("-" * 110)
    for name, per_h in out["models"].items():
        for hz, r in per_h.items():
            print(f"{name:<22}{hz:<9}{r['n']:>7}{r['mae']:>9.1f}{r['rmse']:>9.1f}"
                  f"{r['pinball']:>9.2f}{r['crps']:>9.2f}{r['coverage']:>8.3f}"
                  f"{r.get('skill_vs_ref', float('nan')):>9.3f}"
                  f"{r['dispersion_spread_over_reality']:>11.2f}")
        print("-" * 110)
    print(f"skill = 1 - MAE/MAE(smart persistence point forecast); higher is better, "
          f"0 = no gain")
    print(f"cover = observed coverage of the {nominal:.0%} central band "
          f"(nominal {nominal:.2f}); sprd/real in k_t space, 1.00 is the goal")

    print()
    print(f"STRATIFIED CALIBRATION by current cloud regime - track {out['track']}")
    print("=" * 102)
    print(f"{'model':<22}{'horizon':<9}{'regime':<9}{'n':>7}{'cover':>8}"
          f"{'sprd/real':>11}{'MAE':>9}{'bias':>9}")
    print("-" * 102)
    for name, per_h in out["models"].items():
        for hz, r in per_h.items():
            for s in r["stratified"]:
                if s.get("n", 0) == 0:
                    print(f"{name:<22}{hz:<9}{s['regime']:<9}{0:>7}{'-':>8}{'-':>11}{'-':>9}{'-':>9}")
                    continue
                print(f"{name:<22}{hz:<9}{s['regime']:<9}{s['n']:>7}{s['coverage']:>8.3f}"
                      f"{s['spread_over_reality']:>11.2f}{s['mae']:>9.1f}{s['bias']:>9.1f}")
        print("-" * 102)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--track", default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true",
                    help="fit and persist the served model (CatBoost + k_t) to "
                         "artifacts/checkpoints, then exit")
    ap.add_argument("--subhourly", action="store_true",
                    help="also score the sub-hourly rain-history feature block "
                         "(see SUBHOURLY_SPECS; measured as no gain)")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.save:
        save_catboost(cfg, args.track, args.seed)
        return 0
    out = run(cfg, args.track, args.seed, args.subhourly)
    print_table(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
