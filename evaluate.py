"""Evaluate checkpoints and baselines on IDENTICAL rows, with honest diagnostics.

Reports, per horizon:
  * MAE / RMSE / bias, pinball, CRPS
  * coverage of the central band, against the level implied by the QUANTILES
    (with [0.1 ... 0.9] that is an 80% band, not 90% - see metrics.py)
  * stratified coverage and spread/reality IN k_t SPACE by cloud regime
  * skill score vs smart persistence
  * the gate's alpha distribution, with a collapse warning
  * ablation across variants x seeds, with mean +/- std over seeds

Every metric is printed with its sample size. Models are scored on the same rows.

Usage:
    python evaluate.py --track A                 # everything found for this track
    python evaluate.py --track A --baselines     # baselines only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# baselines imports lightgbm, which MUST load before torch on macOS or LightGBM
# segfaults when it starts its OpenMP pool. Keep these two lines in this order.
import baselines as bl  # noqa: E402
import metrics  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from data._client import load_config, log, resolve_path, setup_logging  # noqa: E402
from data.dataset import make_datasets  # noqa: E402
from models.fusion import MultimodalNowcaster  # noqa: E402
from train import collate, pick_device  # noqa: E402


@torch.no_grad()
def predict_dataset(model, ds, cfg, device, batch_size: int = 128) -> dict:
    """-> {pred (N,H,Q), y (N,H), cs (N,H), kt_now (N,), alpha (N,) or None}."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    P, Y, CS, KT, A = [], [], [], [], []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        pred = model(b)
        hi = b["future_cs_raw"].unsqueeze(-1) * 1.25
        pred = torch.clamp(pred, min=0.0).minimum(hi)
        P.append(pred.cpu().numpy())
        Y.append(batch["y"].numpy())
        CS.append(batch["future_cs_raw"].numpy())
        KT.append(batch["kt_now"].numpy())
        if getattr(model, "last_alpha", None) is not None:
            A.append(model.last_alpha.flatten().cpu().numpy())
    return {
        "pred": np.concatenate(P), "y": np.concatenate(Y),
        "cs": np.concatenate(CS), "kt_now": np.concatenate(KT),
        "alpha": np.concatenate(A) if A else None,
    }


def load_checkpoint(path: Path, cfg: dict, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = MultimodalNowcaster(
        n_features=ck["n_features"], n_grid_channels=ck["n_grid_channels"],
        n_horizons=len(ck["horizons"]), n_quantiles=len(ck["quantiles"]),
        n_nwp=ck["n_nwp"], variant=ck["variant"], cfg=ck.get("config", cfg),
        target_space=ck.get("target_space", "ghi"),
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, ck


def gate_report(alpha: np.ndarray | None, threshold: float) -> dict:
    if alpha is None or len(alpha) == 0:
        return {"available": False}
    rng = float(alpha.max() - alpha.min())
    return {
        "available": True, "n": int(len(alpha)),
        "min": float(alpha.min()), "max": float(alpha.max()),
        "mean": float(alpha.mean()), "std": float(alpha.std()),
        "p10": float(np.percentile(alpha, 10)), "p50": float(np.percentile(alpha, 50)),
        "p90": float(np.percentile(alpha, 90)),
        "range": rng,
        "collapsed": bool(rng < threshold),
    }


def evaluate_checkpoint(cfg: dict, path: Path, track: str, device) -> dict:
    model, ck = load_checkpoint(path, cfg, device)
    ds = make_datasets(cfg, track, variant=ck["variant"])["test"]
    out = predict_dataset(model, ds, cfg, device)
    quantiles = ck["quantiles"]
    bins = [tuple(b) for b in cfg["eval"]["cloud_bins"]]

    rows = ds.df
    per_h = {}
    for k, h in enumerate(ck["horizons"]):
        ref_mae = metrics.mae(bl.smart_persistence_point(rows, h),
                              rows[f"ghi_t{h}h"].to_numpy())
        per_h[f"t+{h}h"] = metrics.full_report(
            out["pred"][:, k, :], out["y"][:, k], out["cs"][:, k],
            out["kt_now"], quantiles, bins, ref_mae=ref_mae)
    return {
        # kt runs are reported as a separate variant so the seed aggregation in
        # ablation_table() never averages the two target spaces together
        "checkpoint": path.name,
        "variant": ck["variant"] + ("_kt" if ck.get("target_space") == "kt" else ""),
        "seed": ck["seed"],
        "track": track, "n_test": int(len(ds)), "params": ck["params"]["total"],
        "trained_at": ck.get("trained_at"),
        "best_val_pinball": ck.get("best_val_pinball"),
        "horizons": per_h,
        "gate": gate_report(out["alpha"], cfg["eval"]["gate_collapse_threshold"]),
    }


def print_model_table(results: list[dict], cfg: dict) -> None:
    q = cfg["quantiles"]
    nominal = max(q) - min(q)
    print()
    print(f"MODEL RESULTS   (central band = {nominal:.0%} nominal)")
    print("=" * 112)
    print(f"{'variant':<11}{'seed':>5}{'horizon':<9}{'n':>7}{'MAE':>9}{'RMSE':>9}"
          f"{'pinball':>9}{'CRPS':>9}{'cover':>8}{'skill':>9}{'sprd/real':>11}")
    print("-" * 112)
    for r in results:
        for hz, m in r["horizons"].items():
            print(f"{r['variant']:<11}{r['seed']:>5}{hz:<9}{m['n']:>7}{m['mae']:>9.1f}"
                  f"{m['rmse']:>9.1f}{m['pinball']:>9.2f}{m['crps']:>9.2f}"
                  f"{m['coverage']:>8.3f}{m.get('skill_vs_ref', float('nan')):>9.3f}"
                  f"{m['dispersion_spread_over_reality']:>11.2f}")
    print("-" * 112)


def print_ablation(results: list[dict], cfg: dict) -> None:
    """Mean +/- std over seeds. Never report a comparison from a single seed."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        for hz, m in r["horizons"].items():
            agg[r["variant"]][hz].append(m)
    seeds = sorted({r["seed"] for r in results})
    print()
    print(f"ABLATION over {len(seeds)} seeds {seeds}   (mean +/- std)")
    print("=" * 104)
    print(f"{'variant':<11}{'horizon':<9}{'runs':>6}{'pinball':>18}{'MAE':>18}"
          f"{'skill':>16}{'sprd/real':>16}")
    print("-" * 104)
    for variant, per_h in agg.items():
        for hz, ms in per_h.items():
            def ms_(key):
                v = np.array([m[key] for m in ms if m.get(key) is not None], dtype=float)
                return f"{v.mean():8.2f} +/-{v.std():5.2f}" if len(v) else " " * 15
            print(f"{variant:<11}{hz:<9}{len(ms):>6}{ms_('pinball'):>18}{ms_('mae'):>18}"
                  f"{ms_('skill_vs_ref'):>16}{ms_('dispersion_spread_over_reality'):>16}")
        print("-" * 104)
    if len(seeds) < 3:
        print(f"WARNING: only {len(seeds)} seed(s). The prior project's ablation "
              f"ordering INVERTED between two single-seed runs. Do not report an "
              f"ordering from fewer than 3 seeds.")


def print_stratified(results: list[dict], cfg: dict) -> None:
    q = cfg["quantiles"]
    nominal = max(q) - min(q)
    print()
    print(f"STRATIFIED CALIBRATION by current cloud regime   "
          f"(coverage target {nominal:.2f}, spread/reality target 1.00)")
    print("=" * 104)
    print(f"{'variant':<11}{'seed':>5}{'horizon':<9}{'regime':<9}{'n':>7}{'cover':>8}"
          f"{'sprd/real':>11}{'MAE':>9}{'bias':>9}   flag")
    print("-" * 104)
    for r in results:
        for hz, m in r["horizons"].items():
            for s in m["stratified"]:
                if s.get("n", 0) == 0:
                    continue
                sr = s["spread_over_reality"]
                flag = "HEDGING" if sr < 0.70 else ("wide" if sr > 1.4 else "")
                print(f"{r['variant']:<11}{r['seed']:>5}{hz:<9}{s['regime']:<9}{s['n']:>7}"
                      f"{s['coverage']:>8.3f}{sr:>11.2f}{s['mae']:>9.1f}{s['bias']:>9.1f}"
                      f"   {flag}")
    print("-" * 104)
    print("HEDGING = spread/reality < 0.70: the model is collapsing toward the mean "
          "in that regime.")


def print_gates(results: list[dict], cfg: dict) -> None:
    gated = [r for r in results if r["gate"].get("available")]
    if not gated:
        return
    thr = cfg["eval"]["gate_collapse_threshold"]
    print()
    print(f"PHYSICS GATE alpha   (collapse threshold: range < {thr:.2f} of [0,1])")
    print("=" * 96)
    print(f"{'variant':<11}{'seed':>5}{'n':>8}{'min':>8}{'p10':>8}{'p50':>8}{'p90':>8}"
          f"{'max':>8}{'range':>9}   status")
    print("-" * 96)
    for r in gated:
        g = r["gate"]
        print(f"{r['variant']:<11}{r['seed']:>5}{g['n']:>8}{g['min']:>8.3f}{g['p10']:>8.3f}"
              f"{g['p50']:>8.3f}{g['p90']:>8.3f}{g['max']:>8.3f}{g['range']:>9.3f}"
              f"   {'GATE COLLAPSED' if g['collapsed'] else 'ok'}")
    print("-" * 96)
    if any(r["gate"]["collapsed"] for r in gated):
        print("GATE COLLAPSED: alpha spans under the threshold, so the gate is "
              "effectively a constant and the model is single-branch in practice. "
              "This is reported, not worked around.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--track", default="A")
    ap.add_argument("--baselines", action="store_true", help="baselines only")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--checkpoint", default=None, help="evaluate a single checkpoint")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    reports = resolve_path(cfg, "reports")
    ck_dir = resolve_path(cfg, "checkpoints")
    device = pick_device(cfg["train"].get("device", "auto"))

    payload: dict = {"track": args.track}

    if not args.no_baselines:
        base = bl.run(cfg, args.track)
        bl.print_table(base)
        payload["baselines"] = base
    if args.baselines:
        with open(reports / f"evaluation_{args.track}.json", "w") as f:
            json.dump(payload, f, indent=2, default=float)
        return 0

    paths = ([Path(args.checkpoint)] if args.checkpoint
             else sorted(ck_dir.glob(f"{args.track}_*.pt")))
    if not paths:
        log.warning("no checkpoints found for track %s", args.track)
        with open(reports / f"evaluation_{args.track}.json", "w") as f:
            json.dump(payload, f, indent=2, default=float)
        return 0

    results = []
    for p in paths:
        try:
            results.append(evaluate_checkpoint(cfg, p, args.track, device))
            log.info("evaluated %s", p.name)
        except Exception as e:
            log.error("failed on %s: %s", p.name, str(e)[:200])
    if results:
        print_model_table(results, cfg)
        print_ablation(results, cfg)
        print_stratified(results, cfg)
        print_gates(results, cfg)
        payload["models"] = results

    with open(reports / f"evaluation_{args.track}.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)
    log.info("wrote %s", reports / f"evaluation_{args.track}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
