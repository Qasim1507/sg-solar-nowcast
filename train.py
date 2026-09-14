"""Train the multimodal quantile nowcaster.

  * pinball loss on sorted quantiles (never Gaussian NLL - see models/head.py)
  * NaN / dropped batch counts are LOGGED, never silently skipped
  * >= 3 seeds before any variant comparison is reported: in the prior project the
    ablation ordering inverted between two runs on a single seed
  * the seed is fixed and recorded in the checkpoint alongside split sizes

Usage:
    python train.py --track A --variant gated --seed 0
    python train.py --track A --ablation                # all variants x all seeds
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data._client import load_config, log, resolve_path, setup_logging  # noqa: E402
from data.dataset import make_datasets  # noqa: E402
from models.fusion import VARIANTS, MultimodalNowcaster  # noqa: E402
from models.head import pinball_loss  # noqa: E402


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch])
    return out


def run_epoch(model, loader, quantiles, device, opt=None, cs_clamp=True) -> dict:
    train = opt is not None
    model.train(train)
    total, n_batches = 0.0, 0
    n_nan_in, n_nan_out, n_dropped, n_seen = 0, 0, 0, 0
    alphas = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["y"]

        # ---- input hygiene: count and drop, never pass silently
        finite = torch.isfinite(batch["seq"]).all(dim=(1, 2)) & torch.isfinite(y).all(dim=1)
        finite &= torch.isfinite(batch["grid"]).all(dim=(1, 2, 3))
        if batch["nwp"].shape[-1] > 0:
            finite &= torch.isfinite(batch["nwp"]).all(dim=1)
        n_seen += y.shape[0]
        if not finite.all():
            n_nan_in += int((~finite).sum())
            if finite.sum() == 0:
                n_dropped += y.shape[0]
                continue
            batch = {k: v[finite] for k, v in batch.items()}
            y = batch["y"]

        with torch.set_grad_enabled(train):
            pred = model(batch)
            if cs_clamp:
                # physics: GHI cannot exceed clear-sky by much, nor be negative
                hi = batch["future_cs_raw"].unsqueeze(-1) * 1.25
                pred = torch.clamp(pred, min=0.0).minimum(hi)
            loss = pinball_loss(pred, y, quantiles)

            if not torch.isfinite(loss):
                n_nan_out += 1
                continue
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        if getattr(model, "last_alpha", None) is not None:
            alphas.append(model.last_alpha.flatten().cpu())
        total += float(loss)
        n_batches += 1

    return {
        "loss": total / max(n_batches, 1),
        "n_batches": n_batches,
        "n_seen": n_seen,
        "n_nan_inputs": n_nan_in,
        "n_nan_losses": n_nan_out,
        "n_dropped": n_dropped,
        "alpha": torch.cat(alphas).numpy() if alphas else None,
    }


def train_one(cfg: dict, track: str, variant: str, seed: int,
              epochs: int | None = None, target_space: str = "ghi") -> dict:
    set_seed(seed)
    device = pick_device(cfg["train"].get("device", "auto"))
    ds = make_datasets(cfg, track, variant=variant)
    quantiles = cfg["quantiles"]
    tcfg = cfg["train"]
    epochs = epochs or tcfg["epochs"]

    loaders = {
        k: DataLoader(v, batch_size=tcfg["batch_size"], shuffle=(k == "train"),
                      collate_fn=collate, num_workers=0, drop_last=False)
        for k, v in ds.items()
    }
    sample = ds["train"][0]
    model = MultimodalNowcaster(
        n_features=sample["seq"].shape[-1],
        n_grid_channels=sample["grid"].shape[0],
        n_horizons=len(cfg["horizons"]),
        n_quantiles=len(quantiles),
        n_nwp=int(sample["nwp"].shape[-1]),
        target_space=target_space,
        variant=variant, cfg=cfg,
    ).to(device)
    params = model.n_params()
    n_train = len(ds["train"])
    log.info("variant=%s[%s] seed=%d device=%s | params total=%s spatial=%s | "
             "train=%d val=%d test=%d | params/sample=%.1f",
             variant, target_space, seed, device,
             f"{params['total']:,}", f"{params['spatial']:,}",
             n_train, len(ds["val"]), len(ds["test"]), params["total"] / max(n_train, 1))

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    best, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        tr = run_epoch(model, loaders["train"], quantiles, device, opt)
        va = run_epoch(model, loaders["val"], quantiles, device, None)
        sched.step(va["loss"])
        history.append({"epoch": ep, "train_loss": tr["loss"], "val_loss": va["loss"],
                        "n_nan_inputs": tr["n_nan_inputs"], "n_dropped": tr["n_dropped"]})
        if tr["n_nan_inputs"] or tr["n_nan_losses"] or tr["n_dropped"]:
            log.warning("  ep%02d NaN/dropped: inputs=%d losses=%d dropped_batches=%d",
                        ep, tr["n_nan_inputs"], tr["n_nan_losses"], tr["n_dropped"])
        if va["loss"] < best - 1e-6:
            best, bad = va["loss"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            log.info("  ep%02d train=%.3f val=%.3f%s", ep, tr["loss"], va["loss"],
                     "  *" if bad == 0 else "")
        if bad >= tcfg["patience"]:
            log.info("  early stop at epoch %d (best val %.4f)", ep, best)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    ck_dir = resolve_path(cfg, "checkpoints")
    # k_t runs get their own filename so the existing ghi-space checkpoints that
    # back the current REPORT.md stay on disk and both appear in evaluate.py
    suffix = "_kt" if target_space == "kt" else ""
    name = f"{track}_{variant}{suffix}_seed{seed}"
    payload = {
        "state_dict": model.state_dict(),
        "config": cfg,
        "variant": variant,
        "target_space": target_space,
        "track": track,
        "seed": seed,
        "quantiles": quantiles,
        "horizons": cfg["horizons"],
        "n_features": int(sample["seq"].shape[-1]),
        "n_grid_channels": int(sample["grid"].shape[0]),
        "n_nwp": int(sample["nwp"].shape[-1]),
        "n_train": n_train, "n_val": len(ds["val"]), "n_test": len(ds["test"]),
        "best_val_pinball": best,
        "params": params,
        "history": history,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
    }
    torch.save(payload, ck_dir / f"{name}.pt")
    log.info("  saved %s.pt  best_val_pinball=%.4f  (%.0fs)", name, best, time.time() - t0)
    return {"variant": variant, "target_space": target_space,
            "seed": seed, "best_val_pinball": best,
            "checkpoint": f"{name}.pt", "params": params["total"],
            "n_train": n_train, "n_test": len(ds["test"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--track", default="A")
    ap.add_argument("--variant", default="gated", choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--target-space", default="ghi", choices=["ghi", "kt"],
                    help="'kt': head emits clear-sky index, scaled by clear-sky at "
                         "the target hour. Loss stays in W/m2 either way.")
    ap.add_argument("--ablation", action="store_true",
                    help="train every variant across every configured seed")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    reports = resolve_path(cfg, "reports")

    if args.ablation:
        seeds = cfg["train"]["seeds"]
        if len(seeds) < 3:
            log.warning("only %d seeds configured; the spec asks for >= 3 before "
                        "reporting any variant comparison", len(seeds))
        results = []
        for variant in VARIANTS:
            for seed in seeds:
                results.append(train_one(cfg, args.track, variant, seed,
                                         args.epochs, args.target_space))
        sfx = "_kt" if args.target_space == "kt" else ""
        with open(reports / f"ablation_train_{args.track}{sfx}.json", "w") as f:
            json.dump(results, f, indent=2)
        log.info("ablation complete: %d runs", len(results))
    else:
        seed = args.seed if args.seed is not None else cfg["train"]["seeds"][0]
        train_one(cfg, args.track, args.variant, seed, args.epochs, args.target_space)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
