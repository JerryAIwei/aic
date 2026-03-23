"""
train.py
Training loop for ACT policy with full metrics tracking.

Features
────────
• Normalises observations and actions using training-set statistics
• Cosine-annealing LR schedule with linear warm-up
• Gradient clipping
• Per-epoch + per-batch loss logging to JSON and CSV
• Periodic checkpoint saves (best val + every N epochs)
• Progress printed to stdout

Usage
─────
    python train.py [--epochs 50] [--batch_size 32] [--lr 1e-4] [--beta 10]
"""

import argparse
import json
import csv
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from act_model import ACTPolicy

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "data"
OUT_DIR  = BASE / "outputs"
CKPT_DIR = BASE / "checkpoints"


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ACTDataset(Dataset):
    """
    Loads pre-generated episodes and builds (obs, image, action_chunk) samples.
    Each sample corresponds to one timestep; the action_chunk is the window of
    `chunk_size` future actions starting at that timestep.
    """

    def __init__(self, split: str, metadata: dict, chunk_size: int = 10):
        self.chunk_size = chunk_size
        self.obs_mean   = torch.tensor(metadata["obs_mean"],    dtype=torch.float32)
        self.obs_std    = torch.tensor(metadata["obs_std"],     dtype=torch.float32)
        self.act_mean   = torch.tensor(metadata["action_mean"], dtype=torch.float32)
        self.act_std    = torch.tensor(metadata["action_std"],  dtype=torch.float32)

        ep_dir = DATA_DIR / split
        files  = sorted(ep_dir.glob("*.npz"))

        self.samples = []   # list of (obs_t, image_t, action_chunk)
        for f in files:
            d    = np.load(f)
            obs  = torch.from_numpy(d["obs"])      # (T, 13)
            acts = torch.from_numpy(d["actions"])  # (T, 7)
            imgs = torch.from_numpy(d["images"])   # (T, H, W, 3)
            T    = len(obs)
            for t in range(T - chunk_size):
                img_t   = imgs[t].permute(2, 0, 1)        # (3, H, W)
                chunk   = acts[t : t + chunk_size]         # (chunk_size, 7)
                self.samples.append((obs[t], img_t, chunk))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_raw, img, act_raw = self.samples[idx]
        obs_norm = (obs_raw - self.obs_mean) / self.obs_std
        act_norm = (act_raw - self.act_mean) / self.act_std
        return obs_norm, img, act_norm


# ─── LR schedule: linear warm-up + cosine decay ──────────────────────────────

def build_scheduler(optimizer: torch.optim.Optimizer,
                    warmup_epochs: int, total_epochs: int) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ─── Trainer ─────────────────────────────────────────────────────────────────

class ACTTrainer:
    def __init__(self, cfg: dict):
        self.cfg  = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}\n")

        with open(DATA_DIR / "metadata.json") as f:
            self.meta = json.load(f)

        chunk_size = self.meta["chunk_size"]

        # Datasets & loaders
        train_ds = ACTDataset("train", self.meta, chunk_size)
        val_ds   = ACTDataset("val",   self.meta, chunk_size)
        self.train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                       shuffle=True,  num_workers=2, pin_memory=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                                       shuffle=False, num_workers=2, pin_memory=True)

        print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

        # Model
        self.model = ACTPolicy(
            obs_dim        = self.meta["obs_dim"],
            action_dim     = self.meta["action_dim"],
            img_size       = self.meta["img_size"],
            chunk_size     = chunk_size,
            hidden_dim     = cfg["hidden_dim"],
            latent_dim     = cfg["latent_dim"],
            nhead          = cfg["nhead"],
            num_enc_layers = cfg["num_enc_layers"],
            num_dec_layers = cfg["num_dec_layers"],
            beta           = cfg["beta"],
        ).to(self.device)

        print(f"Model parameters: {self.model.count_parameters():,}\n")

        self.optimizer = AdamW(self.model.parameters(), lr=cfg["lr"],
                               weight_decay=cfg["weight_decay"])
        self.scheduler = build_scheduler(self.optimizer,
                                         warmup_epochs=cfg["warmup_epochs"],
                                         total_epochs=cfg["epochs"])

        # Metrics storage
        self.history: dict[str, list] = {
            "epoch": [], "lr": [],
            "train_total": [], "train_l1": [], "train_kl": [],
            "val_total":   [], "val_l1":   [], "val_kl":   [],
        }
        self.batch_log: list[dict] = []
        self.best_val  = float("inf")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Single epoch helpers ──────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        totals = {"total": 0.0, "l1": 0.0, "kl": 0.0}
        n = 0
        with torch.set_grad_enabled(train):
            for obs, img, actions in loader:
                obs, img, actions = (obs.to(self.device),
                                     img.to(self.device),
                                     actions.to(self.device))

                pred, mu, lv = self.model(obs, img, actions)
                losses = self.model.compute_loss(pred, actions, mu, lv)

                if train:
                    self.optimizer.zero_grad()
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.cfg["grad_clip"])
                    self.optimizer.step()

                B = obs.size(0)
                for k in totals:
                    totals[k] += losses[k].item() * B
                n += B

        return {k: v / n for k, v in totals.items()}

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        cfg   = self.cfg
        print(f"{'Epoch':>6}  {'LR':>10}  {'T-total':>10}  {'T-l1':>8}  "
              f"{'T-kl':>8}  {'V-total':>10}  {'V-l1':>8}  {'V-kl':>8}  {'Best':>6}")
        print("─" * 90)

        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()

            train_m = self._run_epoch(self.train_loader, train=True)
            val_m   = self._run_epoch(self.val_loader,   train=False)
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            # Record
            self.history["epoch"].append(epoch)
            self.history["lr"].append(lr)
            for k in ("total", "l1", "kl"):
                self.history[f"train_{k}"].append(train_m[k])
                self.history[f"val_{k}"].append(val_m[k])

            is_best = val_m["total"] < self.best_val
            if is_best:
                self.best_val = val_m["total"]
                torch.save(self.model.state_dict(), CKPT_DIR / "best_model.pt")

            if epoch % cfg["save_every"] == 0:
                torch.save(self.model.state_dict(),
                           CKPT_DIR / f"checkpoint_epoch_{epoch:03d}.pt")

            # Flush metrics to disk every epoch so a kill doesn't lose history
            self._save_metrics()

            elapsed = time.time() - t0
            print(f"{epoch:>6}  {lr:>10.2e}  {train_m['total']:>10.5f}  "
                  f"{train_m['l1']:>8.5f}  {train_m['kl']:>8.5f}  "
                  f"{val_m['total']:>10.5f}  {val_m['l1']:>8.5f}  {val_m['kl']:>8.5f}  "
                  f"{'✓' if is_best else '':>6}   {elapsed:.1f}s")

        torch.save(self.model.state_dict(), CKPT_DIR / "final_model.pt")
        self._save_metrics()
        print(f"\nTraining complete.  Best val loss: {self.best_val:.5f}")
        print(f"Artifacts saved to {OUT_DIR}  and  {CKPT_DIR}")

    # ── Persist metrics ───────────────────────────────────────────────────────

    def _save_metrics(self):
        # Full JSON history
        with open(OUT_DIR / "metrics.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # CSV for quick import
        with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history.keys()))
            writer.writeheader()
            for i in range(len(self.history["epoch"])):
                writer.writerow({k: self.history[k][i] for k in self.history})

        # Save training config alongside metrics
        config_out = {
            "model": {
                "obs_dim":        self.meta["obs_dim"],
                "action_dim":     self.meta["action_dim"],
                "img_size":       self.meta["img_size"],
                "chunk_size":     self.meta["chunk_size"],
                "hidden_dim":     self.cfg["hidden_dim"],
                "latent_dim":     self.cfg["latent_dim"],
                "nhead":          self.cfg["nhead"],
                "num_enc_layers": self.cfg["num_enc_layers"],
                "num_dec_layers": self.cfg["num_dec_layers"],
                "n_params":       self.model.count_parameters(),
            },
            "training": {k: self.cfg[k] for k in (
                "epochs", "batch_size", "lr", "weight_decay",
                "beta", "grad_clip", "warmup_epochs", "save_every"
            )},
            "dataset": self.meta,
            "best_val_loss":  self.best_val,
            "final_train_loss": self.history["train_total"][-1],
            "final_val_loss":   self.history["val_total"][-1],
        }
        with open(OUT_DIR / "run_config.json", "w") as f:
            json.dump(config_out, f, indent=2)


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train ACT policy")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--beta",          type=float, default=10.0,
                   help="KL weight in CVAE loss")
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--save_every",    type=int,   default=10)
    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--latent_dim",    type=int,   default=32)
    p.add_argument("--nhead",         type=int,   default=8)
    p.add_argument("--num_enc_layers",type=int,   default=4)
    p.add_argument("--num_dec_layers",type=int,   default=4)
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    trainer = ACTTrainer(cfg)
    trainer.train()
