"""
train_vision_fast.py
Fast vision-guided fine-tuning from .npz episode files.

Bypasses the LeRobot video-backend bottleneck by loading all .npz episodes
into RAM as numpy arrays, yielding (state, images, action) tuples directly.

Fine-tunes from checkpoints_diffusion_iter10/best_model using:
  - Image augmentation: color jitter + ±8px spatial shift
  - Very low LR (5e-6) to preserve motor skills, update visual encoder

Usage:
    python train_vision_fast.py [--steps 3000] [--npz_dir /tmp/aic_recordings_iter10_backup]
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ── Hyperparameters ────────────────────────────────────────────────────────────
FPS        = 20
N_OBS      = 2      # stack last 2 observations
HORIZON    = 16     # predict 16 future actions
N_ACTION   = 8      # execute first 8

IMG_H, IMG_W = 128, 144
STATE_DIM, ACTION_DIM = 26, 6


# ── In-memory dataset ──────────────────────────────────────────────────────────

class NpzEpisodeDataset(Dataset):
    """
    Loads all .npz episodes into RAM.  Returns (state_2step, images_2step, action_horizon).

    Each .npz file has arrays:
        states   (T, 26) float32
        actions  (T, 6)  float32
        center   (T, H, W, 3) uint8
        left     (T, H, W, 3) uint8
        right    (T, H, W, 3) uint8
    """

    def __init__(self, npz_dir: Path, augment: bool = False):
        self.augment = augment
        self.episodes = []
        files = sorted(npz_dir.glob("ep_*.npz"))
        if not files:
            raise FileNotFoundError(f"No ep_*.npz files in {npz_dir}")

        print(f"Loading {len(files)} .npz episodes into RAM …")
        for f in files:
            d = np.load(str(f))
            if "states" not in d or len(d["states"]) < HORIZON + N_OBS:
                continue
            self.episodes.append({
                "states":  d["states"].astype(np.float32),   # (T, 26)
                "actions": d["actions"].astype(np.float32),  # (T, 6)
                "center":  d["center"],   # (T, H, W, 3) uint8
                "left":    d["left"],
                "right":   d["right"],
            })
        print(f"Loaded {len(self.episodes)} episodes.")

        # Pre-compute valid sample indices: (ep_idx, frame_idx)
        # frame_idx must have N_OBS-1 history and HORIZON future
        self.indices = []
        for ei, ep in enumerate(self.episodes):
            T = len(ep["states"])
            for t in range(N_OBS - 1, T - HORIZON + 1):
                self.indices.append((ei, t))
        print(f"Total samples: {len(self.indices):,}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ei, t = self.indices[idx]
        ep = self.episodes[ei]

        # ── State: stack [t-1, t] ─────────────────────────────────────────────
        state_2 = np.stack([ep["states"][t - 1], ep["states"][t]], axis=0)  # (2, 26)

        # ── Images: stack [t-1, t], convert uint8 → float32 [0,1] ────────────
        def img2tensor(arr, ts):
            frames = [arr[s].astype(np.float32) / 255.0 for s in ts]  # (H,W,3)
            return np.stack(frames, axis=0)  # (2, H, W, 3)

        ts = [t - 1, t]
        center_2 = img2tensor(ep["center"], ts)
        left_2   = img2tensor(ep["left"],   ts)
        right_2  = img2tensor(ep["right"],  ts)

        # ── Actions: next HORIZON steps ───────────────────────────────────────
        actions = ep["actions"][t: t + HORIZON]  # (HORIZON, 6)

        # Return (H,W,C) → (C,H,W) for camera tensors
        def hwc2chw(arr):  # (2, H, W, 3) → (2, 3, H, W)
            return arr.transpose(0, 3, 1, 2)

        return {
            "observation.state":                  torch.from_numpy(state_2),       # (2, 26)
            "observation.images.center_camera":   torch.from_numpy(hwc2chw(center_2)),  # (2,3,H,W)
            "observation.images.left_camera":     torch.from_numpy(hwc2chw(left_2)),
            "observation.images.right_camera":    torch.from_numpy(hwc2chw(right_2)),
            "action":                             torch.from_numpy(actions),        # (HORIZON, 6)
            # All action steps are valid (no padding in our fixed-horizon slices)
            "action_is_pad":                      torch.zeros(HORIZON, dtype=torch.bool),
        }


# ── Image augmentation ─────────────────────────────────────────────────────────

_CAM_KEYS = [
    "observation.images.center_camera",
    "observation.images.left_camera",
    "observation.images.right_camera",
]


def augment_batch(batch: dict) -> dict:
    """Color jitter + ±8px spatial shift on camera tensors (B, 2, 3, H, W)."""
    for key in _CAM_KEYS:
        if key not in batch:
            continue
        imgs = batch[key]          # (B, 2, 3, H, W)
        B, T, C, H, W = imgs.shape
        flat = imgs.reshape(B * T, C, H, W)

        # Brightness jitter
        if random.random() < 0.8:
            fac = torch.empty(B * T, 1, 1, 1, device=flat.device).uniform_(0.75, 1.25)
            flat = (flat * fac).clamp(0, 1)
        # Contrast jitter
        if random.random() < 0.5:
            mean = flat.mean(dim=(-3, -2, -1), keepdim=True)
            fac = random.uniform(0.8, 1.2)
            flat = (mean + (flat - mean) * fac).clamp(0, 1)
        # Spatial shift (same crop for whole batch)
        pad = 8
        flat = F.pad(flat, [pad, pad, pad, pad], mode="reflect")
        top  = random.randint(0, 2 * pad)
        left = random.randint(0, 2 * pad)
        flat = flat[:, :, top: top + H, left: left + W]

        batch[key] = flat.reshape(B, T, C, H, W)
    return batch


# ── Trainer ────────────────────────────────────────────────────────────────────

class VisionFineTuner:
    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # ── Dataset ──────────────────────────────────────────────────────────
        npz_dir = Path(cfg["npz_dir"])
        self.augment = cfg.get("augment", True)
        ds = NpzEpisodeDataset(npz_dir, augment=self.augment)
        n_val  = max(1, len(ds) // 8)
        n_train = len(ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"],
            shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"],
            shuffle=False, num_workers=0, pin_memory=True,
        )
        print(f"Train: {n_train:,}  |  Val: {n_val:,}")

        # ── Model: load pretrained, then fine-tune ────────────────────────────
        ft_ckpt = Path(cfg["finetune_from"])
        print(f"Loading pretrained checkpoint: {ft_ckpt}")
        self.model = DiffusionPolicy.from_pretrained(str(ft_ckpt))
        self.model.to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"DiffusionPolicy parameters: {n_params:,}")

        # ── Normalization from reference dataset (iter10 for min/max stats) ───
        ref_id = cfg.get("ref_dataset", "local/aic_cable_insertion_iter10")
        print(f"Loading normalization stats from: {ref_id}")
        ref_ds = LeRobotDataset(ref_id)
        self.preprocessor, _ = make_pre_post_processors(
            self.model.config, dataset_stats=ref_ds.meta.stats
        )
        print("Preprocessor ready.")
        if self.augment:
            print("Image augmentation: ENABLED (color jitter + spatial shift ±8px)")

        # ── Optimiser ────────────────────────────────────────────────────────
        self.optimizer = AdamW(
            self.model.parameters(), lr=cfg["lr"],
            betas=(0.95, 0.999), eps=1e-8, weight_decay=cfg["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg["steps"], eta_min=cfg["lr"] * 0.01
        )

        # ── Output dirs ───────────────────────────────────────────────────────
        tag = cfg.get("run_name", "vision_fast")
        base = Path(__file__).parent
        self.out_dir  = base / f"outputs_diffusion_{tag}"
        self.ckpt_dir = base / f"checkpoints_diffusion_{tag}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[dict] = []
        self.best_val   = float("inf")
        self.global_step = 0

        json.dump({"config": cfg, "n_train": n_train, "n_val": n_val},
                  open(self.out_dir / "run_config.json", "w"), indent=2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_device(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss = n = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = self._to_device(batch)
                if train and self.augment:
                    batch = augment_batch(batch)
                batch = self.preprocessor(batch)

                out  = self.model(batch)
                loss = out[0] if isinstance(out, tuple) else out

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.global_step += 1
                    if self.global_step >= self.cfg["steps"]:
                        break

                B = next(v.size(0) for v in batch.values() if isinstance(v, torch.Tensor))
                total_loss += loss.item() * B
                n += B

        return {"loss": total_loss / max(n, 1)}

    def train(self):
        steps_per_epoch = len(self.train_loader)
        n_epochs = max(1, -(-self.cfg["steps"] // steps_per_epoch))   # ceil div
        print(f"\nTraining {n_epochs} epochs ({self.cfg['steps']} steps, "
              f"{steps_per_epoch} steps/epoch)\n")
        print(f"{'Epoch':>6}  {'LR':>9}  {'T-loss':>9}  {'V-loss':>9}  {'Best':>5}")
        print("─" * 55)

        for epoch in range(1, n_epochs + 1):
            t0  = time.time()
            t_m = self._run_epoch(self.train_loader, train=True)
            v_m = self._run_epoch(self.val_loader,   train=False)
            lr  = self.scheduler.get_last_lr()[0]

            is_best = v_m["loss"] < self.best_val
            if is_best:
                self.best_val = v_m["loss"]
                self.model.save_pretrained(str(self.ckpt_dir / "best_model"))

            if epoch % self.cfg.get("save_every", 5) == 0:
                self.model.save_pretrained(str(self.ckpt_dir / f"ckpt_epoch_{epoch:03d}"))

            self.history.append({
                "epoch": epoch, "step": self.global_step, "lr": lr,
                "train_loss": t_m["loss"], "val_loss": v_m["loss"],
            })
            self._save_metrics()
            print(f"{epoch:>6}  {lr:>9.2e}  {t_m['loss']:>9.5f}  "
                  f"{v_m['loss']:>9.5f}  {'✓' if is_best else '':>5}   "
                  f"{time.time()-t0:.1f}s")

            if self.global_step >= self.cfg["steps"]:
                break

        self.model.save_pretrained(str(self.ckpt_dir / "final_model"))
        print(f"\nDone. Best val loss: {self.best_val:.5f}")
        print(f"Checkpoint: {self.ckpt_dir / 'best_model'}")

    def _save_metrics(self):
        with open(self.out_dir / "metrics.json", "w") as f:
            json.dump(self.history, f, indent=2)
        if self.history:
            with open(self.out_dir / "metrics.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
                w.writeheader()
                w.writerows(self.history)


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fast vision-guided fine-tuning from .npz files")
    p.add_argument("--npz_dir",       default="/tmp/aic_recordings_iter10_backup",
                   help="Directory containing ep_*.npz episode files")
    p.add_argument("--finetune_from", default="checkpoints_diffusion_iter10/best_model",
                   help="Pretrained DiffusionPolicy checkpoint path")
    p.add_argument("--ref_dataset",   default="local/aic_cable_insertion_iter10",
                   help="Dataset whose stats to use for normalization")
    p.add_argument("--run_name",      default="vision_fast")
    p.add_argument("--steps",         type=int,   default=3000)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=5e-6)
    p.add_argument("--weight_decay",  type=float, default=1e-6)
    p.add_argument("--save_every",    type=int,   default=1)
    p.add_argument("--augment",       action="store_true", default=True)
    p.add_argument("--no_augment",    action="store_false", dest="augment")
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    # Resolve relative path for finetune_from
    ft = Path(cfg["finetune_from"])
    if not ft.is_absolute():
        cfg["finetune_from"] = str(Path(__file__).parent / ft)
    trainer = VisionFineTuner(cfg)
    trainer.train()
