"""
train_act.py
Trains an ACT (Action Chunking with Transformers) policy on the large AIC
cable-insertion dataset using LeRobot.

Reference: https://github.com/huggingface/lerobot

Architecture
────────────
  Vision encoder  : ResNet-18 (× 3 cameras) → spatial softmax features
  State encoder   : Linear(26 → dim_model)
  CVAE encoder    : Transformer (n_vae_encoder_layers) → latent z (32-dim)
  Cross-attention : Transformer decoder (4 layers, 8 heads, dim=256)
  Output head     : MLP → chunk of 20 future actions

AIC robot settings
──────────────────
  state dim       : 26   (TCP pose/vel/err + joints)
  images          : 3 × (128×144)  center, left, right cameras
  action dim      : 6   (cartesian twist)
  chunk_size      : 20  (predict 20 future steps)
  n_action_steps  : 10  (execute first 10, re-plan)

Dataset
───────
  local/aic_cable_insertion_large
  80 train + 20 val episodes, 100 steps each → 10,000 frames
  5 trajectory strategies, 3 cameras stored as mp4 (use_videos=True)

Usage
─────
    python train_act.py [--steps 10000] [--batch_size 8] [--lr 1e-5]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

REPO_ID  = "local/aic_cable_insertion_large"
OUT_DIR  = Path(__file__).parent / "outputs_act"
CKPT_DIR = Path(__file__).parent / "checkpoints_act"

CHUNK_SIZE     = 20
N_ACTION_STEPS = 10


# ─── Config ───────────────────────────────────────────────────────────────────

def make_act_config(lr: float = 1e-5, kl_weight: float = 10.0) -> ACTConfig:
    """
    ACTConfig for AIC robot with 3 cameras.

    All three cameras (center, left, right) are fed to separate ResNet-18
    backbones whose features are concatenated before the transformer decoder.
    """
    return ACTConfig(
        # ── I/O shapes ────────────────────────────────────────────────────────
        input_features={
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(26,)
            ),
            "observation.images.center_camera": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 128, 144)
            ),
            "observation.images.left_camera": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 128, 144)
            ),
            "observation.images.right_camera": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 128, 144)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))
        },
        normalization_mapping={
            "STATE":  NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        # ── Action chunking ───────────────────────────────────────────────────
        chunk_size     = CHUNK_SIZE,
        n_action_steps = N_ACTION_STEPS,
        n_obs_steps    = 1,
        # ── Transformer ───────────────────────────────────────────────────────
        dim_model          = 256,
        n_heads            = 8,
        dim_feedforward    = 2048,
        n_encoder_layers   = 4,
        n_decoder_layers   = 4,
        n_vae_encoder_layers = 4,
        # ── Vision backbone ───────────────────────────────────────────────────
        vision_backbone                    = "resnet18",
        pretrained_backbone_weights        = None,
        replace_final_stride_with_dilation = False,
        # ── CVAE ──────────────────────────────────────────────────────────────
        use_vae    = True,
        latent_dim = 32,
        kl_weight  = kl_weight,
        # ── Regularisation ────────────────────────────────────────────────────
        dropout = 0.1,
        pre_norm = False,
    )


# ─── Trainer ──────────────────────────────────────────────────────────────────

class ACTTrainer:
    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device : {self.device}")

        # ── Dataset ──────────────────────────────────────────────────────────
        print(f"Loading dataset: {REPO_ID} …")
        self.full_ds = LeRobotDataset(
            repo_id=REPO_ID,
            delta_timestamps={
                "observation.state":                  [0.0],
                "observation.images.center_camera":   [0.0],
                "observation.images.left_camera":     [0.0],
                "observation.images.right_camera":    [0.0],
                "action": [i / FPS for i in range(CHUNK_SIZE)],
            },
            video_backend="pyav",   # torchcodec needs libavutil.so.56/57; pyav uses system ffmpeg
        )
        n_total = len(self.full_ds)
        n_val   = max(1, int(0.15 * n_total))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            self.full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"],
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"],
            shuffle=False, num_workers=4, pin_memory=True,
        )
        print(f"Train: {n_train:,}  |  Val: {n_val:,}")

        # ── Model ────────────────────────────────────────────────────────────
        act_cfg = make_act_config(lr=cfg["lr"], kl_weight=cfg["kl_weight"])
        self.model = ACTPolicy(act_cfg, dataset_stats=self.full_ds.meta.stats)
        self.model.to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"ACTPolicy parameters: {n_params:,}\n")

        self.preprocessor, _ = make_pre_post_processors(
            act_cfg, dataset_stats=self.full_ds.meta.stats
        )

        # ── Optimiser ────────────────────────────────────────────────────────
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg["steps"], eta_min=cfg["lr"] * 0.01
        )

        # ── Logging ──────────────────────────────────────────────────────────
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []
        self.best_val = float("inf")
        self.global_step = 0

        json.dump({
            "model": {
                "policy":          "ACT (lerobot)",
                "reference":       "https://github.com/huggingface/lerobot",
                "backbone":        "ResNet-18 × 3 cameras",
                "dim_model":       256,
                "n_heads":         8,
                "n_enc_layers":    4,
                "n_dec_layers":    4,
                "chunk_size":      CHUNK_SIZE,
                "n_action_steps":  N_ACTION_STEPS,
                "latent_dim":      32,
                "n_params":        n_params,
            },
            "training": cfg,
            "dataset": {
                "repo_id": REPO_ID,
                "n_train": n_train,
                "n_val":   n_val,
                "cameras": ["center_camera", "left_camera", "right_camera"],
            },
        }, open(OUT_DIR / "run_config.json", "w"), indent=2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _squeeze_obs(self, batch: dict) -> dict:
        """Squeeze n_obs_steps=1 time dim that LeRobotDataset adds."""
        if "observation.state" in batch and batch["observation.state"].ndim == 3:
            batch["observation.state"] = batch["observation.state"][:, 0, :]
        for k in list(batch.keys()):
            if k.startswith("observation.images.") and batch[k].ndim == 5:
                batch[k] = batch[k][:, 0, ...]  # (B, 1, C, H, W) → (B, C, H, W)
        return batch

    # ── Epoch ─────────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(True)
        totals: dict = {}
        n = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = self._to_device(batch)
                batch = self.preprocessor(batch)
                batch = self._squeeze_obs(batch)

                out = self.model(batch)
                if isinstance(out, tuple):
                    total_loss, info = out
                    loss_dict = {"loss": total_loss,
                                 **{k: v for k, v in info.items()
                                    if isinstance(v, torch.Tensor) and v.ndim == 0}}
                elif isinstance(out, dict):
                    loss_dict  = out
                    total_loss = loss_dict.get("loss", next(iter(loss_dict.values())))
                else:
                    total_loss = out
                    loss_dict  = {"loss": total_loss}

                if train:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.global_step += 1

                B = next(v.size(0) for v in batch.values() if isinstance(v, torch.Tensor))
                for k, v in loss_dict.items():
                    totals[k] = totals.get(k, 0.0) + v.item() * B
                n += B

        return {k: v / n for k, v in totals.items()}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self):
        steps_per_epoch = len(self.train_loader)
        n_epochs = max(1, self.cfg["steps"] // steps_per_epoch)
        print(f"Training for {n_epochs} epochs ({self.cfg['steps']} steps, {steps_per_epoch} steps/epoch)")
        print(f"\n{'Epoch':>6}  {'LR':>9}  {'T-loss':>9}  {'V-loss':>9}  {'Best':>5}")
        print("─" * 55)

        for epoch in range(1, n_epochs + 1):
            t0    = time.time()
            t_m   = self._run_epoch(self.train_loader, train=True)
            v_m   = self._run_epoch(self.val_loader,   train=False)
            lr    = self.scheduler.get_last_lr()[0]

            t_loss = t_m.get("loss", t_m.get("total", 0.0))
            v_loss = v_m.get("loss", v_m.get("total", 0.0))

            is_best = v_loss < self.best_val
            if is_best:
                self.best_val = v_loss
                self.model.save_pretrained(str(CKPT_DIR / "best_model"))

            if epoch % self.cfg.get("save_every", 5) == 0:
                self.model.save_pretrained(str(CKPT_DIR / f"checkpoint_epoch_{epoch:03d}"))

            record = {
                "epoch": epoch, "step": self.global_step, "lr": lr,
                **{f"train_{k}": v for k, v in t_m.items()},
                **{f"val_{k}":   v for k, v in v_m.items()},
            }
            self.history.append(record)
            self._save_metrics()

            print(f"{epoch:>6}  {lr:>9.2e}  {t_loss:>9.5f}  {v_loss:>9.5f}  "
                  f"{'✓' if is_best else '':>5}   {time.time()-t0:.1f}s")

        self.model.save_pretrained(str(CKPT_DIR / "final_model"))
        print(f"\nDone. Best val loss: {self.best_val:.5f}")
        print(f"Artifacts → {OUT_DIR}  and  {CKPT_DIR}")

    def _save_metrics(self):
        with open(OUT_DIR / "metrics.json", "w") as f:
            json.dump(self.history, f, indent=2)
        if self.history:
            with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
                w.writeheader()
                w.writerows(self.history)


# ─── Constants referenced in dataset loading ──────────────────────────────────
FPS = 20


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train ACT policy on AIC cable-insertion dataset (lerobot)"
    )
    p.add_argument("--steps",        type=int,   default=5000,
                   help="Total training gradient steps")
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--kl_weight",    type=float, default=10.0,
                   help="KL regularisation weight for CVAE")
    p.add_argument("--save_every",   type=int,   default=5,
                   help="Save checkpoint every N epochs")
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    trainer = ACTTrainer(cfg)
    trainer.train()
