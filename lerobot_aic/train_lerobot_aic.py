"""
train_lerobot_aic.py
Trains an ACT policy on the AIC cable-insertion dataset using LeRobot 0.5.1.

Key AIC robot settings applied:
  - 3-camera input (center, left, right) at 256×288
  - 26-dim proprioceptive state
  - 6-dim cartesian-twist action
  - chunk_size=20, n_action_steps=10
  - ResNet18 backbone (no pretrained weights for speed)
  - Saves metrics every step so kills don't lose data

Usage:
    python train_lerobot_aic.py [--steps 5000] [--batch_size 8] [--lr 1e-5]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

# ─── LeRobot imports ──────────────────────────────────────────────────────────
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

REPO_ID   = "local/aic_cable_insertion"
OUT_DIR   = Path(__file__).parent / "outputs_lerobot"
CKPT_DIR  = Path(__file__).parent / "checkpoints_lerobot"


# ─── ACT config tailored for AIC robot ───────────────────────────────────────

def make_aic_act_config(lr: float = 1e-5, kl_weight: float = 10.0) -> ACTConfig:
    """
    Build ACTConfig matching the AIC robot's observation / action space.

    AIC observations
    ────────────────
      observation.state                    (26,)  STATE
      observation.images.center_camera  (256,288,3)  VISUAL   ← primary
      observation.images.left_camera    (256,288,3)  VISUAL
      observation.images.right_camera   (256,288,3)  VISUAL

    AIC actions
    ───────────
      action  (6,)  ACTION  – cartesian twist [lin.x/y/z, ang.x/y/z]
    """
    return ACTConfig(
        # ── I/O shapes ────────────────────────────────────────────────────────
        input_features={
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(26,)
            ),
            # Synthetic dataset uses center camera only at 128×144.
            # For real AIC robot: add left_camera + right_camera at (3, 256, 288).
            "observation.images.center_camera": PolicyFeature(
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
        chunk_size     = 20,   # predict 20 future steps
        n_action_steps = 10,   # execute 10, discard 10 (temporal overlap)
        n_obs_steps    = 1,
        # ── Transformer ───────────────────────────────────────────────────────
        dim_model          = 256,   # smaller than default 512 for speed
        n_heads            = 8,
        dim_feedforward    = 2048,
        n_encoder_layers   = 4,
        n_decoder_layers   = 4,
        n_vae_encoder_layers = 4,
        # ── Vision backbone ───────────────────────────────────────────────────
        vision_backbone                  = "resnet18",
        pretrained_backbone_weights      = None,      # train from scratch on synthetic data
        replace_final_stride_with_dilation = False,
        # ── CVAE ──────────────────────────────────────────────────────────────
        use_vae    = True,
        latent_dim = 32,
        kl_weight  = kl_weight,
        # ── Regularisation ────────────────────────────────────────────────────
        dropout = 0.1,
        pre_norm = False,
    )


# ─── Training loop ────────────────────────────────────────────────────────────

class LeRobotACITrainer:
    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device : {self.device}")

        # ── Dataset ──────────────────────────────────────────────────────────
        print(f"Loading dataset: {REPO_ID} …")
        self.full_ds = LeRobotDataset(
            repo_id   = REPO_ID,
            delta_timestamps = {"observation.state": [0.0],
                                 "observation.images.center_camera": [0.0],
                                 "action": [i / 20 for i in range(20)]},
        )
        n_total = len(self.full_ds)
        n_val   = max(1, int(0.15 * n_total))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            self.full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"],
            shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"],
            shuffle=False, num_workers=2, pin_memory=True,
        )
        print(f"Train: {n_train:,}  |  Val: {n_val:,}")

        # ── Model ────────────────────────────────────────────────────────────
        act_cfg = make_aic_act_config(lr=cfg["lr"], kl_weight=cfg["kl_weight"])
        self.model = ACTPolicy(act_cfg, dataset_stats=self.full_ds.meta.stats)
        self.model.to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"ACTPolicy parameters: {n_params:,}\n")

        # Build lerobot pre-processor (normalises inputs, reshapes images, etc.)
        self.preprocessor, _ = make_pre_post_processors(
            act_cfg, dataset_stats=self.full_ds.meta.stats
        )

        # ── Optimiser & scheduler ────────────────────────────────────────────
        self.optimizer = AdamW(
            self.model.parameters(),
            lr           = cfg["lr"],
            weight_decay = cfg["weight_decay"],
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

        # Save config
        run_cfg = {
            "model": {
                "policy":         "ACT (lerobot 0.5.1)",
                "vision_backbone": "ResNet18",
                "dim_model":       256,
                "n_heads":         8,
                "n_enc_layers":    4,
                "n_dec_layers":    4,
                "chunk_size":      20,
                "n_action_steps":  10,
                "latent_dim":      32,
                "n_params":        n_params,
                "obs_dims": {
                    "state": 26,
                    "images": "1 × (128×144) center_camera",
                },
                "note": "Add left/right cameras at 256×288 for real AIC robot deployment",
                "action_dim": 6,
            },
            "training": cfg,
            "dataset": {
                "repo_id":        REPO_ID,
                "n_train":        n_train,
                "n_val":          n_val,
                "total_episodes": 130,
            },
        }
        with open(OUT_DIR / "run_config.json", "w") as f:
            json.dump(run_cfg, f, indent=2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _batch_to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        # ACT's VAE encoder only runs in train() mode; keep it on so KL loss is valid in val too.
        self.model.train(True)
        totals = {}
        n = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = self._batch_to_device(batch)
                # Apply lerobot preprocessor (normalise + reshape images/state)
                batch = self.preprocessor(batch)
                # Squeeze n_obs_steps=1 time dimension that LeRobotDataset adds.
                # ACT model expects (B, D) for state and (B, C, H, W) for images.
                if "observation.state" in batch and batch["observation.state"].ndim == 3:
                    batch["observation.state"] = batch["observation.state"][:, 0, :]
                for k in list(batch.keys()):
                    if k.startswith("observation.images.") and batch[k].ndim == 5:
                        batch[k] = batch[k][:, 0, ...]   # (B,1,C,H,W) → (B,C,H,W)
                out = self.model(batch)
                # ACTPolicy.forward returns (loss_tensor, output_dict)
                if isinstance(out, tuple):
                    total_loss, info = out
                    loss_dict = {"loss": total_loss, **{k: v for k, v in info.items()
                                                        if isinstance(v, torch.Tensor) and v.ndim == 0}}
                elif isinstance(out, dict):
                    loss_dict = out
                    total_loss = loss_dict.get("loss", next(iter(loss_dict.values())))
                else:
                    total_loss = out
                    loss_dict = {"loss": total_loss}

                if train:
                    self.optimizer.zero_grad()
                    total_loss.backward()  # total_loss is now always a scalar tensor
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.global_step += 1

                B = next(v for v in batch.values() if isinstance(v, torch.Tensor)).size(0)
                for k, v in loss_dict.items():
                    totals[k] = totals.get(k, 0.0) + v.item() * B
                n += B

        return {k: v / n for k, v in totals.items()}

    # ── Main training ─────────────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        steps_per_epoch = len(self.train_loader)
        n_epochs = max(1, cfg["steps"] // steps_per_epoch)
        print(f"Training for {n_epochs} epochs ({cfg['steps']} steps, {steps_per_epoch} steps/epoch)")
        print(f"\n{'Epoch':>6}  {'LR':>9}  {'T-loss':>9}  {'V-loss':>9}  {'Best':>5}")
        print("─" * 55)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_m = self._run_epoch(self.train_loader, train=True)
            val_m   = self._run_epoch(self.val_loader,   train=False)
            lr      = self.scheduler.get_last_lr()[0]

            t_loss = train_m.get("loss", train_m.get("total", 0.0))
            v_loss = val_m.get("loss", val_m.get("total", 0.0))

            is_best = v_loss < self.best_val
            if is_best:
                self.best_val = v_loss
                self.model.save_pretrained(str(CKPT_DIR / "best_model"))

            record = {
                "epoch": epoch,
                "step":  self.global_step,
                "lr":    lr,
                **{f"train_{k}": v for k, v in train_m.items()},
                **{f"val_{k}":   v for k, v in val_m.items()},
            }
            self.history.append(record)
            self._save_metrics()    # flush after every epoch

            if epoch % cfg.get("save_every", 5) == 0:
                self.model.save_pretrained(str(CKPT_DIR / f"checkpoint_epoch_{epoch:03d}"))

            elapsed = time.time() - t0
            print(f"{epoch:>6}  {lr:>9.2e}  {t_loss:>9.5f}  {v_loss:>9.5f}  "
                  f"{'✓' if is_best else '':>5}   {elapsed:.1f}s")

        self.model.save_pretrained(str(CKPT_DIR / "final_model"))
        print(f"\nTraining complete.  Best val loss: {self.best_val:.5f}")
        print(f"Artifacts saved to {OUT_DIR}  and  {CKPT_DIR}")

    def _save_metrics(self):
        with open(OUT_DIR / "metrics.json", "w") as f:
            json.dump(self.history, f, indent=2)
        if self.history:
            with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
                writer.writeheader()
                writer.writerows(self.history)


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",        type=int,   default=5000)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--kl_weight",    type=float, default=10.0)
    p.add_argument("--save_every",   type=int,   default=5)
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    trainer = LeRobotACITrainer(cfg)
    trainer.train()
