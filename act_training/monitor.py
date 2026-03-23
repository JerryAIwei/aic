"""
monitor.py
Visualisation tool for ACT training progress.

Can be run *during* training (reads whatever metrics exist so far) or
*after* training for a complete picture.

Generates
─────────
outputs/plots/01_loss_curves.png         – train vs val total loss
outputs/plots/02_loss_components.png     – L1 and KL components side-by-side
outputs/plots/03_lr_curve.png            – learning-rate schedule
outputs/plots/04_dataset_overview.png    – dataset trajectory & action stats
outputs/plots/05_action_predictions.png  – model predictions vs ground truth

Usage
─────
    python monitor.py           # generate all plots
    python monitor.py --watch   # regenerate every 30 s while training
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA_DIR  = BASE / "data"
OUT_DIR   = BASE / "outputs"
PLOT_DIR  = OUT_DIR / "plots"

STYLE = {
    "train_color": "#2563EB",   # blue
    "val_color":   "#DC2626",   # red
    "kl_color":    "#16A34A",   # green
    "l1_color":    "#9333EA",   # purple
    "lr_color":    "#D97706",   # amber
    "bg":          "#F8FAFC",
    "grid":        "#E2E8F0",
}


def load_metrics() -> dict | None:
    path = OUT_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_config() -> dict | None:
    path = OUT_DIR / "run_config.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _style_ax(ax, title="", xlabel="Epoch", ylabel="Loss"):
    ax.set_facecolor(STYLE["bg"])
    ax.grid(color=STYLE["grid"], linewidth=0.8, zorder=0)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 – Total loss curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(m: dict):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    epochs = m["epoch"]

    ax.plot(epochs, m["train_total"], color=STYLE["train_color"],
            lw=2, label="Train total loss")
    ax.plot(epochs, m["val_total"],   color=STYLE["val_color"],
            lw=2, linestyle="--", label="Val total loss")

    best_e = epochs[int(np.argmin(m["val_total"]))]
    best_v = min(m["val_total"])
    ax.axvline(best_e, color=STYLE["val_color"], lw=1, linestyle=":",
               alpha=0.6, label=f"Best val @ epoch {best_e}  ({best_v:.4f})")

    ax.legend(fontsize=9)
    _style_ax(ax, title="Training vs Validation Loss")
    fig.tight_layout()
    path = PLOT_DIR / "01_loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 – Loss components (L1 + KL)
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_components(m: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
    epochs = m["epoch"]

    # L1
    axes[0].plot(epochs, m["train_l1"], color=STYLE["l1_color"], lw=2, label="Train")
    axes[0].plot(epochs, m["val_l1"],   color=STYLE["l1_color"], lw=2,
                 linestyle="--", alpha=0.7, label="Val")
    axes[0].legend(fontsize=9)
    _style_ax(axes[0], title="L1 Reconstruction Loss")

    # KL
    axes[1].plot(epochs, m["train_kl"], color=STYLE["kl_color"], lw=2, label="Train")
    axes[1].plot(epochs, m["val_kl"],   color=STYLE["kl_color"], lw=2,
                 linestyle="--", alpha=0.7, label="Val")
    axes[1].legend(fontsize=9)
    _style_ax(axes[1], title="KL Divergence")

    fig.suptitle("Loss Component Breakdown", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = PLOT_DIR / "02_loss_components.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3 – Learning-rate schedule
# ──────────────────────────────────────────────────────────────────────────────

def plot_lr_curve(m: dict):
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="white")
    ax.plot(m["epoch"], m["lr"], color=STYLE["lr_color"], lw=2)
    _style_ax(ax, title="Learning Rate Schedule", ylabel="Learning Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2e}"))
    fig.tight_layout()
    path = PLOT_DIR / "03_lr_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4 – Dataset overview
# ──────────────────────────────────────────────────────────────────────────────

def plot_dataset_overview():
    ep_path = DATA_DIR / "train" / "episode_000.npz"
    if not ep_path.exists():
        print("  skipping dataset overview (no data found)")
        return

    d    = np.load(ep_path)
    obs  = d["obs"]       # (T, 13)
    acts = d["actions"]   # (T, 7)
    imgs = d["images"]    # (T, H, W, 3)
    T    = len(obs)
    t    = np.arange(T)

    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow",
                   "wrist_1", "wrist_2", "wrist_3", "gripper"]

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Joint trajectories ────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    cmap = plt.cm.tab10(np.linspace(0, 0.7, 7))
    for j in range(7):
        ax0.plot(t, obs[:, j], color=cmap[j], lw=1.3, label=JOINT_NAMES[j])
    ax0.legend(fontsize=6, ncol=4, loc="upper right")
    _style_ax(ax0, title="Joint Angles (episode 0)", ylabel="Radians")

    # ── TCP pose ──────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 2:])
    labels = ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]
    cmap2 = plt.cm.Set2(np.linspace(0, 1, 6))
    for j in range(6):
        ax1.plot(t, obs[:, 7 + j], color=cmap2[j], lw=1.3, label=labels[j])
    ax1.legend(fontsize=6, ncol=3, loc="upper right")
    _style_ax(ax1, title="TCP Pose (episode 0)", ylabel="m / rad")

    # ── Action distribution ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    all_acts = []
    for f in sorted((DATA_DIR / "train").glob("*.npz"))[:30]:
        all_acts.append(np.load(f)["actions"])
    all_acts = np.concatenate(all_acts, axis=0)
    ax2.violinplot([all_acts[:, j] for j in range(7)],
                   positions=range(7), showmedians=True)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(JOINT_NAMES, fontsize=7, rotation=20)
    _style_ax(ax2, title="Action Distribution (30 episodes)",
              xlabel="Joint", ylabel="Target angle (rad)")

    # ── Gripper open/close over episode ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2:])
    ax3.fill_between(t, acts[:, 6], alpha=0.5, color="#F59E0B")
    ax3.plot(t, acts[:, 6], color="#D97706", lw=1.5)
    ax3.axhline(0, color="#94A3B8", lw=0.8, linestyle="--")
    _style_ax(ax3, title="Gripper Command (episode 0)", ylabel="Value")

    # ── Sample frames ─────────────────────────────────────────────────────────
    sample_t = [0, T // 4, T // 2, 3 * T // 4]
    for idx, ti in enumerate(sample_t):
        ax = fig.add_subplot(gs[2, idx])
        ax.imshow(np.clip(imgs[ti], 0, 1))
        ax.set_title(f"t = {ti}", fontsize=8)
        ax.axis("off")

    fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold")
    path = PLOT_DIR / "04_dataset_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5 – Action predictions vs ground truth
# ──────────────────────────────────────────────────────────────────────────────

def plot_action_predictions():
    import torch, sys
    sys.path.insert(0, str(BASE))
    from act_model import ACTPolicy

    ckpt = CKPT_DIR = BASE / "checkpoints" / "best_model.pt"
    cfg_path = OUT_DIR / "run_config.json"
    if not ckpt.exists() or not cfg_path.exists():
        print("  skipping action predictions (checkpoint not found)")
        return

    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)

    mc = cfg["model"]
    model = ACTPolicy(
        obs_dim=mc["obs_dim"], action_dim=mc["action_dim"],
        img_size=mc["img_size"], chunk_size=mc["chunk_size"],
        hidden_dim=mc["hidden_dim"], latent_dim=mc["latent_dim"],
        nhead=mc["nhead"], num_enc_layers=mc["num_enc_layers"],
        num_dec_layers=mc["num_dec_layers"],
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    obs_mean = torch.tensor(meta["obs_mean"])
    obs_std  = torch.tensor(meta["obs_std"])
    act_mean = torch.tensor(meta["action_mean"])
    act_std  = torch.tensor(meta["action_std"])

    # Load one validation episode
    ep = np.load(DATA_DIR / "val" / "episode_000.npz")
    obs_arr  = torch.from_numpy(ep["obs"])
    imgs_arr = torch.from_numpy(ep["images"])
    acts_arr = torch.from_numpy(ep["actions"])
    chunk_size = mc["chunk_size"]

    # Run inference on a few timesteps
    n_show = 4
    step_indices = np.linspace(0, len(obs_arr) - chunk_size - 1, n_show, dtype=int)

    JOINT_NAMES = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3", "gripper"]
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 3 * n_show), facecolor="white")

    for row, t in enumerate(step_indices):
        obs_t = ((obs_arr[t] - obs_mean) / obs_std).unsqueeze(0)
        img_t = imgs_arr[t].permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            pred, _, _ = model(obs_t, img_t)
        pred = (pred.squeeze(0) * act_std + act_mean).numpy()   # (chunk, 7)
        gt   = acts_arr[t : t + chunk_size].numpy()             # (chunk, 7)

        ax = axes[row]
        x  = np.arange(chunk_size)
        width = 0.35
        for j in range(7):
            offset = (j - 3) * width / 7
            ax.bar(x + offset,       gt[:, j],   width / 7, alpha=0.6,
                   color=f"C{j}", label=f"GT {JOINT_NAMES[j]}" if row == 0 else "")
            ax.bar(x + offset + width, pred[:, j], width / 7, alpha=0.9,
                   color=f"C{j}", hatch="//", label=f"Pred {JOINT_NAMES[j]}" if row == 0 else "")

        ax.set_title(f"t = {t}", fontsize=9)
        ax.set_xlabel("Step in chunk", fontsize=8)
        ax.set_ylabel("Joint angle (rad)", fontsize=8)
        ax.set_facecolor(STYLE["bg"])
        ax.grid(color=STYLE["grid"], axis="y", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=6, ncol=7, loc="upper right")
    fig.suptitle("Action Predictions vs Ground Truth (val episode 0)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = PLOT_DIR / "05_action_predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_plots():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    m = load_metrics()
    if m is None:
        print("No metrics found yet. Run train.py first (or wait for it to finish).")
    else:
        print("Generating loss plots …")
        plot_loss_curves(m)
        plot_loss_components(m)
        plot_lr_curve(m)

    print("Generating dataset overview …")
    plot_dataset_overview()

    print("Generating prediction comparison …")
    plot_action_predictions()
    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true",
                   help="Re-generate plots every 30 s (for live monitoring)")
    args = p.parse_args()

    if args.watch:
        print("Watch mode – press Ctrl+C to stop\n")
        while True:
            print(f"[{time.strftime('%H:%M:%S')}] Refreshing plots …")
            generate_all_plots()
            print(f"  sleeping 30 s …\n")
            time.sleep(30)
    else:
        generate_all_plots()
