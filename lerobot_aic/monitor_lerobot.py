"""
monitor_lerobot.py
Visualises the LeRobot ACT training run.

Generates
─────────
outputs_lerobot/plots/01_loss_curves.png
outputs_lerobot/plots/02_loss_components.png
outputs_lerobot/plots/03_lr_curve.png
outputs_lerobot/plots/04_dataset_overview.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

BASE     = Path(__file__).parent
OUT_DIR  = BASE / "outputs_lerobot"
PLOT_DIR = OUT_DIR / "plots"

COL = {"train": "#2563EB", "val": "#DC2626", "lr": "#D97706",
       "l1": "#9333EA", "kl": "#16A34A", "bg": "#F8FAFC", "grid": "#E2E8F0"}


def load_history():
    p = OUT_DIR / "metrics.json"
    if not p.exists():
        return None
    return json.load(open(p))


def _ax_style(ax, title="", xlabel="Epoch", ylabel="Loss"):
    ax.set_facecolor(COL["bg"])
    ax.grid(color=COL["grid"], lw=0.8, zorder=0)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)


def plot_loss_curves(history):
    epochs  = [r["epoch"] for r in history]
    t_loss  = [r.get("train_loss", r.get("train_l1_loss", 0)) for r in history]
    v_loss  = [r.get("val_loss",   r.get("val_l1_loss",   0)) for r in history]

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    ax.plot(epochs, t_loss, color=COL["train"], lw=2, label="Train total loss")
    ax.plot(epochs, v_loss, color=COL["val"],   lw=2, ls="--", label="Val total loss")
    best_e  = epochs[int(np.argmin(v_loss))]
    best_v  = min(v_loss)
    ax.axvline(best_e, color=COL["val"], lw=1, ls=":", alpha=0.7,
               label=f"Best val @ epoch {best_e}  ({best_v:.4f})")
    ax.legend(fontsize=9)
    _ax_style(ax, "Training vs Validation Loss (LeRobot ACT)")
    fig.tight_layout()
    p = PLOT_DIR / "01_loss_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p.name}")


def plot_loss_components(history):
    epochs = [r["epoch"] for r in history]
    t_l1   = [r.get("train_l1_loss",  r.get("train_loss", 0)) for r in history]
    t_kl   = [r.get("train_kld_loss", 0) for r in history]
    v_l1   = [r.get("val_l1_loss",    r.get("val_loss",   0)) for r in history]
    v_kl   = [r.get("val_kld_loss",   0) for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
    axes[0].plot(epochs, t_l1, color=COL["l1"], lw=2, label="Train")
    axes[0].plot(epochs, v_l1, color=COL["l1"], lw=2, ls="--", alpha=0.7, label="Val")
    axes[0].legend(fontsize=9)
    _ax_style(axes[0], "L1 Reconstruction Loss")

    axes[1].plot(epochs, t_kl, color=COL["kl"], lw=2, label="Train")
    axes[1].plot(epochs, v_kl, color=COL["kl"], lw=2, ls="--", alpha=0.7, label="Val")
    axes[1].legend(fontsize=9)
    _ax_style(axes[1], "KL Divergence")

    fig.suptitle("Loss Component Breakdown — LeRobot ACT (AIC robot)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = PLOT_DIR / "02_loss_components.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p.name}")


def plot_lr_curve(history):
    epochs = [r["epoch"] for r in history]
    lrs    = [r["lr"] for r in history]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="white")
    ax.plot(epochs, lrs, color=COL["lr"], lw=2)
    _ax_style(ax, "Learning Rate Schedule (Cosine Annealing)", ylabel="LR")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2e}"))
    fig.tight_layout()
    p = PLOT_DIR / "03_lr_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p.name}")


def plot_dataset_overview():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    try:
        ds = LeRobotDataset(repo_id="local/aic_cable_insertion")
    except Exception as e:
        print(f"  skipping dataset overview ({e})")
        return

    # Sample one episode
    ep0 = [ds[i] for i in range(60)]
    states  = np.stack([e["observation.state"].numpy() for e in ep0])
    actions = np.stack([e["action"].numpy()          for e in ep0])
    # images: may be (3,H,W) or (1,3,H,W) depending on dataset version
    raw_imgs = []
    for e in ep0:
        img = e["observation.images.center_camera"]
        if img.ndim == 4: img = img[0]      # (1,3,H,W) → (3,H,W)
        raw_imgs.append(img.permute(1,2,0).numpy())
    imgs = np.clip(np.stack(raw_imgs) * 0.5 + 0.5, 0, 1)
    T = len(states)
    t = np.arange(T)

    STATE_LABELS = ["tcp_x","tcp_y","tcp_z","q_x","q_y","q_z","q_w",
                    "vel_lx","vel_ly","vel_lz","vel_ax","vel_ay","vel_az",
                    "err_x","err_y","err_z","err_rx","err_ry","err_rz",
                    "j0","j1","j2","j3","j4","j5","gripper"]

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # State (first 7 dims: TCP pose)
    ax0 = fig.add_subplot(gs[0, :2])
    cmap = plt.cm.tab10(np.linspace(0, 0.7, 7))
    for j in range(7):
        ax0.plot(t, states[:, j], color=cmap[j], lw=1.3, label=STATE_LABELS[j])
    ax0.legend(fontsize=6, ncol=4)
    _ax_style(ax0, "TCP Pose (episode 0, norm.)", ylabel="Normalised")

    # Joint positions (dims 19-25)
    ax1 = fig.add_subplot(gs[0, 2:])
    cmap2 = plt.cm.Set2(np.linspace(0, 1, 7))
    for j in range(7):
        ax1.plot(t, states[:, 19+j], color=cmap2[j], lw=1.3, label=STATE_LABELS[19+j])
    ax1.legend(fontsize=6, ncol=4)
    _ax_style(ax1, "Joint Positions (episode 0, norm.)", ylabel="Normalised")

    # Action distribution
    ax2 = fig.add_subplot(gs[1, :2])
    all_acts = np.stack([ds[i]["action"].numpy() for i in range(min(600, len(ds)))])
    # action may be (6,) or (chunk,6) — take first step
    if all_acts.ndim == 3: all_acts = all_acts[:, 0, :]
    ax2.violinplot([all_acts[:, j] for j in range(6)],
                   positions=range(6), showmedians=True)
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(["lin.x","lin.y","lin.z","ang.x","ang.y","ang.z"], fontsize=8)
    _ax_style(ax2, "Cartesian Action Distribution (600 samples)", xlabel="Dim", ylabel="Normalised")

    # Gripper
    ax3 = fig.add_subplot(gs[1, 2:])
    ax3.fill_between(t, states[:, 25], alpha=0.4, color="#F59E0B")
    ax3.plot(t, states[:, 25], color="#D97706", lw=1.5)
    ax3.axhline(0, color="#94A3B8", lw=0.8, ls="--")
    _ax_style(ax3, "Gripper State (episode 0, norm.)", ylabel="Normalised")

    # Sample frames
    for idx, ti in enumerate([0, T//4, T//2, 3*T//4]):
        ax = fig.add_subplot(gs[2, idx])
        ax.imshow(np.clip(imgs[ti], 0, 1))
        ax.set_title(f"t={ti}", fontsize=8)
        ax.axis("off")

    fig.suptitle("AIC Dataset Overview (LeRobot format)", fontsize=14, fontweight="bold")
    p = PLOT_DIR / "04_dataset_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p.name}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    if history:
        print("Generating loss plots …")
        plot_loss_curves(history)
        plot_loss_components(history)
        plot_lr_curve(history)
    else:
        print("No metrics.json yet.")
    print("Generating dataset overview …")
    plot_dataset_overview()
    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
