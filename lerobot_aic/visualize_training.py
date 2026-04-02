#!/usr/bin/env python3
"""
visualize_training.py
Interactive visualization of AIC training results.

Generates training curve plots, model comparisons, and dataset overviews.
Works with outputs from train_diffusion.py, train_act.py, and train_lerobot_aic.py.

Usage:
    python lerobot_aic/visualize_training.py                    # all available runs
    python lerobot_aic/visualize_training.py --run diffusion    # specific run
    python lerobot_aic/visualize_training.py --live             # auto-refresh during training
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

BASE     = Path(__file__).parent
VIS_DIR  = BASE / "visualizations"

# ── Color palette ─────────────────────────────────────────────────────────────
PAL = {
    "diffusion":  "#2563EB",  # blue
    "act":        "#DC2626",  # red
    "lerobot":    "#9333EA",  # purple
    "train":      "#2563EB",
    "val":        "#DC2626",
    "lr":         "#D97706",
    "bg":         "#F8FAFC",
    "grid":       "#E2E8F0",
}

# ── Discover runs ─────────────────────────────────────────────────────────────

def discover_runs() -> dict[str, dict]:
    """Find all output directories with metrics.json and return metadata."""
    runs = {}
    for d in sorted(BASE.glob("outputs_*")):
        metrics_path = d / "metrics.json"
        config_path  = d / "run_config.json"
        if not metrics_path.exists():
            continue
        name = d.name.replace("outputs_", "")
        with open(metrics_path) as f:
            metrics = json.load(f)
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        runs[name] = {
            "dir":     d,
            "metrics": metrics,
            "config":  config,
            "name":    name,
        }
    return runs


def _ax_style(ax, title="", xlabel="Epoch", ylabel="Loss"):
    ax.set_facecolor(PAL["bg"])
    ax.grid(color=PAL["grid"], lw=0.8, zorder=0)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)


# ── Plot 1: Loss curves for a single run ─────────────────────────────────────

def plot_single_run(run: dict, out_dir: Path):
    history = run["metrics"]
    name    = run["name"]
    if not history:
        return

    epochs  = [r["epoch"] for r in history]
    t_loss  = [r.get("train_loss", r.get("train_l1_loss", 0)) for r in history]
    v_loss  = [r.get("val_loss",   r.get("val_l1_loss",   0)) for r in history]
    lrs     = [r.get("lr", 0) for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor="white")

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, t_loss, color=PAL["train"], lw=2, label="Train loss")
    ax.plot(epochs, v_loss, color=PAL["val"],   lw=2, ls="--", label="Val loss")
    if v_loss:
        best_idx = int(np.argmin(v_loss))
        best_e   = epochs[best_idx]
        best_v   = v_loss[best_idx]
        ax.axvline(best_e, color=PAL["val"], lw=1, ls=":", alpha=0.7,
                   label=f"Best val @ ep {best_e} ({best_v:.5f})")
        ax.scatter([best_e], [best_v], color=PAL["val"], s=60, zorder=5)
    ax.legend(fontsize=8)
    _ax_style(ax, f"Loss Curves — {name}")

    # Log-scale loss
    ax = axes[1]
    ax.semilogy(epochs, t_loss, color=PAL["train"], lw=2, label="Train")
    ax.semilogy(epochs, v_loss, color=PAL["val"],   lw=2, ls="--", label="Val")
    ax.legend(fontsize=8)
    _ax_style(ax, f"Loss (log scale) — {name}")

    # LR schedule
    ax = axes[2]
    ax.plot(epochs, lrs, color=PAL["lr"], lw=2)
    _ax_style(ax, "Learning Rate Schedule", ylabel="LR")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))

    fig.suptitle(f"Training Run: {name}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = out_dir / f"{name}_training.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ── Plot 2: ACT loss components (L1 + KL) ────────────────────────────────────

def plot_act_components(run: dict, out_dir: Path):
    history = run["metrics"]
    if not history or "train_l1_loss" not in history[0]:
        return  # not an ACT run

    epochs = [r["epoch"] for r in history]
    t_l1   = [r.get("train_l1_loss", 0)  for r in history]
    t_kl   = [r.get("train_kld_loss", 0) for r in history]
    v_l1   = [r.get("val_l1_loss", 0)    for r in history]
    v_kl   = [r.get("val_kld_loss", 0)   for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")

    axes[0].plot(epochs, t_l1, color="#9333EA", lw=2, label="Train")
    axes[0].plot(epochs, v_l1, color="#9333EA", lw=2, ls="--", alpha=0.7, label="Val")
    axes[0].legend(fontsize=9)
    _ax_style(axes[0], "L1 Reconstruction Loss")

    axes[1].plot(epochs, t_kl, color="#16A34A", lw=2, label="Train")
    axes[1].plot(epochs, v_kl, color="#16A34A", lw=2, ls="--", alpha=0.7, label="Val")
    axes[1].legend(fontsize=9)
    _ax_style(axes[1], "KL Divergence")

    fig.suptitle(f"ACT Loss Components — {run['name']}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = out_dir / f"{run['name']}_components.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ── Plot 3: Multi-run comparison ──────────────────────────────────────────────

def plot_comparison(runs: dict[str, dict], out_dir: Path):
    if len(runs) < 2:
        return

    colors = list(plt.cm.tab10.colors)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    # Validation loss comparison
    ax = axes[0]
    for i, (name, run) in enumerate(runs.items()):
        history = run["metrics"]
        epochs  = [r["epoch"] for r in history]
        v_loss  = [r.get("val_loss", r.get("val_l1_loss", 0)) for r in history]
        ax.plot(epochs, v_loss, color=colors[i % len(colors)], lw=2, label=name)
    ax.legend(fontsize=8)
    _ax_style(ax, "Validation Loss Comparison")

    # Best val loss bar chart
    ax = axes[1]
    names, bests = [], []
    for name, run in runs.items():
        v_losses = [r.get("val_loss", r.get("val_l1_loss", float("inf"))) for r in run["metrics"]]
        names.append(name)
        bests.append(min(v_losses) if v_losses else float("inf"))
    bars = ax.barh(names, bests, color=[colors[i % len(colors)] for i in range(len(names))])
    for bar, val in zip(bars, bests):
        ax.text(bar.get_width() + max(bests) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", fontsize=9)
    _ax_style(ax, "Best Validation Loss", xlabel="Loss", ylabel="")

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ── Plot 4: Dataset overview ─────────────────────────────────────────────────

def plot_dataset_overview(out_dir: Path):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("  Skipping dataset overview (lerobot not installed)")
        return

    # Try multiple possible dataset names
    for repo_id in ["local/aic_cable_insertion_large", "local/aic_cable_insertion_sim",
                     "local/aic_cable_insertion"]:
        try:
            ds = LeRobotDataset(repo_id=repo_id, video_backend="pyav")
            break
        except Exception:
            continue
    else:
        print("  No dataset found — run create_larger_dataset.py first")
        return

    print(f"  Dataset: {repo_id}  ({len(ds):,} frames)")

    # Sample one episode
    n_sample = min(100, len(ds))
    samples  = [ds[i] for i in range(n_sample)]
    states   = np.stack([s["observation.state"].numpy() for s in samples])
    actions  = np.stack([s["action"].numpy() for s in samples])

    # Handle action shape: may be (6,) or (chunk, 6)
    if actions.ndim == 3:
        actions = actions[:, 0, :]

    STATE_LABELS = [
        "tcp_x", "tcp_y", "tcp_z", "q_x", "q_y", "q_z", "q_w",
        "vel_lx", "vel_ly", "vel_lz", "vel_ax", "vel_ay", "vel_az",
        "err_x", "err_y", "err_z", "err_rx", "err_ry", "err_rz",
        "j0", "j1", "j2", "j3", "j4", "j5", "gripper",
    ]

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # TCP pose
    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.tab10(np.linspace(0, 0.7, 7))
    for j in range(7):
        ax.plot(states[:, j], color=cmap[j], lw=1.3, label=STATE_LABELS[j])
    ax.legend(fontsize=6, ncol=4)
    _ax_style(ax, "TCP Pose (episode 0)", xlabel="Step", ylabel="Value")

    # Joint positions
    ax = fig.add_subplot(gs[0, 1])
    cmap2 = plt.cm.Set2(np.linspace(0, 1, 7))
    for j in range(min(7, states.shape[1] - 19)):
        ax.plot(states[:, 19 + j], color=cmap2[j], lw=1.3, label=STATE_LABELS[19 + j])
    ax.legend(fontsize=6, ncol=4)
    _ax_style(ax, "Joint Positions (episode 0)", xlabel="Step", ylabel="Value")

    # Action distribution
    ax = fig.add_subplot(gs[1, 0])
    n_act_sample = min(600, len(ds))
    all_acts = np.stack([ds[i]["action"].numpy() for i in range(n_act_sample)])
    if all_acts.ndim == 3:
        all_acts = all_acts[:, 0, :]
    ax.violinplot([all_acts[:, j] for j in range(min(6, all_acts.shape[1]))],
                  showmedians=True)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(["lin.x", "lin.y", "lin.z", "ang.x", "ang.y", "ang.z"], fontsize=8)
    _ax_style(ax, f"Action Distribution ({n_act_sample} samples)", xlabel="Dim", ylabel="Value")

    # Sample camera frames
    ax = fig.add_subplot(gs[1, 1])
    cam_key = None
    for k in ["observation.images.center_camera",
              "observation.images.left_camera",
              "observation.images.right_camera"]:
        if k in samples[0]:
            cam_key = k
            break
    if cam_key:
        indices = [0, n_sample // 4, n_sample // 2, 3 * n_sample // 4]
        imgs = []
        for idx in indices:
            img = samples[idx][cam_key]
            if img.ndim == 4:
                img = img[0]
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img * 0.5 + 0.5, 0, 1) if img.min() < 0 else np.clip(img / 255.0, 0, 1)
            imgs.append(img)
        combined = np.concatenate(imgs, axis=1)
        ax.imshow(combined)
        ax.set_title(f"Frames at t=0, {n_sample//4}, {n_sample//2}, {3*n_sample//4}", fontsize=9)
    ax.axis("off")

    fig.suptitle(f"Dataset Overview — {repo_id}", fontsize=14, fontweight="bold")
    p = out_dir / "dataset_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(runs: dict[str, dict]):
    if not runs:
        print("No training runs found.")
        return

    print("\n" + "=" * 75)
    print(f"  {'Run':<35} {'Epochs':>8} {'Best Val':>12} {'Final Val':>12}")
    print("=" * 75)

    for name, run in runs.items():
        history = run["metrics"]
        if not history:
            continue
        v_losses = [r.get("val_loss", r.get("val_l1_loss", float("inf"))) for r in history]
        best_val = min(v_losses)
        final_val = v_losses[-1]
        n_epochs = history[-1].get("epoch", len(history))
        n_params = run["config"].get("model", {}).get("n_params", "?")
        print(f"  {name:<35} {n_epochs:>8} {best_val:>12.5f} {final_val:>12.5f}")

    print("=" * 75)


# ── Live monitoring ───────────────────────────────────────────────────────────

def live_monitor(interval: int = 30):
    """Re-generate plots every `interval` seconds during training."""
    print(f"Live monitoring (refresh every {interval}s). Press Ctrl+C to stop.\n")
    while True:
        runs = discover_runs()
        if runs:
            VIS_DIR.mkdir(parents=True, exist_ok=True)
            for name, run in runs.items():
                plot_single_run(run, VIS_DIR)
                plot_act_components(run, VIS_DIR)
            plot_comparison(runs, VIS_DIR)
            print_summary(runs)
        else:
            print("  No training outputs yet. Waiting...")
        print(f"\n  Next refresh in {interval}s …")
        time.sleep(interval)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize AIC training results")
    parser.add_argument("--run", type=str, default=None,
                        help="Specific run name to visualize (e.g. 'diffusion', 'act')")
    parser.add_argument("--live", action="store_true",
                        help="Live monitor mode — auto-refresh plots during training")
    parser.add_argument("--interval", type=int, default=30,
                        help="Refresh interval in seconds for --live mode (default: 30)")
    parser.add_argument("--no-dataset", action="store_true",
                        help="Skip dataset overview plot")
    args = parser.parse_args()

    if args.live:
        live_monitor(args.interval)
        return

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()

    if args.run:
        # Filter to specific run
        matches = {k: v for k, v in runs.items() if args.run.lower() in k.lower()}
        if not matches:
            print(f"No run matching '{args.run}'. Available: {list(runs.keys())}")
            return
        runs = matches

    if not runs:
        print("No training outputs found in lerobot_aic/outputs_*/")
        print("Run training first:")
        print("  python lerobot_aic/create_larger_dataset.py")
        print("  python lerobot_aic/train_diffusion.py --steps 5000")
        return

    print(f"Found {len(runs)} training run(s): {list(runs.keys())}\n")
    print("Generating plots …")

    for name, run in runs.items():
        plot_single_run(run, VIS_DIR)
        plot_act_components(run, VIS_DIR)

    if len(runs) >= 2:
        plot_comparison(runs, VIS_DIR)

    if not args.no_dataset:
        print("\nDataset overview …")
        plot_dataset_overview(VIS_DIR)

    print_summary(runs)

    print(f"\nAll visualizations saved to: {VIS_DIR}/")
    print("Open the PNG files to view results.")


if __name__ == "__main__":
    main()
