"""
generate_dataset.py
Generates a synthetic robot-arm dataset for ACT policy training.

Simulates a 6-DOF UR-style arm performing a cable-insertion task in 6 phases:
  home → pre-grasp → grasp → lift → approach → insert

Each episode adds trajectory variation (noise + random offsets) to mimic
real teleoperated demonstrations.

Outputs
-------
data/train/episode_NNN.npz  – 150 training episodes
data/val/episode_NNN.npz    – 50 validation episodes
data/metadata.json          – normalization stats and dataset description
"""

import numpy as np
import json
import os
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
N_TRAIN        = 150
N_VAL          = 50
EPISODE_LEN    = 100          # timesteps per episode
OBS_DIM        = 13           # 7 joint angles + 6 TCP pose
ACTION_DIM     = 7            # 6 joints + gripper
IMG_SIZE       = 32           # synthetic image side (px)
CHUNK_SIZE     = 10           # action chunk size used later in training
NOISE_SCALE    = 0.025
DATA_DIR       = Path(__file__).parent / "data"

# ─── Trajectory waypoints (joint-space, 7-DOF: 6 arm + 1 gripper) ────────────
WAYPOINTS = {
    "home":       np.array([ 0.00, -1.57,  1.57, -1.57, -1.57,  0.00,  0.0]),
    "pre_grasp":  np.array([ 0.30, -1.20,  1.20, -1.50, -1.50,  0.30,  1.0]),
    "grasp":      np.array([ 0.30, -1.10,  1.10, -1.50, -1.50,  0.30,  1.0]),
    "grasped":    np.array([ 0.30, -1.10,  1.10, -1.50, -1.50,  0.30, -0.5]),
    "lift":       np.array([ 0.30, -1.30,  1.30, -1.60, -1.50,  0.30, -0.5]),
    "pre_insert": np.array([ 0.60, -1.40,  1.40, -1.70, -1.50,  0.60, -0.5]),
    "insert":     np.array([ 0.60, -1.28,  1.52, -1.84, -1.50,  0.60, -0.5]),
}
WP_SEQUENCE = list(WAYPOINTS.values())
WP_TIMES    = [0.00, 0.10, 0.25, 0.35, 0.50, 0.70, 1.00]


def smooth_step(alpha: float) -> float:
    """Cubic smooth-step (no velocity discontinuity at endpoints)."""
    return 3 * alpha**2 - 2 * alpha**3


def interpolate_waypoints(t: np.ndarray) -> np.ndarray:
    """Piecewise smooth interpolation through waypoints."""
    traj = np.zeros((len(t), len(WP_SEQUENCE[0])))
    for i, ti in enumerate(t):
        for j in range(len(WP_TIMES) - 1):
            if WP_TIMES[j] <= ti <= WP_TIMES[j + 1]:
                alpha = (ti - WP_TIMES[j]) / (WP_TIMES[j + 1] - WP_TIMES[j])
                alpha = smooth_step(alpha)
                traj[i] = (1 - alpha) * WP_SEQUENCE[j] + alpha * WP_SEQUENCE[j + 1]
                break
    return traj


def simple_fk(joints: np.ndarray) -> np.ndarray:
    """
    Simplified planar forward kinematics → TCP pose [x, y, z, rx, ry, rz].
    Not physically accurate but produces smooth, correlated signals.
    """
    L1, L2 = 0.425, 0.39          # upper / lower arm link lengths
    q0, q1, q2, q3, q4, q5 = joints[:6]
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    z = 0.7 + 0.1 * np.sin(q0)
    return np.array([x * np.cos(q0), y * np.sin(q0), z, q3, q4, q5])


def generate_episode(seed: int) -> dict:
    rng = np.random.RandomState(seed)

    t = np.linspace(0, 1, EPISODE_LEN)
    joint_traj = interpolate_waypoints(t)                          # (T, 7)

    # Per-episode offset + timestep noise
    offset = rng.randn(7) * 0.04
    noise  = rng.randn(EPISODE_LEN, 7) * NOISE_SCALE
    joint_traj = joint_traj + offset[None, :] + noise
    joint_traj[:, 6] = np.clip(joint_traj[:, 6], -1.0, 1.0)       # gripper ∈ [-1, 1]

    # TCP pose via simplified FK
    tcp = np.stack([simple_fk(q) for q in joint_traj])             # (T, 6)

    # Observations = [joint_angles | tcp_pose]
    obs = np.concatenate([joint_traj, tcp], axis=1).astype(np.float32)  # (T, 13)

    # Actions = next joint position targets (1-step shift)
    actions = np.concatenate([joint_traj[1:], joint_traj[-1:]], axis=0).astype(np.float32)  # (T, 7)

    # Synthetic images: background + blob whose position encodes TCP x,y
    images = _render_images(tcp, rng)                              # (T, H, W, 3)

    return {"obs": obs, "actions": actions, "images": images}


def _render_images(tcp: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Render simple top-down view: white background, red TCP dot."""
    T   = len(tcp)
    H = W = IMG_SIZE
    imgs = np.ones((T, H, W, 3), dtype=np.float32) * 0.85

    # Normalise TCP x,y to pixel coordinates
    x_norm = (tcp[:, 0] + 0.6) / 1.2   # rough range → [0, 1]
    y_norm = (tcp[:, 1] + 0.6) / 1.2

    y_grid, x_grid = np.ogrid[:H, :W]
    for i in range(T):
        cx = int(np.clip(x_norm[i] * W, 1, W - 2))
        cy = int(np.clip(y_norm[i] * H, 1, H - 2))
        mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= 4
        imgs[i, mask, 0] = 0.9
        imgs[i, mask, 1] = 0.15
        imgs[i, mask, 2] = 0.15

    imgs += rng.randn(T, H, W, 3).astype(np.float32) * 0.03
    return np.clip(imgs, 0.0, 1.0)


def compute_normalization_stats(episodes: list[dict]) -> dict:
    all_obs     = np.concatenate([e["obs"]     for e in episodes], axis=0)
    all_actions = np.concatenate([e["actions"] for e in episodes], axis=0)
    return {
        "obs_mean":     all_obs.mean(0).tolist(),
        "obs_std":      (all_obs.std(0) + 1e-6).tolist(),
        "action_mean":  all_actions.mean(0).tolist(),
        "action_std":   (all_actions.std(0) + 1e-6).tolist(),
        "obs_dim":      OBS_DIM,
        "action_dim":   ACTION_DIM,
        "img_size":     IMG_SIZE,
        "chunk_size":   CHUNK_SIZE,
        "episode_len":  EPISODE_LEN,
        "n_train":      N_TRAIN,
        "n_val":        N_VAL,
    }


def main():
    print("Generating ACT training dataset …")
    print(f"  episodes : {N_TRAIN} train  /  {N_VAL} val")
    print(f"  steps    : {EPISODE_LEN}  |  obs dim: {OBS_DIM}  |  action dim: {ACTION_DIM}")
    print(f"  img size : {IMG_SIZE}×{IMG_SIZE}  |  chunk size: {CHUNK_SIZE}\n")

    train_eps, val_eps = [], []

    # Training episodes
    for i in range(N_TRAIN):
        ep = generate_episode(seed=i)
        path = DATA_DIR / "train" / f"episode_{i:03d}.npz"
        np.savez_compressed(path, **ep)
        train_eps.append(ep)
        if (i + 1) % 30 == 0:
            print(f"  [train] {i + 1}/{N_TRAIN}")

    # Validation episodes
    for i in range(N_VAL):
        ep = generate_episode(seed=N_TRAIN + i)
        path = DATA_DIR / "val" / f"episode_{i:03d}.npz"
        np.savez_compressed(path, **ep)
        val_eps.append(ep)
        if (i + 1) % 10 == 0:
            print(f"  [val]   {i + 1}/{N_VAL}")

    # Normalization stats + metadata
    stats = compute_normalization_stats(train_eps)
    stats["description"] = (
        "Synthetic cable-insertion dataset. "
        "Each episode simulates a 6-DOF UR arm moving through 6 phases: "
        "home → pre-grasp → grasp → lift → approach → insert."
    )
    stats["phases"] = list(WAYPOINTS.keys())
    stats["noise_scale"] = NOISE_SCALE

    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset saved to {DATA_DIR}")
    print(f"Normalization stats saved to {DATA_DIR / 'metadata.json'}")
    print("\nSample episode stats:")
    ep0 = train_eps[0]
    print(f"  obs shape    : {ep0['obs'].shape}")
    print(f"  action shape : {ep0['actions'].shape}")
    print(f"  image shape  : {ep0['images'].shape}")
    print(f"  action range : [{ep0['actions'].min():.3f}, {ep0['actions'].max():.3f}]")


if __name__ == "__main__":
    main()
