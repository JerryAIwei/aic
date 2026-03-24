"""
create_larger_dataset.py
Creates a large LeRobot-format dataset for AIC cable-insertion training.
Compatible with https://github.com/huggingface/lerobot training pipelines.

Dataset: local/aic_cable_insertion_large
  - 500 train + 100 val episodes
  - 100 steps per episode  →  60,000 total frames
  - 3 cameras: center, left, right  (matching real AIC robot spec)
  - 5 trajectory strategies for diversity
  - 26-dim state  |  6-dim cartesian-twist action

Usage:
    python create_larger_dataset.py
    python create_larger_dataset.py --train 500 --val 100 --steps 100
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

# ─── Dataset parameters ───────────────────────────────────────────────────────
REPO_ID     = "local/aic_cable_insertion_large"
N_TRAIN     = 500
N_VAL       = 100
EPISODE_LEN = 100
FPS         = 20
IMG_H       = 128
IMG_W       = 144
STATE_DIM   = 26
ACTION_DIM  = 6

# ─── Trajectory waypoints in joint space ──────────────────────────────────────
# Home → pre-grasp → grasp → lift → approach → insert
_WAYPOINTS_BASE = [
    np.array([ 0.00, -1.57,  1.57, -1.57, -1.57,  0.00]),
    np.array([ 0.30, -1.20,  1.20, -1.50, -1.50,  0.30]),
    np.array([ 0.30, -1.10,  1.10, -1.50, -1.50,  0.30]),
    np.array([ 0.30, -1.30,  1.30, -1.60, -1.50,  0.30]),
    np.array([ 0.60, -1.40,  1.40, -1.70, -1.50,  0.60]),
    np.array([ 0.60, -1.28,  1.52, -1.84, -1.50,  0.60]),
]
_WP_T_BASE = [0.0, 0.12, 0.28, 0.42, 0.65, 1.0]

# ─── 5 trajectory strategies ──────────────────────────────────────────────────
STRATEGIES = [
    "straight_approach",   # direct approach, nominal speed
    "curved_approach",     # slight arc to avoid collision
    "slow_insertion",      # slower, more precise
    "angled_approach",     # approach from different angle
    "recovery",            # small error then correction
]


def _smooth(a: float) -> float:
    """Smoothstep easing."""
    return 3 * a**2 - 2 * a**3


def _interp(t_arr: np.ndarray, waypoints, wp_t) -> np.ndarray:
    """Piecewise smooth interpolation through waypoints."""
    out = np.zeros((len(t_arr), 6))
    for i, t in enumerate(t_arr):
        t = float(np.clip(t, wp_t[0], wp_t[-1]))
        for j in range(len(wp_t) - 1):
            if wp_t[j] <= t <= wp_t[j + 1]:
                a = _smooth((t - wp_t[j]) / (wp_t[j + 1] - wp_t[j]))
                out[i] = (1 - a) * waypoints[j] + a * waypoints[j + 1]
                break
    return out


def _build_trajectory(strategy: str, rng: np.random.RandomState) -> tuple:
    """Return (waypoints, wp_t) modified for the given strategy."""
    wps = [w.copy() for w in _WAYPOINTS_BASE]
    wpt = list(_WP_T_BASE)

    if strategy == "straight_approach":
        pass  # nominal

    elif strategy == "curved_approach":
        # Add an extra intermediate waypoint with lateral offset
        extra = wps[3].copy()
        extra[0] += rng.uniform(0.05, 0.12)   # base rotation offset
        extra[1] += rng.uniform(-0.08, 0.08)
        wps.insert(4, extra)
        wpt = [0.0, 0.12, 0.28, 0.42, 0.54, 0.65, 1.0]

    elif strategy == "slow_insertion":
        # Compress time near insertion (wp 4→5 takes longer)
        wpt = [0.0, 0.10, 0.25, 0.40, 0.60, 1.0]

    elif strategy == "angled_approach":
        # Different base rotation at home and approach
        wps[0][0] += rng.uniform(-0.15, 0.15)
        wps[4][0] += rng.uniform(-0.10, 0.10)
        wps[5][0] += rng.uniform(-0.05, 0.05)

    elif strategy == "recovery":
        # Intentional small error at waypoint 4, then correction
        error = rng.randn(6) * 0.08
        wps[4] = wps[4] + error
        # Correction: extra waypoint between 4 and 5 moving back toward nominal
        correction = wps[4] * 0.3 + wps[5] * 0.7
        wps.insert(5, correction)
        wpt = [0.0, 0.12, 0.28, 0.42, 0.65, 0.80, 1.0]

    return wps, wpt


def _fk(q6: np.ndarray) -> np.ndarray:
    """Simplified forward kinematics → [x, y, z, qx, qy, qz, qw]."""
    L1, L2 = 0.425, 0.39
    x = (L1 * np.cos(q6[1]) + L2 * np.cos(q6[1] + q6[2])) * np.cos(q6[0])
    y = (L1 * np.cos(q6[1]) + L2 * np.cos(q6[1] + q6[2])) * np.sin(q6[0])
    z = 0.7 + 0.1 * np.sin(q6[0])
    rx, ry, rz = q6[3] * 0.5, q6[4] * 0.5, q6[5] * 0.5
    w = max(0.0, 1.0 - rx**2 - ry**2 - rz**2) ** 0.5
    return np.array([x, y, z, rx, ry, rz, w], dtype=np.float32)


def _twist(q_curr: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    """Approximate Cartesian twist from joint delta."""
    return ((q_next - q_curr) * 0.3).astype(np.float32)


# ─── Rendering ────────────────────────────────────────────────────────────────

# Socket target position (normalized)
_SOCKET_X, _SOCKET_Y = 0.45, 0.10


def _render_center(tcp: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Top-down view: grey background, red TCP dot, blue socket target."""
    img = np.full((IMG_H, IMG_W, 3), 195, dtype=np.uint8)
    yg, xg = np.ogrid[:IMG_H, :IMG_W]

    # Socket target (blue circle)
    sx = int(np.clip((_SOCKET_X + 0.6) / 1.2 * IMG_W, 2, IMG_W - 3))
    sy = int(np.clip((_SOCKET_Y + 0.6) / 1.2 * IMG_H, 2, IMG_H - 3))
    socket_mask = (xg - sx) ** 2 + (yg - sy) ** 2 <= 25
    img[socket_mask, 0] = 50
    img[socket_mask, 1] = 80
    img[socket_mask, 2] = 220

    # TCP (red dot)
    cx = int(np.clip((tcp[0] + 0.6) / 1.2 * IMG_W, 2, IMG_W - 3))
    cy = int(np.clip((tcp[1] + 0.6) / 1.2 * IMG_H, 2, IMG_H - 3))
    tcp_mask = (xg - cx) ** 2 + (yg - cy) ** 2 <= 64
    img[tcp_mask, 0] = 220
    img[tcp_mask, 1] = 50
    img[tcp_mask, 2] = 50

    noise = rng.randint(-10, 10, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _render_left(tcp: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Left side view: height (z) vs x position."""
    img = np.full((IMG_H, IMG_W, 3), 210, dtype=np.uint8)
    yg, xg = np.ogrid[:IMG_H, :IMG_W]

    # Socket
    sx = int(np.clip((_SOCKET_X + 0.6) / 1.2 * IMG_W, 2, IMG_W - 3))
    sy = int(np.clip((0.7 + 0.6) / 1.4 * IMG_H, 2, IMG_H - 3))
    img[(xg - sx) ** 2 + (yg - sy) ** 2 <= 25, 2] = 200

    # TCP
    cx = int(np.clip((tcp[0] + 0.6) / 1.2 * IMG_W, 2, IMG_W - 3))
    cy = int(np.clip((tcp[2] + 0.1) / 1.4 * IMG_H, 2, IMG_H - 3))
    img[(xg - cx) ** 2 + (yg - cy) ** 2 <= 64, 0] = 200
    img[(xg - cx) ** 2 + (yg - cy) ** 2 <= 64, 1] = 60

    noise = rng.randint(-8, 8, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _render_right(tcp: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Right side view: height (z) vs y position."""
    img = np.full((IMG_H, IMG_W, 3), 205, dtype=np.uint8)
    yg, xg = np.ogrid[:IMG_H, :IMG_W]

    # Socket
    sy_val = int(np.clip((_SOCKET_Y + 0.6) / 1.2 * IMG_W, 2, IMG_W - 3))
    sz = int(np.clip((0.7 + 0.1) / 1.4 * IMG_H, 2, IMG_H - 3))
    img[(xg - sy_val) ** 2 + (yg - sz) ** 2 <= 25, 2] = 200

    # TCP
    cy_pix = int(np.clip((tcp[1] + 0.6) / 1.2 * IMG_W, 2, IMG_W - 3))
    cz_pix = int(np.clip((tcp[2] + 0.1) / 1.4 * IMG_H, 2, IMG_H - 3))
    img[(xg - cy_pix) ** 2 + (yg - cz_pix) ** 2 <= 64, 0] = 200
    img[(xg - cy_pix) ** 2 + (yg - cz_pix) ** 2 <= 64, 1] = 60

    noise = rng.randint(-8, 8, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ─── Episode generation ───────────────────────────────────────────────────────

def generate_episode(seed: int, strategy: str) -> tuple:
    """
    Returns (states, actions, images_center, images_left, images_right, task_str).
    states  : (T, 26) float32
    actions : (T, 6)  float32
    images_* : list[T] of (IMG_H, IMG_W, 3) uint8
    """
    rng = np.random.RandomState(seed)
    wps, wpt = _build_trajectory(strategy, rng)

    t = np.linspace(0, 1, EPISODE_LEN)

    # Speed variation per episode
    speed = rng.uniform(0.80, 1.20)
    t_scaled = np.clip(t * speed, 0.0, 1.0)

    q = _interp(t_scaled, wps, wpt)
    q += rng.randn(*q.shape) * 0.020          # step noise
    q += (rng.randn(6) * 0.05)[None, :]       # per-episode joint offset

    gripper = np.where(t < 0.28, 1.0, -0.5).astype(np.float32)

    states, actions = [], []
    imgs_c, imgs_l, imgs_r = [], [], []

    for i in range(EPISODE_LEN):
        tcp = _fk(q[i])

        dq = (q[i + 1] - q[i]) * FPS if i < EPISODE_LEN - 1 else np.zeros(6)
        tcp_vel = np.concatenate([dq[:3] * 0.3, dq[3:] * 0.2])
        tcp_err = rng.randn(6).astype(np.float32) * 0.002
        jpos    = np.append(q[i], gripper[i]).astype(np.float32)

        state = np.concatenate([tcp, tcp_vel, tcp_err, jpos]).astype(np.float32)
        states.append(state)

        qn = q[i + 1] if i < EPISODE_LEN - 1 else q[i]
        actions.append(_twist(q[i], qn))

        imgs_c.append(_render_center(tcp, rng))
        imgs_l.append(_render_left(tcp, rng))
        imgs_r.append(_render_right(tcp, rng))

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        imgs_c, imgs_l, imgs_r,
        f"cable_insertion_{strategy}",
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(n_train: int = N_TRAIN, n_val: int = N_VAL, episode_len: int = EPISODE_LEN):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [
                "tcp_pose.position.x", "tcp_pose.position.y", "tcp_pose.position.z",
                "tcp_pose.orientation.x", "tcp_pose.orientation.y",
                "tcp_pose.orientation.z", "tcp_pose.orientation.w",
                "tcp_velocity.linear.x", "tcp_velocity.linear.y", "tcp_velocity.linear.z",
                "tcp_velocity.angular.x", "tcp_velocity.angular.y", "tcp_velocity.angular.z",
                "tcp_error.x", "tcp_error.y", "tcp_error.z",
                "tcp_error.rx", "tcp_error.ry", "tcp_error.rz",
                "joint_positions.0", "joint_positions.1", "joint_positions.2",
                "joint_positions.3", "joint_positions.4", "joint_positions.5",
                "joint_positions.6",
            ],
        },
        "observation.images.center_camera": {
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.left_camera": {
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.right_camera": {
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["linear.x", "linear.y", "linear.z",
                      "angular.x", "angular.y", "angular.z"],
        },
    }

    root = HF_LEROBOT_HOME / REPO_ID
    if root.exists():
        print(f"Removing existing dataset at {root}")
        shutil.rmtree(root)

    total = n_train + n_val
    print(f"Creating dataset: {REPO_ID}")
    print(f"  {n_train} train + {n_val} val  |  {episode_len} steps/ep")
    print(f"  {total * episode_len:,} total frames  |  3 cameras (center, left, right)")
    print(f"  Strategies: {STRATEGIES}\n")

    ds = LeRobotDataset.create(
        repo_id    = REPO_ID,
        fps        = FPS,
        features   = features,
        robot_type = "ur5e_aic",
        use_videos = False,
    )

    strategy_counts = {s: 0 for s in STRATEGIES}

    for ep_idx in range(total):
        # Round-robin strategies within each split for balanced coverage
        strategy = STRATEGIES[ep_idx % len(STRATEGIES)]
        strategy_counts[strategy] += 1

        states, actions, imgs_c, imgs_l, imgs_r, task_str = generate_episode(
            seed=ep_idx + 7919,   # prime offset to avoid overlap with small dataset
            strategy=strategy,
        )

        for step in range(episode_len):
            ds.add_frame({
                "observation.state":                    states[step],
                "observation.images.center_camera":     imgs_c[step],
                "observation.images.left_camera":       imgs_l[step],
                "observation.images.right_camera":      imgs_r[step],
                "action":                               actions[step],
                "task":                                 task_str,
            })
        ds.save_episode()

        if (ep_idx + 1) % 100 == 0 or ep_idx == total - 1:
            split = "train" if ep_idx < n_train else "val"
            print(f"  [{split}] {ep_idx + 1}/{total}")

    print(f"\nDataset saved: {root}")
    print(f"  Total frames : {len(ds):,}")
    print(f"  Strategy distribution: {strategy_counts}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",    type=int, default=N_TRAIN,     help="Train episodes")
    p.add_argument("--val",      type=int, default=N_VAL,       help="Val episodes")
    p.add_argument("--steps",    type=int, default=EPISODE_LEN, help="Steps per episode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(n_train=args.train, n_val=args.val, episode_len=args.steps)
