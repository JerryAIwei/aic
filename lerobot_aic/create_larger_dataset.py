"""
create_larger_dataset.py
Creates a LeRobot-format dataset for AIC cable-insertion training.
Compatible with https://github.com/huggingface/lerobot training pipelines.

Dataset: local/aic_cable_insertion_large
  - 80 train + 20 val = 100 episodes  (100 steps each → 10,000 total frames)
  - use_videos=True  →  images stored as mp4 per camera (linear creation time)
  - 3 cameras: center, left, right  (matching real AIC robot spec)
  - 5 trajectory strategies for diversity
  - 26-dim state  |  6-dim cartesian-twist action

Usage:
    python create_larger_dataset.py
    python create_larger_dataset.py --train 80 --val 20 --steps 100
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

# ─── Dataset parameters ───────────────────────────────────────────────────────
REPO_ID     = "local/aic_cable_insertion_large"
N_TRAIN     = 80
N_VAL       = 20
EPISODE_LEN = 100
FPS         = 20
IMG_H       = 128
IMG_W       = 144
STATE_DIM   = 26
ACTION_DIM  = 6

# ─── Trajectory waypoints in joint space ──────────────────────────────────────
_WAYPOINTS_BASE = [
    np.array([ 0.00, -1.57,  1.57, -1.57, -1.57,  0.00]),
    np.array([ 0.30, -1.20,  1.20, -1.50, -1.50,  0.30]),
    np.array([ 0.30, -1.10,  1.10, -1.50, -1.50,  0.30]),
    np.array([ 0.30, -1.30,  1.30, -1.60, -1.50,  0.30]),
    np.array([ 0.60, -1.40,  1.40, -1.70, -1.50,  0.60]),
    np.array([ 0.60, -1.28,  1.52, -1.84, -1.50,  0.60]),
]
_WP_T_BASE = [0.0, 0.12, 0.28, 0.42, 0.65, 1.0]

STRATEGIES = [
    "straight_approach",
    "curved_approach",
    "slow_insertion",
    "angled_approach",
    "recovery",
]


# ─── Trajectory helpers ───────────────────────────────────────────────────────

def _smooth(a: float) -> float:
    return 3 * a**2 - 2 * a**3


def _interp(t_arr: np.ndarray, waypoints, wp_t) -> np.ndarray:
    out = np.zeros((len(t_arr), 6))
    for i, t in enumerate(t_arr):
        t = float(np.clip(t, wp_t[0], wp_t[-1]))
        for j in range(len(wp_t) - 1):
            if wp_t[j] <= t <= wp_t[j + 1]:
                a = _smooth((t - wp_t[j]) / (wp_t[j + 1] - wp_t[j]))
                out[i] = (1 - a) * waypoints[j] + a * waypoints[j + 1]
                break
    return out


def _build_trajectory(strategy: str, rng: np.random.RandomState):
    wps = [w.copy() for w in _WAYPOINTS_BASE]
    wpt = list(_WP_T_BASE)

    if strategy == "straight_approach":
        pass

    elif strategy == "curved_approach":
        extra = wps[3].copy()
        extra[0] += rng.uniform(0.05, 0.12)
        extra[1] += rng.uniform(-0.08, 0.08)
        wps.insert(4, extra)
        wpt = [0.0, 0.12, 0.28, 0.42, 0.54, 0.65, 1.0]

    elif strategy == "slow_insertion":
        wpt = [0.0, 0.10, 0.25, 0.40, 0.60, 1.0]

    elif strategy == "angled_approach":
        wps[0][0] += rng.uniform(-0.15, 0.15)
        wps[4][0] += rng.uniform(-0.10, 0.10)
        wps[5][0] += rng.uniform(-0.05, 0.05)

    elif strategy == "recovery":
        error = rng.randn(6) * 0.08
        wps[4] = wps[4] + error
        correction = wps[4] * 0.3 + wps[5] * 0.7
        wps.insert(5, correction)
        wpt = [0.0, 0.12, 0.28, 0.42, 0.65, 0.80, 1.0]

    return wps, wpt


def _fk(q6: np.ndarray) -> np.ndarray:
    L1, L2 = 0.425, 0.39
    x = (L1 * np.cos(q6[1]) + L2 * np.cos(q6[1] + q6[2])) * np.cos(q6[0])
    y = (L1 * np.cos(q6[1]) + L2 * np.cos(q6[1] + q6[2])) * np.sin(q6[0])
    z = 0.7 + 0.1 * np.sin(q6[0])
    rx, ry, rz = q6[3] * 0.5, q6[4] * 0.5, q6[5] * 0.5
    w = max(0.0, 1.0 - rx**2 - ry**2 - rz**2) ** 0.5
    return np.array([x, y, z, rx, ry, rz, w], dtype=np.float32)


def _twist(q_curr: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    return ((q_next - q_curr) * 0.3).astype(np.float32)


# ─── Rendering ────────────────────────────────────────────────────────────────

_SOCKET_X, _SOCKET_Y = 0.45, 0.10


def _render_center(tcp, rng):
    img = np.full((IMG_H, IMG_W, 3), 195, dtype=np.uint8)
    yg, xg = np.ogrid[:IMG_H, :IMG_W]
    sx = int(np.clip((_SOCKET_X + 0.6) / 1.2 * IMG_W, 2, IMG_W - 3))
    sy = int(np.clip((_SOCKET_Y + 0.6) / 1.2 * IMG_H, 2, IMG_H - 3))
    img[(xg - sx)**2 + (yg - sy)**2 <= 25, 2] = 220
    cx = int(np.clip((tcp[0] + 0.6) / 1.2 * IMG_W, 2, IMG_W - 3))
    cy = int(np.clip((tcp[1] + 0.6) / 1.2 * IMG_H, 2, IMG_H - 3))
    img[(xg - cx)**2 + (yg - cy)**2 <= 64, 0] = 220
    img[(xg - cx)**2 + (yg - cy)**2 <= 64, 1] = 50
    img[(xg - cx)**2 + (yg - cy)**2 <= 64, 2] = 50
    noise = rng.randint(-10, 10, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _render_left(tcp, rng):
    img = np.full((IMG_H, IMG_W, 3), 210, dtype=np.uint8)
    yg, xg = np.ogrid[:IMG_H, :IMG_W]
    cx = int(np.clip((tcp[0] + 0.6) / 1.2 * IMG_W, 2, IMG_W - 3))
    cz = int(np.clip((tcp[2] + 0.1) / 1.4 * IMG_H, 2, IMG_H - 3))
    img[(xg - cx)**2 + (yg - cz)**2 <= 64, 0] = 200
    img[(xg - cx)**2 + (yg - cz)**2 <= 64, 1] = 60
    noise = rng.randint(-8, 8, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _render_right(tcp, rng):
    img = np.full((IMG_H, IMG_W, 3), 205, dtype=np.uint8)
    yg, xg = np.ogrid[:IMG_H, :IMG_W]
    cy = int(np.clip((tcp[1] + 0.6) / 1.2 * IMG_W, 2, IMG_W - 3))
    cz = int(np.clip((tcp[2] + 0.1) / 1.4 * IMG_H, 2, IMG_H - 3))
    img[(xg - cy)**2 + (yg - cz)**2 <= 64, 0] = 200
    img[(xg - cy)**2 + (yg - cz)**2 <= 64, 1] = 60
    noise = rng.randint(-8, 8, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ─── Episode generation ───────────────────────────────────────────────────────

def generate_episode(seed: int, strategy: str, episode_len: int):
    rng = np.random.RandomState(seed)
    wps, wpt = _build_trajectory(strategy, rng)
    t = np.linspace(0, 1, episode_len)
    speed = rng.uniform(0.80, 1.20)
    t_scaled = np.clip(t * speed, 0.0, 1.0)
    q = _interp(t_scaled, wps, wpt)
    q += rng.randn(*q.shape) * 0.020
    q += (rng.randn(6) * 0.05)[None, :]
    gripper = np.where(t < 0.28, 1.0, -0.5).astype(np.float32)

    states, actions = [], []
    imgs_c, imgs_l, imgs_r = [], [], []

    for i in range(episode_len):
        tcp = _fk(q[i])
        dq = (q[i + 1] - q[i]) * FPS if i < episode_len - 1 else np.zeros(6)
        tcp_vel = np.concatenate([dq[:3] * 0.3, dq[3:] * 0.2])
        tcp_err = rng.randn(6).astype(np.float32) * 0.002
        jpos    = np.append(q[i], gripper[i]).astype(np.float32)
        state   = np.concatenate([tcp, tcp_vel, tcp_err, jpos]).astype(np.float32)
        states.append(state)
        qn = q[i + 1] if i < episode_len - 1 else q[i]
        actions.append(_twist(q[i], qn))
        imgs_c.append(_render_center(tcp, rng))
        imgs_l.append(_render_left(tcp, rng))
        imgs_r.append(_render_right(tcp, rng))

    return (
        np.array(states,  dtype=np.float32),
        np.array(actions, dtype=np.float32),
        imgs_c, imgs_l, imgs_r,
        f"cable_insertion_{strategy}",
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(n_train: int = N_TRAIN, n_val: int = N_VAL, episode_len: int = EPISODE_LEN):
    import time
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
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.left_camera": {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.right_camera": {
            "dtype": "video",
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
    print(f"  {n_train} train + {n_val} val  |  {episode_len} steps/ep  |  {total*episode_len:,} frames")
    print(f"  3 cameras (video)  |  5 trajectory strategies\n")

    ds = LeRobotDataset.create(
        repo_id    = REPO_ID,
        fps        = FPS,
        features   = features,
        robot_type = "ur5e_aic",
        use_videos = True,   # mp4 per camera — O(n) creation time
    )

    t0 = time.time()
    strategy_counts = {s: 0 for s in STRATEGIES}

    for ep_idx in range(total):
        strategy = STRATEGIES[ep_idx % len(STRATEGIES)]
        strategy_counts[strategy] += 1

        states, actions, imgs_c, imgs_l, imgs_r, task_str = generate_episode(
            seed=ep_idx + 7919,
            strategy=strategy,
            episode_len=episode_len,
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

        if (ep_idx + 1) % 10 == 0 or ep_idx == total - 1:
            split = "train" if ep_idx < n_train else "val"
            elapsed = time.time() - t0
            rate = (ep_idx + 1) / elapsed
            eta  = (total - ep_idx - 1) / rate if rate > 0 else 0
            print(f"  [{split}] {ep_idx+1:>3}/{total}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    print(f"\nDataset saved: {root}")
    print(f"  Total frames    : {len(ds):,}")
    print(f"  Total time      : {time.time()-t0:.0f}s")
    print(f"  Strategy counts : {strategy_counts}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=N_TRAIN)
    p.add_argument("--val",   type=int, default=N_VAL)
    p.add_argument("--steps", type=int, default=EPISODE_LEN)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(n_train=args.train, n_val=args.val, episode_len=args.steps)
