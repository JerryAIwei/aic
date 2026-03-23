"""
create_aic_dataset.py
Creates a synthetic LeRobot-format dataset that matches the AIC robot's
observation/action space for ACT policy training.

AIC Robot observation space (from aic_robot_aic_controller.py):
  observation.state      : 26-dim float32
      tcp_pose (7D: pos xyz + quat xyzw)
      tcp_velocity (6D: linear xyz + angular xyz)
      tcp_error (6D: x y z rx ry rz)
      joint_positions (7D: 6 arm joints + gripper)

  observation.images.center_camera  : uint8 (256, 288, 3)   [1152×1024 × 0.25]
  observation.images.left_camera    : uint8 (256, 288, 3)
  observation.images.right_camera   : uint8 (256, 288, 3)

  action: 6-dim float32  (cartesian twist: linear.x/y/z + angular.x/y/z)
  fps: 20

Output: ~/.cache/huggingface/lerobot/local/aic_cable_insertion  (LeRobot v3.0)
"""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ─── constants ───────────────────────────────────────────────────────────────
REPO_ID      = "local/aic_cable_insertion"
N_TRAIN      = 30           # reduced for fast creation (no O(n²) image I/O)
N_VAL        = 10
EPISODE_LEN  = 60           # steps per episode
FPS          = 20
IMG_H, IMG_W = 128, 144     # 1152×1024 × 0.125 (quarter of AIC camera, scales up in training)

STATE_DIM    = 26          # 7+6+6+7
ACTION_DIM   = 6           # 6D cartesian twist

# Trajectory waypoints in joint space (for simulation only; actions = cartesian twist)
_WAYPOINTS = [
    np.array([ 0.00, -1.57,  1.57, -1.57, -1.57,  0.00]),   # home
    np.array([ 0.30, -1.20,  1.20, -1.50, -1.50,  0.30]),   # pre-grasp
    np.array([ 0.30, -1.10,  1.10, -1.50, -1.50,  0.30]),   # grasp
    np.array([ 0.30, -1.30,  1.30, -1.60, -1.50,  0.30]),   # lift
    np.array([ 0.60, -1.40,  1.40, -1.70, -1.50,  0.60]),   # approach
    np.array([ 0.60, -1.28,  1.52, -1.84, -1.50,  0.60]),   # insert
]
_WP_T = [0, 0.12, 0.28, 0.42, 0.65, 1.0]


def _smooth(a):
    return 3 * a**2 - 2 * a**3


def _interp_joints(t_arr):
    out = np.zeros((len(t_arr), 6))
    for i, t in enumerate(t_arr):
        for j in range(len(_WP_T) - 1):
            if _WP_T[j] <= t <= _WP_T[j + 1]:
                a = _smooth((t - _WP_T[j]) / (_WP_T[j + 1] - _WP_T[j]))
                out[i] = (1 - a) * _WAYPOINTS[j] + a * _WAYPOINTS[j + 1]
                break
    return out


def _simple_fk(q6):
    """Approx FK → [x, y, z, qx, qy, qz, qw]."""
    L1, L2 = 0.425, 0.39
    x = (L1 * np.cos(q6[1]) + L2 * np.cos(q6[1] + q6[2])) * np.cos(q6[0])
    y = (L1 * np.cos(q6[1]) + L2 * np.cos(q6[1] + q6[2])) * np.sin(q6[0])
    z = 0.7 + 0.1 * np.sin(q6[0])
    # tiny rotation quaternion derived from wrist joints
    rx, ry, rz = q6[3] * 0.5, q6[4] * 0.5, q6[5] * 0.5
    w = (1 - rx**2 - ry**2 - rz**2) ** 0.5 if (rx**2 + ry**2 + rz**2) < 1 else 0.0
    return np.array([x, y, z, rx, ry, rz, w], dtype=np.float32)


def _cartesian_twist(q_curr, q_next):
    """Approximate cartesian twist from joint delta (action)."""
    dx = q_next - q_curr
    # Map through simplified Jacobian columns (very rough but correlated)
    J = np.eye(6)[:, :6] * 0.3
    twist = J @ dx
    return twist.astype(np.float32)  # (6,)


def generate_episode(seed: int):
    rng = np.random.RandomState(seed)
    t   = np.linspace(0, 1, EPISODE_LEN)
    q   = _interp_joints(t)   # (T, 6) arm joints

    # Per-episode variation
    q  += rng.randn(*q.shape) * 0.02
    q  += (rng.randn(6) * 0.04)[None, :]
    gripper = np.where(t < 0.28, 1.0, -0.5).astype(np.float32)  # open / closed

    states  = []
    actions = []
    images  = {
        "observation.images.center_camera": [],
    }

    for i in range(EPISODE_LEN):
        tcp_pose = _simple_fk(q[i])                                    # 7

        # Velocity: numerical diff
        if i < EPISODE_LEN - 1:
            dq = (q[i + 1] - q[i]) * FPS
        else:
            dq = np.zeros(6)
        tcp_vel = np.concatenate([dq[:3] * 0.3, dq[3:] * 0.2])        # 6

        tcp_err = rng.randn(6).astype(np.float32) * 0.002              # 6
        jpos    = np.append(q[i], gripper[i]).astype(np.float32)       # 7

        state = np.concatenate([tcp_pose, tcp_vel, tcp_err, jpos]).astype(np.float32)  # 26
        states.append(state)

        # Action = cartesian twist toward next step
        qn = q[i + 1] if i < EPISODE_LEN - 1 else q[i]
        actions.append(_cartesian_twist(q[i], qn))

        # Synthetic images – center camera only (add left/right when using real robot)
        for cam_key, hue_shift in [
            ("observation.images.center_camera", 0),
        ]:
            img = _render_frame(tcp_pose, rng, hue_shift)
            images[cam_key].append(img)

    return np.array(states), np.array(actions), images


def _render_frame(tcp_pose, rng, hue_shift):
    """Render a simple top-down image: grey background + coloured TCP dot."""
    img = np.full((IMG_H, IMG_W, 3), 200, dtype=np.uint8)

    # Map tcp x,y → pixel
    cx = int(np.clip((tcp_pose[0] + 0.6) / 1.2 * IMG_W, 2, IMG_W - 3))
    cy = int(np.clip((tcp_pose[1] + 0.6) / 1.2 * IMG_H, 2, IMG_H - 3))

    # Draw dot (radius 8)
    yg, xg = np.ogrid[:IMG_H, :IMG_W]
    mask = (xg - cx)**2 + (yg - cy)**2 <= 64
    base_r = max(0, min(255, 220 + hue_shift))
    img[mask, 0] = base_r
    img[mask, 1] = 50
    img[mask, 2] = 50

    # Gaussian noise
    noise = rng.randint(-8, 8, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Dataset features spec (LeRobot v3.0 format)
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
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [
                "linear.x", "linear.y", "linear.z",
                "angular.x", "angular.y", "angular.z",
            ],
        },
    }

    # Delete existing dataset if present
    from lerobot.utils.constants import HF_LEROBOT_HOME
    dataset_root = HF_LEROBOT_HOME / REPO_ID
    if dataset_root.exists():
        print(f"Removing existing dataset at {dataset_root}")
        shutil.rmtree(dataset_root)

    print(f"Creating LeRobot dataset: {REPO_ID}")
    print(f"  state dim: {STATE_DIM}  |  action dim: {ACTION_DIM}")
    print(f"  images: {IMG_H}×{IMG_W} (center camera only)  |  fps: {FPS}")
    print(f"  episodes: {N_TRAIN} train + {N_VAL} val  |  steps/ep: {EPISODE_LEN}")
    print("  (reduce image resolution and episode count to speed up LeRobot dataset creation)\n")

    ds = LeRobotDataset.create(
        repo_id     = REPO_ID,
        fps         = FPS,
        features    = features,
        robot_type  = "ur5e_aic",
        use_videos  = False,   # store as images (faster for synthetic data)
    )

    total = N_TRAIN + N_VAL
    for ep_idx in range(total):
        states, actions, images_dict = generate_episode(seed=ep_idx)

        for step in range(EPISODE_LEN):
            frame = {
                "observation.state": states[step],
                "action":            actions[step],
                "task":              "cable_insertion",
            }
            for cam_key, frames_list in images_dict.items():
                frame[cam_key] = frames_list[step]

            ds.add_frame(frame)

        ds.save_episode()

        if (ep_idx + 1) % 20 == 0 or ep_idx == total - 1:
            split = "train" if ep_idx < N_TRAIN else "val"
            print(f"  [{split}] episode {ep_idx + 1}/{total}")

    print(f"\nDataset saved to: {dataset_root}")
    print(f"  Total frames: {len(ds)}")


if __name__ == "__main__":
    main()
