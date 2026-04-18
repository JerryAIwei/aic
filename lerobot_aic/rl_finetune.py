"""
rl_finetune.py — Reward-Weighted Regression (RWR) fine-tuning for AIC Diffusion Policy.

Algorithm
─────────
1. COLLECT  : Run CheatCode at hard difficulty (board ±8 cm, cable ±5°).
              Parse AIC score components from engine logs per episode.
2. SCORE    : Compute partial AIC score (Tier1 + Tier3 + duration + force + contact).
              Assign per-episode weight = exp(score / temperature).
3. TRAIN    : Load iter10/best_model. Fine-tune with score-weighted episode sampling
              (WeightedRandomSampler biases gradient steps toward high-quality episodes).
4. EVALUATE : Run N trials at hard difficulty; report pre/post AIC score comparison.

Why RWR?
─────────
Standard imitation learning treats all demonstrations equally. RWR re-weights the
training distribution so that faster, gentler insertions (higher AIC score) contribute
more to the gradient than slower or rougher ones. This is a practical on-policy RL
approximation that works within the existing LeRobot training stack.

Usage
─────
    python rl_finetune.py \\
        --base_ckpt     lerobot_aic/checkpoints_diffusion_iter10/best_model \\
        --base_dataset  local/aic_cable_insertion_iter10 \\
        --n_episodes    20 \\
        --steps         5000 \\
        --n_eval        5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

from scoring import compute_trial_score

# ── Constants ─────────────────────────────────────────────────────────────────

ROOTFS    = "/opt/aic_rootfs"
ROS_SETUP = "/ws_aic/install/setup.bash"
WORKSPACE = Path("/workspace/aic")

SAVE_DIR   = Path("/tmp/aic_recordings")   # must match RecordCheatCode.SAVE_DIR
CONFIG_DIR = Path("/tmp/aic_rl_configs")
LOG_DIR    = Path("/tmp/aic_rl_logs")

_CABLE = dict(roll=0.4432, pitch=-0.4838, yaw=1.3303)
_BOARD = dict(x=0.15, y=-0.2, z=1.14, yaw=3.1415)

IMG_H, IMG_W = 128, 144
STATE_DIM, ACTION_DIM = 26, 6
FPS = 20
N_OBS, HORIZON, N_ACTION = 2, 16, 8

# RWR temperature: lower → sharper weighting, higher → flatter (uniform)
RWR_TEMPERATURE = 10.0
# Min weight ratio to prevent zero-weight episodes being starved entirely
MIN_WEIGHT_RATIO = 0.1


# ── ROS environment ───────────────────────────────────────────────────────────

def _ros_env(extra: dict | None = None) -> dict:
    env = os.environ.copy()
    rootfs_libs = (f"{ROOTFS}/usr/lib/x86_64-linux-gnu:"
                   f"{ROOTFS}/lib/x86_64-linux-gnu:{ROOTFS}/usr/lib")
    ogre = f"{ROOTFS}/usr/lib/x86_64-linux-gnu/OGRE-2.3"
    gz_v = f"{ROOTFS}/opt/ros/kilted/opt/gz_ogre_next_vendor/lib"
    sys  = "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"]            = f"{sys}:{rootfs_libs}:{ogre}:{gz_v}:{env.get('LD_LIBRARY_PATH','')}"
    env["RMW_IMPLEMENTATION"]          = "rmw_zenoh_cpp"
    env["ZENOH_ROUTER_CHECK_ATTEMPTS"] = "-1"
    env["ZENOH_CONFIG_OVERRIDE"]       = "transport/shared_memory/enabled=false"
    env["OGRE2_RESOURCE_PATH"]         = "/usr/lib/x86_64-linux-gnu/OGRE-2.3/OGRE"
    env.pop("DISPLAY", None)
    rootfs_py = (f"{ROOTFS}/usr/lib/python3/dist-packages:"
                 f"{ROOTFS}/usr/lib/python3.12/dist-packages")
    env["PYTHONPATH"] = f"{rootfs_py}:{env.get('PYTHONPATH','')}"
    if extra:
        env.update(extra)
    return env


def _popen(cmd: str, log: Path | None = None, env: dict | None = None) -> subprocess.Popen:
    if log:
        cmd = f"({cmd}) 2>&1 | tee {log}"
    return subprocess.Popen(
        ["bash", "-c", f". {ROS_SETUP} && {cmd}"],
        env=env or _ros_env(), start_new_session=True,
    )


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _cleanup() -> None:
    for pat in ["aic_model", "rmw_zenohd", "gz_server", "gzserver",
                "component_container", "aic_engine", "aic_adapter"]:
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    time.sleep(4)


# ── Scene config (hard difficulty) ────────────────────────────────────────────

def gen_hard_scenarios(n: int, seed: int = 1337) -> list[dict]:
    """Generate N hard scenarios: board ±8 cm, cable ±5° — beyond training distribution."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        out.append({
            "task_board_x":   round(_BOARD["x"]   + rng.uniform(-0.080, 0.080), 4),
            "task_board_y":   round(_BOARD["y"]   + rng.uniform(-0.080, 0.080), 4),
            "task_board_z":   _BOARD["z"],
            "task_board_yaw": round(_BOARD["yaw"] + rng.uniform(-0.160, 0.160), 4),
            "cable_roll":     round(_CABLE["roll"]  + rng.uniform(-0.087, 0.087), 4),
            "cable_pitch":    round(_CABLE["pitch"] + rng.uniform(-0.087, 0.087), 4),
            "cable_yaw":      round(_CABLE["yaw"]   + rng.uniform(-0.087, 0.087), 4),
        })
    return out


def _write_config(scenario: dict, path: Path) -> None:
    sc = scenario
    yaml = f"""# Auto-generated RL collection config
scoring:
  topics:
    - topic:
        name: "/joint_states"
        type: "sensor_msgs/msg/JointState"
    - topic:
        name: "/tf"
        type: "tf2_msgs/msg/TFMessage"
    - topic:
        name: "/tf_static"
        type: "tf2_msgs/msg/TFMessage"
        latched: true
    - topic:
        name: "/scoring/tf"
        type: "tf2_msgs/msg/TFMessage"
    - topic:
        name: "/aic/gazebo/contacts/off_limit"
        type: "ros_gz_interfaces/msg/Contacts"
    - topic:
        name: "/fts_broadcaster/wrench"
        type: "geometry_msgs/msg/WrenchStamped"
    - topic:
        name: "/aic_controller/joint_commands"
        type: "aic_control_interfaces/msg/JointMotionUpdate"
    - topic:
        name: "/aic_controller/pose_commands"
        type: "aic_control_interfaces/msg/MotionUpdate"
    - topic:
        name: "/scoring/insertion_event"
        type: "std_msgs/msg/String"
    - topic:
        name: "/aic_controller/controller_state"
        type: "aic_control_interfaces/msg/ControllerState"

task_board_limits:
  nic_rail:
    min_translation: -0.0215
    max_translation: 0.0234
  sc_rail:
    min_translation: -0.06
    max_translation: 0.055
  mount_rail:
    min_translation: -0.09425
    max_translation: 0.09425

trials:
  trial_1:
    scene:
        task_board:
          pose:
            x: {sc['task_board_x']}
            y: {sc['task_board_y']}
            z: {sc['task_board_z']}
            roll: 0.0
            pitch: 0.0
            yaw: {sc['task_board_yaw']}
          nic_rail_0:
            entity_present: True
            entity_name: "nic_card_0"
            entity_pose:
              translation: 0.036
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
          nic_rail_1:
            entity_present: False
          nic_rail_2:
            entity_present: False
          nic_rail_3:
            entity_present: False
          nic_rail_4:
            entity_present: False
          sc_rail_0:
            entity_present: True
            entity_name: "sc_mount_0"
            entity_pose:
              translation: 0.042
              roll: 0.0
              pitch: 0.0
              yaw: 0.1
          sc_rail_1:
            entity_present: False
          lc_mount_rail_0:
            entity_present: True
            entity_name: "lc_mount_0"
            entity_pose:
              translation: 0.02
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
          sfp_mount_rail_0:
            entity_present: True
            entity_name: "sfp_mount_0"
            entity_pose:
              translation: 0.03
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
          sc_mount_rail_0:
            entity_present: True
            entity_name: "sc_mount_0"
            entity_pose:
              translation: -0.02
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
          lc_mount_rail_1:
            entity_present: True
            entity_name: "lc_mount_1"
            entity_pose:
              translation: -0.01
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
          sfp_mount_rail_1:
            entity_present: False
          sc_mount_rail_1:
            entity_present: False
        cables:
          cable_0:
            pose:
              gripper_offset:
                x: 0.0
                y: 0.015385
                z: 0.04245
              roll: {sc['cable_roll']}
              pitch: {sc['cable_pitch']}
              yaw: {sc['cable_yaw']}
            attach_cable_to_gripper: True
            cable_type: "sfp_sc_cable"
    tasks:
      task_1:
        cable_type: "sfp_sc"
        cable_name: "cable_0"
        plug_type: "sfp"
        plug_name: "sfp_tip"
        port_type: "sfp"
        port_name: "sfp_port_0"
        target_module_name: "nic_card_mount_0"
        time_limit: 240

robot:
  home_joint_positions:
    shoulder_pan_joint: -0.1597
    shoulder_lift_joint: -1.3542
    elbow_joint: -1.6648
    wrist_1_joint: -1.6933
    wrist_2_joint: 1.5710
    wrist_3_joint: 1.4110
"""
    path.write_text(yaml)


# ── Log parsing ───────────────────────────────────────────────────────────────

def _parse_episode_logs(log_dir: Path) -> dict:
    """Extract AIC score components from engine/policy logs."""
    raw = {"valid": False, "insertion_result": "none",
           "duration_s": None, "max_force_N": None,
           "has_off_limit_contact": False}
    for log in log_dir.glob("*.log"):
        try:
            text = log.read_text(errors="ignore")
        except Exception:
            continue
        if re.search(r"activat|lifecycle.*active", text, re.IGNORECASE):
            raw["valid"] = True
        if re.search(r"total score is:\s*1\.0", text, re.IGNORECASE):
            raw["valid"] = True
            raw["insertion_result"] = "correct"
        if re.search(r"insertion_event|cable.*insert.*success", text, re.IGNORECASE):
            raw["valid"] = True
            if raw["insertion_result"] == "none":
                raw["insertion_result"] = "correct"
        m = re.search(r"elapsed[_\s]seconds?[:\s=]+([0-9.]+)", text, re.IGNORECASE)
        if m and raw["duration_s"] is None:
            raw["duration_s"] = float(m.group(1))
        m = re.search(r"max[_\s]force[:\s=]+([0-9.]+)", text, re.IGNORECASE)
        if m and raw["max_force_N"] is None:
            raw["max_force_N"] = float(m.group(1))
        if re.search(r"off.limit contact|off_limit.*detect", text, re.IGNORECASE):
            raw["has_off_limit_contact"] = True
    return raw


def _compute_episode_score(raw: dict) -> float:
    """Compute partial AIC score from log-parsed metrics."""
    trial = {
        "valid":                 raw["valid"],
        "insertion_result":      raw["insertion_result"],
        "duration_s":            raw["duration_s"],
        "has_off_limit_contact": raw["has_off_limit_contact"],
    }
    if raw["max_force_N"] is not None and raw["max_force_N"] > 20.0:
        trial["force_magnitudes"]  = np.array([raw["max_force_N"]] * 2)
        trial["force_timestamps"]  = np.array([0.0, 2.0])
    else:
        trial["force_magnitudes"]  = np.array([0.0, 0.0])
        trial["force_timestamps"]  = np.array([0.0, 1.0])
    s = compute_trial_score(trial)
    return float(s["total"])


# ── Phase 1: Collect ─────────────────────────────────────────────────────────

def _wait_for_episode(timeout: int = 200) -> Path | None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    latest = SAVE_DIR / "latest.txt"
    latest.unlink(missing_ok=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if latest.exists():
            text = latest.read_text().strip()
            if text:
                p = Path(text)
                if p.exists():
                    return p
        time.sleep(2)
    return None


def collect_rl_episodes(n: int, seed: int, log_dir: Path) -> list[dict]:
    """
    Collect n CheatCode demonstrations at hard difficulty.
    Returns list of {npz_data, score, raw_metrics}.
    """
    CONFIG_DIR.mkdir(exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    scenarios = gen_hard_scenarios(n, seed=seed)
    episodes = []

    print(f"\n{'─'*60}")
    print(f"  Phase 1: Collect {n} episodes at hard difficulty")
    print(f"  Difficulty: board ±8 cm / ±9.2°, cable ±5°")
    print(f"{'─'*60}")

    for i, sc in enumerate(scenarios):
        _cleanup()
        ep_num = i + 1
        ep_log_dir = log_dir / f"collect_{ep_num:02d}"
        ep_log_dir.mkdir(exist_ok=True)
        cfg_path = CONFIG_DIR / f"collect_{ep_num:02d}.yaml"
        _write_config(sc, cfg_path)
        env = _ros_env()

        print(f"\n  [{ep_num:2d}/{n}] board=({sc['task_board_x']:.3f},{sc['task_board_y']:.3f}) "
              f"yaw={sc['task_board_yaw']:.3f}")

        zenoh = _popen("ros2 run rmw_zenoh_cpp rmw_zenohd",
                       log=ep_log_dir / "zenoh.log", env=env)
        time.sleep(3)
        policy = _popen(
            "ros2 run aic_model aic_model "
            "--ros-args -p use_sim_time:=true "
            "-p policy:=aic_example_policies.ros.RecordCheatCode",
            log=ep_log_dir / "policy.log", env=env,
        )
        time.sleep(2)
        sim = _popen(
            "ros2 launch aic_bringup aic_gz_bringup.launch.py "
            "gazebo_gui:=false ground_truth:=true "
            "spawn_task_board:=false start_aic_engine:=true "
            "shutdown_on_aic_engine_exit:=true "
            f"aic_engine_config_file:={cfg_path}",
            log=ep_log_dir / "sim.log", env=env,
        )
        time.sleep(40)
        ep_path = _wait_for_episode(timeout=200)
        for proc in [policy, sim, zenoh]:
            _kill(proc)
        time.sleep(3)

        if ep_path is None:
            print(f"    → TIMEOUT, skipping")
            continue

        raw  = _parse_episode_logs(ep_log_dir)
        sc_  = _compute_episode_score(raw)
        data = np.load(str(ep_path))
        T    = len(data["states"])
        episodes.append({
            "states":  data["states"],
            "actions": data["actions"],
            "center":  data["center"],
            "left":    data["left"],
            "right":   data["right"],
            "score":   sc_,
            "raw":     raw,
        })
        print(f"    → {T} steps  score={sc_:.1f}  "
              f"success={raw['insertion_result']}  "
              f"dur={raw['duration_s']}s  force={raw['max_force_N']}N")

    scores = [e["score"] for e in episodes]
    if scores:
        print(f"\n  Collected {len(episodes)} episodes")
        print(f"  Score: mean={np.mean(scores):.1f}  "
              f"min={np.min(scores):.1f}  max={np.max(scores):.1f}")
    return episodes


# ── Phase 2: Build weighted dataset ───────────────────────────────────────────

def build_rl_dataset(new_episodes: list[dict], base_repo_id: str,
                     rl_repo_id: str) -> tuple[LeRobotDataset, list[float]]:
    """
    Combine base dataset with new scored episodes.
    Returns (dataset, per_frame_weights).
    """
    from lerobot.utils.constants import HF_LEROBOT_HOME

    rl_root = HF_LEROBOT_HOME / rl_repo_id
    if rl_root.exists():
        shutil.rmtree(rl_root)

    # Features (same as collect_sim_demos.py)
    features = {
        "observation.state": {
            "dtype": "float32", "shape": (STATE_DIM,),
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
            "dtype": "video", "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.left_camera": {
            "dtype": "video", "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.right_camera": {
            "dtype": "video", "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32", "shape": (ACTION_DIM,),
            "names": ["linear.x", "linear.y", "linear.z",
                      "angular.x", "angular.y", "angular.z"],
        },
    }

    print(f"\n{'─'*60}")
    print(f"  Phase 2: Building RL dataset '{rl_repo_id}'")
    print(f"{'─'*60}")

    # Load base dataset to count episodes and their frame lengths
    base_ds = LeRobotDataset(
        repo_id=base_repo_id,
        delta_timestamps={
            "observation.state": [0.0, -1/FPS],
            "observation.images.center_camera": [0.0, -1/FPS],
            "observation.images.left_camera":   [0.0, -1/FPS],
            "observation.images.right_camera":  [0.0, -1/FPS],
            "action": [i/FPS for i in range(HORIZON)],
        },
        video_backend="pyav",
    )
    n_base_episodes = base_ds.num_episodes
    n_base_frames   = len(base_ds)

    # Compute RWR weights for new episodes
    # Base dataset episodes get weight = mean_new_score (neutral)
    new_scores = np.array([e["score"] for e in new_episodes], dtype=float)
    if len(new_scores) > 0:
        # Normalize scores then exponentiate for soft weighting
        score_mean = float(np.mean(new_scores))
        score_std  = max(float(np.std(new_scores)), 1.0)
        norm_scores = (new_scores - score_mean) / score_std
        raw_weights = np.exp(norm_scores / (RWR_TEMPERATURE / score_std))
        # Clip: min weight is MIN_WEIGHT_RATIO × max_weight
        max_w = float(np.max(raw_weights))
        ep_weights_new = np.clip(raw_weights, MIN_WEIGHT_RATIO * max_w, None)
        # Base episodes get weight = 1.0 (standard)
        ep_weights_base = np.ones(n_base_episodes)
        all_ep_weights = np.concatenate([ep_weights_base,
                                         ep_weights_new / ep_weights_new.mean()])
    else:
        ep_weights_base = np.ones(n_base_episodes)
        all_ep_weights = ep_weights_base

    # Build combined LeRobot dataset by adding new episodes
    # We create a fresh dataset and copy base + new episodes into it
    ds = LeRobotDataset.create(
        repo_id=rl_repo_id, fps=FPS, features=features,
        robot_type="ur5e_aic", use_videos=True,
    )

    # Add base dataset frames
    print(f"  Copying {n_base_frames:,} frames from base dataset …")
    # NOTE: We can't directly copy from an existing LeRobotDataset to a new one
    # without iterating frame by frame (LeRobotDataset.create vs existing).
    # For practical purposes, we directly link to base dataset for training
    # and only add new episodes. Weighted sampler will handle the rest.
    # Frame-level copy would be too slow for large datasets.
    # → We skip the base copy and instead use a split approach in training:
    #    train on base_ds + new episodes with per-episode weights.
    del ds  # discard the empty combined dataset

    print(f"  New RL episodes: {len(new_episodes)} "
          f"(scores: {new_scores.min():.1f}–{new_scores.max():.1f})")
    print(f"  Will use base dataset + new episodes in weighted training.")

    # Build new-episodes-only dataset for combining
    ds_new = LeRobotDataset.create(
        repo_id=rl_repo_id, fps=FPS, features=features,
        robot_type="ur5e_aic", use_videos=True,
    )
    ep_frame_counts = []
    for ep in new_episodes:
        T = len(ep["states"])
        for t in range(T):
            ds_new.add_frame({
                "observation.state": ep["states"][t],
                "observation.images.center_camera": ep["center"][t],
                "observation.images.left_camera":   ep["left"][t],
                "observation.images.right_camera":  ep["right"][t],
                "action":                           ep["actions"][t],
                "task": "cable_insertion_rl",
            })
        ds_new.save_episode()
        ep_frame_counts.append(T)
        print(f"  added episode: {T} steps, score={ep['score']:.1f}")

    print(f"  RL dataset: {rl_root}  ({len(ds_new):,} new frames)")
    return base_ds, ds_new, all_ep_weights, ep_frame_counts


# ── Phase 3: Fine-tune with RWR ───────────────────────────────────────────────

def make_diffusion_config() -> DiffusionConfig:
    return DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(26,)),
            "observation.images.center_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 144)),
            "observation.images.left_camera":   PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 144)),
            "observation.images.right_camera":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 144)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        normalization_mapping={
            "STATE":  NormalizationMode.MIN_MAX,
            "VISUAL": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        n_obs_steps=N_OBS, horizon=HORIZON, n_action_steps=N_ACTION,
        drop_n_last_frames=HORIZON - N_ACTION,
        vision_backbone="resnet18", pretrained_backbone_weights=None,
        use_group_norm=True, spatial_softmax_num_keypoints=32,
        resize_shape=None, crop_shape=None, crop_ratio=1.0,
        down_dims=(256, 512, 1024), kernel_size=5, n_groups=8,
        diffusion_step_embed_dim=128, use_film_scale_modulation=True,
        noise_scheduler_type="DDPM", num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2", beta_start=0.0001, beta_end=0.02,
        prediction_type="epsilon", clip_sample=True, clip_sample_range=1.0,
        do_mask_loss_for_padding=False,
    )


def _build_weighted_loader(
    base_ds: LeRobotDataset,
    ds_new: LeRobotDataset,
    new_ep_scores: list[float],
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """
    Build a weighted DataLoader that mixes base dataset + new RL episodes.
    High-scoring new episodes are sampled more frequently (RWR).
    """
    delta_ts = {
        "observation.state": [0.0, -1/FPS],
        "observation.images.center_camera": [0.0, -1/FPS],
        "observation.images.left_camera":   [0.0, -1/FPS],
        "observation.images.right_camera":  [0.0, -1/FPS],
        "action": [i/FPS for i in range(HORIZON)],
    }

    # Compute per-frame weights for the new dataset
    n_new_frames = len(ds_new)
    n_new_ep     = ds_new.num_episodes

    if new_ep_scores:
        scores_arr = np.array(new_ep_scores, dtype=float)
        score_mean = float(np.mean(scores_arr))
        score_std  = max(float(np.std(scores_arr)), 1.0)
        norm       = (scores_arr - score_mean) / score_std
        raw_w      = np.exp(norm)
        max_w      = float(np.max(raw_w))
        ep_w       = np.clip(raw_w, MIN_WEIGHT_RATIO * max_w, None)
        ep_w       = ep_w / ep_w.mean()  # normalize to mean=1
    else:
        ep_w = np.ones(max(n_new_ep, 1))

    # Map frame → episode → weight
    ep_idx_col = ds_new.hf_dataset["episode_index"]
    frame_weights_new = np.array([ep_w[min(ei, len(ep_w)-1)] for ei in ep_idx_col])

    # Base dataset frames: uniform weight = 1.0 per frame
    ep_idx_base = base_ds.hf_dataset["episode_index"]
    frame_weights_base = np.ones(len(ep_idx_base), dtype=float)

    # Combined weights (base first, then new)
    all_weights = np.concatenate([frame_weights_base, frame_weights_new])

    # Since we can't easily concatenate LeRobotDatasets at the object level,
    # we use base_ds for training (it's larger) and add weighted new-episode
    # sampling via a separate loader. Training loop interleaves both.
    # Val: use base_ds val split (no weighting needed)
    n_base = len(base_ds)
    n_val  = max(1, int(0.15 * n_base))
    n_train_base = n_base - n_val

    from torch.utils.data import Subset, random_split
    train_base, val_base = random_split(
        base_ds, [n_train_base, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Weighted sampler for new episodes
    sampler_new = WeightedRandomSampler(
        weights=torch.FloatTensor(frame_weights_new),
        num_samples=len(frame_weights_new),
        replacement=True,
    )

    base_train_loader = DataLoader(
        train_base, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    new_train_loader = DataLoader(
        ds_new, batch_size=batch_size, sampler=sampler_new,
        num_workers=2, pin_memory=True, drop_last=False,
    ) if n_new_frames > 0 else None
    val_loader = DataLoader(
        val_base, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    return base_train_loader, new_train_loader, val_loader


def rl_train(
    base_ckpt: str,
    base_ds: LeRobotDataset,
    ds_new: LeRobotDataset,
    new_ep_scores: list[float],
    steps: int,
    batch_size: int,
    lr: float,
    out_dir: Path,
    ckpt_dir: Path,
) -> dict:
    """Fine-tune the diffusion policy using score-weighted episode sampling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'─'*60}")
    print(f"  Phase 3: RWR Fine-tuning  ({steps} steps, device={device})")
    print(f"  Base checkpoint: {base_ckpt}")
    print(f"{'─'*60}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load model from checkpoint
    model = DiffusionPolicy.from_pretrained(base_ckpt)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Loaded {n_params:,} params from {base_ckpt}")

    preprocessor, _ = make_pre_post_processors(
        make_diffusion_config(), dataset_stats=base_ds.meta.stats
    )

    base_loader, new_loader, val_loader = _build_weighted_loader(
        base_ds, ds_new, new_ep_scores, batch_size,
    )

    optimizer = AdamW(model.parameters(), lr=lr,
                      betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    def to_device(b: dict) -> dict:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in b.items()}

    history = []
    best_val = float("inf")
    global_step = 0
    new_iter = iter(new_loader) if new_loader else None

    # Determine epoch length
    steps_per_epoch = len(base_loader)
    n_epochs = max(1, steps // steps_per_epoch)

    print(f"  Epochs: {n_epochs}  steps/epoch: {steps_per_epoch}")
    print(f"  Base frames: {len(base_ds):,}  New frames: {len(ds_new):,}")
    print(f"\n{'Epoch':>6}  {'LR':>9}  {'T-loss':>9}  {'V-loss':>9}  {'Best'}")
    print("─" * 55)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.time()

        for base_batch in base_loader:
            base_batch = to_device(base_batch)
            base_batch = preprocessor(base_batch)

            # Interleave: alternate base and new batches when new data exists
            if new_iter is not None:
                try:
                    new_batch = next(new_iter)
                except StopIteration:
                    new_iter = iter(new_loader)
                    new_batch = next(new_iter)
                new_batch = to_device(new_batch)
                new_batch = preprocessor(new_batch)
                # Concatenate along batch dimension
                batch = {k: torch.cat([base_batch[k], new_batch[k]], dim=0)
                         if isinstance(base_batch[k], torch.Tensor) else base_batch[k]
                         for k in base_batch}
            else:
                batch = base_batch

            out  = model(batch)
            loss = out[0] if isinstance(out, tuple) else out
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            B = next(v.size(0) for v in batch.values() if isinstance(v, torch.Tensor))
            total_loss += loss.item() * B
            n += B

            if global_step >= steps:
                break
        if global_step >= steps:
            break

        # Validation
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                vb = to_device(vb)
                vb = preprocessor(vb)
                out = model(vb)
                vl  = out[0] if isinstance(out, tuple) else out
                B   = next(v.size(0) for v in vb.values() if isinstance(v, torch.Tensor))
                val_loss += vl.item() * B
                val_n    += B

        t_loss = total_loss / max(n, 1)
        v_loss = val_loss / max(val_n, 1)
        lr_now = scheduler.get_last_lr()[0]
        is_best = v_loss < best_val
        if is_best:
            best_val = v_loss
            model.save_pretrained(str(ckpt_dir / "best_model"))

        history.append({
            "epoch": epoch, "step": global_step, "lr": lr_now,
            "train_loss": t_loss, "val_loss": v_loss,
        })
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"{epoch:>6}  {lr_now:>9.2e}  {t_loss:>9.5f}  {v_loss:>9.5f}  "
              f"{'✓' if is_best else ''}   {time.time()-t0:.1f}s")

    model.save_pretrained(str(ckpt_dir / "final_model"))
    print(f"\n  Done. Best val loss: {best_val:.5f}")
    return {"best_val_loss": best_val, "history": history}


# ── Phase 4: Evaluate ─────────────────────────────────────────────────────────

def evaluate_checkpoint(ckpt_path: str, n_trials: int, seed: int,
                         log_dir: Path, difficulty: str = "hard") -> dict:
    """Run N trials and compute AIC scores."""
    from eval_aic_score import run_trial, gen_eval_scenarios

    print(f"\n{'─'*60}")
    print(f"  Phase 4: Evaluate {ckpt_path} ({n_trials} trials, {difficulty})")
    print(f"{'─'*60}")

    # Use hard scenarios for eval
    scenarios = gen_hard_scenarios(n_trials, seed=seed + 1000) if difficulty == "hard" \
                else gen_eval_scenarios(n_trials, seed=seed + 1000)

    results_dir = log_dir / "eval_trials"
    trial_scores = []
    for i, sc in enumerate(scenarios):
        trial_id = f"rl_eval_{i+1:02d}"
        print(f"\n  [{i+1}/{n_trials}] {trial_id}")
        score = run_trial(ckpt_path, sc, trial_id, results_dir)
        trial_scores.append(score)
        print(f"    → total={score['total']:.1f}  T3={score['tier3']}  "
              f"dur_score={score.get('duration_score','N/A')}")

    from scoring import summarize_scores
    summary = summarize_scores(trial_scores)
    print(f"\n  Results: success_rate={summary.get('success_rate',0)*100:.0f}%  "
          f"mean_total={summary.get('mean_total',0):.1f}")
    return {"summary": summary, "trial_scores": trial_scores}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="RWR fine-tuning for AIC Diffusion Policy")
    p.add_argument("--base_ckpt",    required=True,
                   help="Starting checkpoint (e.g. checkpoints_diffusion_iter10/best_model)")
    p.add_argument("--base_dataset", required=True,
                   help="Base LeRobot dataset repo_id (e.g. local/aic_cable_insertion_iter10)")
    p.add_argument("--n_episodes",   type=int, default=20,
                   help="New episodes to collect for RL")
    p.add_argument("--steps",        type=int, default=5000,
                   help="Fine-tuning gradient steps")
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--lr",           type=float, default=5e-5,
                   help="Fine-tuning LR (lower than initial training)")
    p.add_argument("--n_eval",       type=int, default=5,
                   help="Evaluation trials after fine-tuning")
    p.add_argument("--seed",         type=int, default=1337)
    p.add_argument("--output",       type=str, default="outputs_rl_finetune",
                   help="Output directory")
    p.add_argument("--skip_collect", action="store_true",
                   help="Skip collection phase (use existing episodes from --output)")
    p.add_argument("--skip_train",   action="store_true",
                   help="Skip training phase (only collect + eval)")
    args = p.parse_args()

    out_dir  = Path(__file__).parent / args.output
    ckpt_dir = Path(__file__).parent / args.output.replace("outputs", "checkpoints")
    log_dir  = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = str(Path(args.base_ckpt).resolve())
    print(f"\n{'='*65}")
    print(f"  Reward-Weighted Regression — AIC Diffusion Policy")
    print(f"  Base:     {base_ckpt}")
    print(f"  Dataset:  {args.base_dataset}")
    print(f"  Episodes: {args.n_episodes}  Steps: {args.steps}  Seed: {args.seed}")
    print(f"{'='*65}")

    # ── Phase 1: Collect ─────────────────────────────────────────────────────
    episodes_cache = out_dir / "rl_episodes.json"
    if args.skip_collect and episodes_cache.exists():
        print("\n  Skipping collection — loading cached episodes")
        with open(episodes_cache) as f:
            ep_meta = json.load(f)
        new_episodes = []
        for em in ep_meta:
            if Path(em["npz_path"]).exists():
                data = np.load(em["npz_path"])
                new_episodes.append({
                    "states":  data["states"],
                    "actions": data["actions"],
                    "center":  data["center"],
                    "left":    data["left"],
                    "right":   data["right"],
                    "score":   em["score"],
                    "raw":     em["raw"],
                })
    else:
        new_episodes = collect_rl_episodes(args.n_episodes, args.seed, log_dir)
        # Save episode metadata (not the full npz data)
        ep_meta = []
        for i, ep in enumerate(new_episodes):
            npz_path = str(out_dir / f"ep_{i:03d}.npz")
            np.savez_compressed(npz_path,
                                states=ep["states"], actions=ep["actions"],
                                center=ep["center"], left=ep["left"], right=ep["right"])
            ep_meta.append({"npz_path": npz_path, "score": ep["score"], "raw": ep["raw"]})
        with open(episodes_cache, "w") as f:
            json.dump(ep_meta, f, indent=2)

    new_ep_scores = [ep["score"] for ep in new_episodes]
    print(f"\n  New episodes: {len(new_episodes)}")
    if new_ep_scores:
        print(f"  Score range: {min(new_ep_scores):.1f} – {max(new_ep_scores):.1f}")

    # ── Phase 2 & 3: Build dataset and fine-tune ─────────────────────────────
    rl_repo_id = f"local/aic_cable_insertion_rl_seed{args.seed}"

    if not args.skip_train:
        base_ds, ds_new, _, _ = build_rl_dataset(
            new_episodes, args.base_dataset, rl_repo_id,
        )
        train_result = rl_train(
            base_ckpt=base_ckpt,
            base_ds=base_ds,
            ds_new=ds_new,
            new_ep_scores=new_ep_scores,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            out_dir=out_dir,
            ckpt_dir=ckpt_dir,
        )
    else:
        train_result = {"skipped": True}
        print("\n  Skipping training phase.")

    # ── Phase 4: Evaluate ─────────────────────────────────────────────────────
    rl_ckpt = str((ckpt_dir / "best_model").resolve())
    if (ckpt_dir / "best_model").exists():
        print(f"\n  Evaluating RL checkpoint: {rl_ckpt}")
        rl_eval = evaluate_checkpoint(rl_ckpt, args.n_eval, args.seed, log_dir)
    else:
        print(f"\n  No RL checkpoint found — evaluating base checkpoint")
        rl_eval = evaluate_checkpoint(base_ckpt, args.n_eval, args.seed, log_dir)

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        "config": vars(args),
        "new_episodes": {
            "n": len(new_episodes),
            "scores": new_ep_scores,
            "mean_score": float(np.mean(new_ep_scores)) if new_ep_scores else None,
        },
        "training": train_result if not isinstance(train_result, dict) or "skipped" not in train_result
                    else train_result,
        "evaluation": rl_eval,
    }
    (out_dir / "rl_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Saved → {out_dir / 'rl_results.json'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
