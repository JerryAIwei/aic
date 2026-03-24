"""
collect_sim_demos.py
Collect imitation-learning demonstrations from the AIC Gazebo simulation.

For each scenario:
  1. Launch Zenoh router
  2. Launch aic_model with RecordCheatCode (records obs + actions to /tmp/aic_recordings/)
  3. Launch simulation with ground_truth:=true + aic_engine (exits after one insertion)
  4. Read the saved .npz episode file
  5. Add the episode to a LeRobot dataset

Diversity is achieved by perturbing:
  - task_board_x/y     ± 2 cm
  - task_board_yaw     ± 0.05 rad
  - cable_roll/pitch/yaw  ± 0.03 rad

Usage:
    python collect_sim_demos.py [--n_episodes 20] [--dataset_name local/aic_cable_insertion_sim]
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

ROOTFS = "/opt/aic_rootfs"
ROS_SETUP = "/ws_aic/install/setup.bash"
WORKSPACE = Path("/workspace/aic")
SAVE_DIR = Path("/tmp/aic_recordings")

IMG_H, IMG_W = 128, 144
STATE_DIM, ACTION_DIM = 26, 6
FPS = 20

REPO_ID_DEFAULT = "local/aic_cable_insertion_sim"

# Default poses from sample_config.yaml / launch defaults
_CABLE = dict(roll=0.4432, pitch=-0.4838, yaw=1.3303)
_BOARD = dict(x=0.15, y=-0.2, z=1.14, yaw=3.1415)


# ── scenario generation ───────────────────────────────────────────────────────

def gen_scenarios(n: int, seed: int = 42) -> list[dict]:
    """Return n dicts with varied cable/board launch parameters."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        out.append({
            "task_board_x":   round(_BOARD["x"]   + rng.uniform(-0.020, 0.020), 4),
            "task_board_y":   round(_BOARD["y"]   + rng.uniform(-0.020, 0.020), 4),
            "task_board_z":   _BOARD["z"],
            "task_board_yaw": round(_BOARD["yaw"] + rng.uniform(-0.050, 0.050), 4),
            "cable_roll":     round(_CABLE["roll"]  + rng.uniform(-0.030, 0.030), 4),
            "cable_pitch":    round(_CABLE["pitch"] + rng.uniform(-0.030, 0.030), 4),
            "cable_yaw":      round(_CABLE["yaw"]   + rng.uniform(-0.030, 0.030), 4),
        })
    return out


# ── environment helpers ───────────────────────────────────────────────────────

def _ros_env() -> dict:
    """Build env dict that mirrors start_simulator.sh gpu-mode setup."""
    env = os.environ.copy()
    rootfs_libs = (
        f"{ROOTFS}/usr/lib/x86_64-linux-gnu:"
        f"{ROOTFS}/lib/x86_64-linux-gnu:"
        f"{ROOTFS}/usr/lib"
    )
    ogre = f"{ROOTFS}/usr/lib/x86_64-linux-gnu/OGRE-2.3"
    gz_vendor = f"{ROOTFS}/opt/ros/kilted/opt/gz_ogre_next_vendor/lib"
    sys_libs = "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{sys_libs}:{rootfs_libs}:{ogre}:{gz_vendor}:{existing}"
    env["RMW_IMPLEMENTATION"] = "rmw_zenoh_cpp"
    env["ZENOH_ROUTER_CHECK_ATTEMPTS"] = "-1"
    env["ZENOH_CONFIG_OVERRIDE"] = "transport/shared_memory/enabled=false"
    env["OGRE2_RESOURCE_PATH"] = "/usr/lib/x86_64-linux-gnu/OGRE-2.3/OGRE"
    env.pop("DISPLAY", None)   # headless
    # Make workspace's aic_example_policies importable (for RecordCheatCode)
    ws_policies = str(WORKSPACE / "aic_example_policies")
    env["PYTHONPATH"] = f"{ws_policies}:{env.get('PYTHONPATH', '')}"
    return env


def _popen(cmd: str, log: Path | None = None, env: dict | None = None) -> subprocess.Popen:
    """Popen a bash -c command, optionally teeing to log file."""
    if log:
        cmd = f"({cmd}) 2>&1 | tee {log}"
    return subprocess.Popen(
        ["bash", "-c", f". {ROS_SETUP} && {cmd}"],
        env=env or _ros_env(),
    )


# ── episode runner ────────────────────────────────────────────────────────────

def _wait_for_episode(timeout: int = 200) -> Path | None:
    """Poll /tmp/aic_recordings/latest.txt until an episode path appears."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    latest = SAVE_DIR / "latest.txt"
    latest.unlink(missing_ok=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if latest.exists():
            ep_path = Path(latest.read_text().strip())
            if ep_path.exists():
                return ep_path
        time.sleep(2)
    return None


def run_episode(scenario: dict, ep_idx: int) -> Path | None:
    """
    Spin up one simulation episode:
      1. Zenoh router
      2. RecordCheatCode policy
      3. Simulation + aic_engine (exits after insertion)
    Returns path to the saved .npz episode, or None on timeout/failure.
    """
    env = _ros_env()
    log_dir = Path("/tmp/aic_logs")
    log_dir.mkdir(exist_ok=True)

    # 1. Zenoh router
    zenoh = _popen(
        "ros2 run rmw_zenoh_cpp rmw_zenohd",
        log=log_dir / f"zenoh_{ep_idx}.log",
        env=env,
    )
    time.sleep(3)

    # 2. Policy node (waits for engine to send InsertCable goal)
    policy = _popen(
        "ros2 run aic_model aic_model "
        "--ros-args -p use_sim_time:=true "
        "-p policy:=aic_example_policies.ros.RecordCheatCode.RecordCheatCode",
        log=log_dir / f"policy_{ep_idx}.log",
        env=env,
    )
    time.sleep(2)

    # 3. Simulation + engine
    sc = scenario
    sim_cmd = (
        "ros2 launch aic_bringup aic_gz_bringup.launch.py "
        "gazebo_gui:=false "
        "ground_truth:=true "
        "spawn_task_board:=true "
        "start_aic_engine:=true "
        "shutdown_on_aic_engine_exit:=true "
        "attach_cable_to_gripper:=true "
        f"task_board_x:={sc['task_board_x']} "
        f"task_board_y:={sc['task_board_y']} "
        f"task_board_z:={sc['task_board_z']} "
        f"task_board_yaw:={sc['task_board_yaw']} "
        f"cable_roll:={sc['cable_roll']} "
        f"cable_pitch:={sc['cable_pitch']} "
        f"cable_yaw:={sc['cable_yaw']}"
    )
    sim = _popen(sim_cmd, log=log_dir / f"sim_{ep_idx}.log", env=env)

    # Wait for sim startup then poll for episode completion
    time.sleep(20)
    ep_path = _wait_for_episode(timeout=180)

    # Teardown
    for proc in [policy, sim, zenoh]:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(3)

    return ep_path


# ── LeRobot dataset builder ───────────────────────────────────────────────────

def build_dataset(episodes: list[dict], repo_id: str) -> None:
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

    root = HF_LEROBOT_HOME / repo_id
    if root.exists():
        print(f"Removing existing dataset at {root}")
        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=FPS, features=features,
        robot_type="ur5e_aic", use_videos=True,
    )

    for i, ep in enumerate(episodes):
        T = len(ep["states"])
        for t in range(T):
            ds.add_frame({
                "observation.state":                  ep["states"][t],
                "observation.images.center_camera":   ep["center"][t],
                "observation.images.left_camera":     ep["left"][t],
                "observation.images.right_camera":    ep["right"][t],
                "action":                             ep["actions"][t],
                "task":                               "cable_insertion_sim",
            })
        ds.save_episode()
        print(f"  episode {i + 1}/{len(episodes)}: {T} steps")

    print(f"\nDataset: {root}  ({len(ds):,} frames total)")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Collect sim demos for imitation learning")
    p.add_argument("--n_episodes",   type=int, default=20,
                   help="Number of demonstration episodes to collect")
    p.add_argument("--dataset_name", default=REPO_ID_DEFAULT,
                   help="LeRobot repo_id for the output dataset")
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    scenarios = gen_scenarios(args.n_episodes, seed=args.seed)
    print(f"Collecting {args.n_episodes} sim episodes → '{args.dataset_name}'")
    print(f"Log dir: /tmp/aic_logs/\n")

    episode_data: list[dict] = []

    for i, sc in enumerate(scenarios):
        print(f"[{i + 1}/{args.n_episodes}] board_x={sc['task_board_x']:.3f}  "
              f"board_y={sc['task_board_y']:.3f}  "
              f"cable_roll={sc['cable_roll']:.3f}")
        ep_path = run_episode(sc, i)

        if ep_path is None:
            print(f"  WARNING: episode {i + 1} timed out — skipping\n")
            continue

        data = np.load(str(ep_path))
        T = len(data["states"])
        episode_data.append({
            "states":  data["states"],
            "actions": data["actions"],
            "center":  data["center"],
            "left":    data["left"],
            "right":   data["right"],
        })
        print(f"  OK: {T} steps  →  {ep_path.name}\n")

    if not episode_data:
        print("ERROR: no episodes collected.")
        return

    print(f"\nBuilding LeRobot dataset ({len(episode_data)} episodes) …")
    build_dataset(episode_data, args.dataset_name)
    print("Collection complete.")


if __name__ == "__main__":
    main()
