"""
collect_sim_demos.py
Collect imitation-learning demonstrations from the AIC Gazebo simulation.

For each scenario:
  1. Generate a single-trial aic_engine config with the scenario's board/cable poses
  2. Launch Zenoh router
  3. Launch aic_model with RecordCheatCode (records obs + actions to /tmp/aic_recordings/)
  4. Launch simulation with ground_truth:=true + aic_engine (exits after one insertion)
  5. Read the saved .npz episode file
  6. Add the episode to a LeRobot dataset

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
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

ROOTFS = "/opt/aic_rootfs"
ROS_SETUP = "/ws_aic/install/setup.bash"
WORKSPACE = Path("/workspace/aic")
SAVE_DIR = Path("/tmp/aic_recordings")
CONFIG_DIR = Path("/tmp/aic_configs")

IMG_H, IMG_W = 128, 144
STATE_DIM, ACTION_DIM = 26, 6
FPS = 20

REPO_ID_DEFAULT = "local/aic_cable_insertion_sim"

# Default poses matching sample_config.yaml
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


def write_trial_config(scenario: dict, path: Path) -> None:
    """Write a single-trial aic_engine config with the scenario's board and cable poses."""
    sc = scenario
    yaml = f"""# Auto-generated single-trial config for demo collection
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
    # rootfs dist-packages provide transforms3d and other ROS Python deps
    rootfs_py = f"{ROOTFS}/usr/lib/python3/dist-packages:{ROOTFS}/usr/lib/python3.12/dist-packages"
    env["PYTHONPATH"] = f"{rootfs_py}:{env.get('PYTHONPATH', '')}"
    return env


def _popen(cmd: str, log: Path | None = None, env: dict | None = None) -> subprocess.Popen:
    """Popen a bash -c command in its own process group, optionally teeing to log file."""
    if log:
        cmd = f"({cmd}) 2>&1 | tee {log}"
    return subprocess.Popen(
        ["bash", "-c", f". {ROS_SETUP} && {cmd}"],
        env=env or _ros_env(),
        start_new_session=True,   # new process group → kill whole tree with os.killpg
    )


def _kill(proc: subprocess.Popen) -> None:
    """Kill the entire process group spawned by proc."""
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


def _cleanup_stale_nodes() -> None:
    """Kill any lingering aic_model / zenoh / gazebo processes from prior runs."""
    for pattern in ["aic_model", "rmw_zenohd", "gz_server", "gzserver"]:
        subprocess.run(["pkill", "-9", "-f", pattern],
                       capture_output=True)
    time.sleep(2)


# ── episode runner ────────────────────────────────────────────────────────────

def _wait_for_episode(timeout: int = 200) -> Path | None:
    """Poll /tmp/aic_recordings/latest.txt until an episode path appears."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    latest = SAVE_DIR / "latest.txt"
    latest.unlink(missing_ok=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if latest.exists():
            text = latest.read_text().strip()
            if text:   # guard against empty file
                ep_path = Path(text)
                if ep_path.exists() and ep_path.is_file():
                    return ep_path
        time.sleep(2)
    return None


def run_episode(scenario: dict, ep_idx: int) -> Path | None:
    """
    Spin up one simulation episode:
      1. Write per-scenario aic_engine config (single trial with varied scene)
      2. Zenoh router
      3. RecordCheatCode policy
      4. Simulation + aic_engine (exits after insertion)
    Returns path to the saved .npz episode, or None on timeout/failure.
    """
    _cleanup_stale_nodes()

    env = _ros_env()
    log_dir = Path("/tmp/aic_logs")
    log_dir.mkdir(exist_ok=True)
    CONFIG_DIR.mkdir(exist_ok=True)

    # 1. Per-scenario engine config
    cfg_path = CONFIG_DIR / f"trial_{ep_idx}.yaml"
    write_trial_config(scenario, cfg_path)

    # 2. Zenoh router
    zenoh = _popen(
        "ros2 run rmw_zenoh_cpp rmw_zenohd",
        log=log_dir / f"zenoh_{ep_idx}.log",
        env=env,
    )
    time.sleep(3)

    # 3. Policy node (waits for engine to send InsertCable goal)
    policy = _popen(
        "ros2 run aic_model aic_model "
        "--ros-args -p use_sim_time:=true "
        "-p policy:=aic_example_policies.ros.RecordCheatCode",
        log=log_dir / f"policy_{ep_idx}.log",
        env=env,
    )
    time.sleep(2)

    # 4. Simulation + engine (single-trial config, ground truth mode)
    sim_cmd = (
        "ros2 launch aic_bringup aic_gz_bringup.launch.py "
        "gazebo_gui:=false "
        "ground_truth:=true "
        "spawn_task_board:=false "   # engine handles board spawning via config
        "start_aic_engine:=true "
        "shutdown_on_aic_engine_exit:=true "
        f"aic_engine_config_file:={cfg_path}"
    )
    sim = _popen(sim_cmd, log=log_dir / f"sim_{ep_idx}.log", env=env)

    # Wait for Gazebo + TF relay + engine to fully start
    time.sleep(40)
    ep_path = _wait_for_episode(timeout=200)

    # Teardown — kill entire process groups so no orphan ROS nodes remain
    for proc in [policy, sim, zenoh]:
        _kill(proc)
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

def load_existing_npzs() -> list[dict]:
    """Load all valid .npz episodes already in SAVE_DIR."""
    episodes = []
    for p in sorted(SAVE_DIR.glob("ep_*.npz")):
        try:
            d = np.load(str(p))
            if "states" in d and len(d["states"]) > 0:
                episodes.append({
                    "states":  d["states"],
                    "actions": d["actions"],
                    "center":  d["center"],
                    "left":    d["left"],
                    "right":   d["right"],
                })
                print(f"  Loaded existing: {p.name}  ({len(d['states'])} steps)")
        except Exception as e:
            print(f"  WARNING: could not load {p.name}: {e}")
    return episodes


def main():
    p = argparse.ArgumentParser(description="Collect sim demos for imitation learning")
    p.add_argument("--n_episodes",   type=int, default=20,
                   help="Total demonstration episodes (including resumed)")
    p.add_argument("--dataset_name", default=REPO_ID_DEFAULT,
                   help="LeRobot repo_id for the output dataset")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--resume",       action="store_true",
                   help="Load existing .npz files from SAVE_DIR before collecting more")
    p.add_argument("--start_idx",    type=int, default=0,
                   help="Skip the first N scenarios (use with --resume)")
    args = p.parse_args()

    scenarios = gen_scenarios(args.n_episodes, seed=args.seed)
    print(f"Collecting {args.n_episodes} sim episodes → '{args.dataset_name}'")
    print(f"Log dir: /tmp/aic_logs/\n")

    episode_data: list[dict] = []

    # Optionally load existing episodes
    if args.resume:
        episode_data = load_existing_npzs()
        print(f"Resumed with {len(episode_data)} existing episodes.\n")

    for i, sc in enumerate(scenarios):
        if i < args.start_idx:
            continue   # skip already-collected scenarios

        ep_num = i + 1
        print(f"[{ep_num}/{args.n_episodes}] board_x={sc['task_board_x']:.3f}  "
              f"board_y={sc['task_board_y']:.3f}  "
              f"cable_roll={sc['cable_roll']:.3f}")
        ep_path = run_episode(sc, i)

        if ep_path is None:
            print(f"  WARNING: episode {ep_num} timed out — skipping\n")
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
