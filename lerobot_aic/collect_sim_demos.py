"""
collect_sim_demos.py
Collect imitation-learning demonstrations from the AIC Gazebo simulation,
or build a LeRobot dataset from pre-collected .npz episode files.

Modes
─────
  collect (default):
    Run the Gazebo simulation to record episodes, then build dataset.
    Requires ROS2 + AIC rootfs environment.

  build-only (--build_only):
    Build a LeRobot dataset from existing .npz files — no simulation needed.
    Use this locally after copying recorded episodes from a Docker session.

For each simulation scenario:
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

Usage
─────
  # Collect from simulation (requires Docker/rootfs environment):
  python collect_sim_demos.py --n_episodes 20 --dataset_name local/aic_cable_insertion_sim

  # Build dataset from pre-collected .npz files (works locally):
  python collect_sim_demos.py --build_only --data_dir /path/to/npz/files

  # Build with quality filtering:
  python collect_sim_demos.py --build_only --data_dir ./recordings --min_steps 30 --max_steps 1000

  # Build with image storage instead of video (faster for small datasets):
  python collect_sim_demos.py --build_only --no_video --data_dir ./recordings
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

def build_dataset(episodes: list[dict], repo_id: str, use_videos: bool = True) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    # Determine which cameras are available from the first episode
    has_left = "left" in episodes[0] and episodes[0]["left"] is not None
    has_right = "right" in episodes[0] and episodes[0]["right"] is not None
    img_dtype = "video" if use_videos else "image"

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
            "dtype": img_dtype, "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32", "shape": (ACTION_DIM,),
            "names": ["linear.x", "linear.y", "linear.z",
                      "angular.x", "angular.y", "angular.z"],
        },
    }
    if has_left:
        features["observation.images.left_camera"] = {
            "dtype": img_dtype, "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        }
    if has_right:
        features["observation.images.right_camera"] = {
            "dtype": img_dtype, "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        }

    root = HF_LEROBOT_HOME / repo_id
    if root.exists():
        print(f"Removing existing dataset at {root}")
        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=FPS, features=features,
        robot_type="ur5e_aic", use_videos=use_videos,
    )

    total_frames = 0
    for i, ep in enumerate(episodes):
        T = len(ep["states"])
        for t in range(T):
            frame = {
                "observation.state":                  ep["states"][t],
                "observation.images.center_camera":   ep["center"][t],
                "action":                             ep["actions"][t],
                "task":                               "cable_insertion_sim",
            }
            if has_left:
                frame["observation.images.left_camera"] = ep["left"][t]
            if has_right:
                frame["observation.images.right_camera"] = ep["right"][t]
            ds.add_frame(frame)
        ds.save_episode()
        total_frames += T
        print(f"  episode {i + 1}/{len(episodes)}: {T} steps")

    cameras = ["center"]
    if has_left:
        cameras.append("left")
    if has_right:
        cameras.append("right")
    print(f"\nDataset: {root}")
    print(f"  Episodes : {len(episodes)}")
    print(f"  Frames   : {total_frames:,}")
    print(f"  Cameras  : {', '.join(cameras)}")
    print(f"  Storage  : {'video (mp4)' if use_videos else 'images (png)'}")
    print(f"  State dim: {STATE_DIM}  |  Action dim: {ACTION_DIM}  |  FPS: {FPS}")


# ── entry point ───────────────────────────────────────────────────────────────

def load_existing_npzs(data_dir: Path, min_steps: int = 1, max_steps: int = 0) -> list[dict]:
    """Load all valid .npz episodes from a directory.

    Args:
        data_dir: Directory containing ep_*.npz files.
        min_steps: Skip episodes shorter than this (default: 1 = keep all).
        max_steps: Truncate episodes longer than this (0 = no limit).

    Returns:
        List of episode dicts with keys: states, actions, center, [left, right].
    """
    episodes = []
    npz_files = sorted(data_dir.glob("ep_*.npz"))
    if not npz_files:
        # Also try *.npz as fallback pattern
        npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        print(f"  WARNING: no .npz files found in {data_dir}")
        return episodes

    skipped = 0
    for p in npz_files:
        try:
            d = np.load(str(p), allow_pickle=False)
            if "states" not in d or len(d["states"]) == 0:
                print(f"  WARNING: {p.name} has no states — skipping")
                skipped += 1
                continue
            T = len(d["states"])
            if T < min_steps:
                print(f"  WARNING: {p.name} too short ({T} < {min_steps} steps) — skipping")
                skipped += 1
                continue

            end = min(T, max_steps) if max_steps > 0 else T
            ep = {
                "states":  d["states"][:end],
                "actions": d["actions"][:end],
                "center":  d["center"][:end],
            }
            # Left/right cameras are optional (single-camera setups)
            if "left" in d:
                ep["left"] = d["left"][:end]
            else:
                ep["left"] = None
            if "right" in d:
                ep["right"] = d["right"][:end]
            else:
                ep["right"] = None

            episodes.append(ep)
            n_steps = len(ep["states"])
            cams = 1 + (1 if ep["left"] is not None else 0) + (1 if ep["right"] is not None else 0)
            print(f"  Loaded: {p.name}  ({n_steps} steps, {cams} cameras)")
        except Exception as e:
            print(f"  WARNING: could not load {p.name}: {e}")
            skipped += 1

    if skipped:
        print(f"  Skipped {skipped} files.")
    return episodes


def print_episode_stats(episodes: list[dict]) -> None:
    """Print summary statistics for loaded episodes."""
    if not episodes:
        print("No episodes to summarize.")
        return
    lengths = [len(ep["states"]) for ep in episodes]
    total_frames = sum(lengths)
    has_left = any(ep.get("left") is not None for ep in episodes)
    has_right = any(ep.get("right") is not None for ep in episodes)
    n_cameras = 1 + int(has_left) + int(has_right)

    print(f"\n{'─' * 50}")
    print(f"Episode Summary")
    print(f"{'─' * 50}")
    print(f"  Episodes     : {len(episodes)}")
    print(f"  Total frames : {total_frames:,}")
    print(f"  Steps/ep     : min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
    print(f"  Cameras      : {n_cameras}")
    print(f"  State dim    : {episodes[0]['states'].shape[1] if len(episodes[0]['states'].shape) > 1 else 'N/A'}")
    print(f"  Action dim   : {episodes[0]['actions'].shape[1] if len(episodes[0]['actions'].shape) > 1 else 'N/A'}")
    if len(episodes[0]['center'].shape) >= 3:
        h, w = episodes[0]['center'].shape[1], episodes[0]['center'].shape[2]
        print(f"  Image size   : {h}×{w}")
    print(f"{'─' * 50}\n")


def main():
    p = argparse.ArgumentParser(
        description="Collect sim demos or build LeRobot datasets for diffusion policy training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build dataset from existing .npz recordings:
  python collect_sim_demos.py --build_only --data_dir /tmp/aic_recordings

  # Build with quality filtering and image-only storage:
  python collect_sim_demos.py --build_only --data_dir ./recordings --min_steps 30 --no_video

  # Collect from simulation (requires Docker/rootfs env):
  python collect_sim_demos.py --n_episodes 20

  # Resume collection and add more episodes:
  python collect_sim_demos.py --n_episodes 40 --resume --start_idx 20
""",
    )
    # ── mode ──
    p.add_argument("--build_only",   action="store_true",
                   help="Skip simulation; build dataset from existing .npz files only")
    # ── data source ──
    p.add_argument("--data_dir",     type=str, default=None,
                   help="Directory with .npz episode files (default: /tmp/aic_recordings)")
    p.add_argument("--n_episodes",   type=int, default=20,
                   help="Total demonstration episodes to collect (ignored with --build_only)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--resume",       action="store_true",
                   help="Load existing .npz files before collecting more")
    p.add_argument("--start_idx",    type=int, default=0,
                   help="Skip the first N scenarios (use with --resume)")
    # ── dataset output ──
    p.add_argument("--dataset_name", default=REPO_ID_DEFAULT,
                   help="LeRobot repo_id for the output dataset")
    p.add_argument("--no_video",     action="store_true",
                   help="Store images as PNG instead of mp4 video (faster for small datasets)")
    # ── filtering ──
    p.add_argument("--min_steps",    type=int, default=1,
                   help="Skip episodes shorter than this many steps")
    p.add_argument("--max_steps",    type=int, default=0,
                   help="Truncate episodes longer than this (0 = no limit)")

    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else SAVE_DIR
    use_videos = not args.no_video

    # ── Build-only mode ──────────────────────────────────────────────────────
    if args.build_only:
        print(f"Build-only mode: loading .npz episodes from {data_dir}")
        if not data_dir.exists():
            print(f"ERROR: data directory does not exist: {data_dir}")
            return

        episode_data = load_existing_npzs(
            data_dir, min_steps=args.min_steps, max_steps=args.max_steps
        )
        if not episode_data:
            print("ERROR: no valid episodes found.")
            return

        print_episode_stats(episode_data)
        print(f"Building LeRobot dataset ({len(episode_data)} episodes) → '{args.dataset_name}' …")
        build_dataset(episode_data, args.dataset_name, use_videos=use_videos)
        print("\nDataset build complete.")
        print(f"\nTo train diffusion policy on this dataset:")
        print(f"  python train_diffusion.py --repo_id {args.dataset_name}")
        return

    # ── Collection mode (requires simulation) ────────────────────────────────
    scenarios = gen_scenarios(args.n_episodes, seed=args.seed)
    print(f"Collecting {args.n_episodes} sim episodes → '{args.dataset_name}'")
    print(f"Log dir: /tmp/aic_logs/\n")

    episode_data: list[dict] = []

    # Optionally load existing episodes
    if args.resume:
        episode_data = load_existing_npzs(
            data_dir, min_steps=args.min_steps, max_steps=args.max_steps
        )
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

        data = np.load(str(ep_path), allow_pickle=False)
        T = len(data["states"])

        if T < args.min_steps:
            print(f"  WARNING: episode {ep_num} too short ({T} < {args.min_steps}) — skipping\n")
            continue

        end = min(T, args.max_steps) if args.max_steps > 0 else T
        ep = {
            "states":  data["states"][:end],
            "actions": data["actions"][:end],
            "center":  data["center"][:end],
            "left":    data["left"][:end] if "left" in data else None,
            "right":   data["right"][:end] if "right" in data else None,
        }
        episode_data.append(ep)
        print(f"  OK: {end} steps  →  {ep_path.name}\n")

    if not episode_data:
        print("ERROR: no episodes collected.")
        return

    print_episode_stats(episode_data)
    print(f"Building LeRobot dataset ({len(episode_data)} episodes) …")
    build_dataset(episode_data, args.dataset_name, use_videos=use_videos)
    print("\nCollection complete.")
    print(f"\nTo train diffusion policy on this dataset:")
    print(f"  python train_diffusion.py --repo_id {args.dataset_name}")


if __name__ == "__main__":
    main()
