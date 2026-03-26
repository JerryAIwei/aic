"""
eval_success_rate.py
Run N randomised AIC trials for two policy checkpoints and compare success rates.

Each trial:
  - Picks a random scene (board/cable pose variation)
  - Launches: Zenoh → policy(ckpt) → sim+engine
  - Reads insertion_event from scoring log to determine success
  - Records success / time / max_force

Usage:
    python eval_success_rate.py \
        --n_trials 5 \
        --ckpt_20ep checkpoints_diffusion_aic_cable_insertion_sim/best_model \
        --ckpt_40ep checkpoints_diffusion_aic_cable_insertion_sim_40/best_model \
        --output    outputs_diffusion_aic_cable_insertion_sim_40/success_rate.json
"""

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

ROOTFS   = "/opt/aic_rootfs"
ROS_SETUP = "/ws_aic/install/setup.bash"
WORKSPACE = Path("/workspace/aic")
CONFIG_DIR = Path("/tmp/aic_eval_configs")

# Default board/cable poses (matching collect_sim_demos.py)
_CABLE = dict(roll=0.4432, pitch=-0.4838, yaw=1.3303)
_BOARD = dict(x=0.15, y=-0.2, z=1.14, yaw=3.1415)


# ── Environment ───────────────────────────────────────────────────────────────

def _ros_env(extra: dict | None = None) -> dict:
    env = os.environ.copy()
    rootfs_libs = (
        f"{ROOTFS}/usr/lib/x86_64-linux-gnu:"
        f"{ROOTFS}/lib/x86_64-linux-gnu:"
        f"{ROOTFS}/usr/lib"
    )
    ogre      = f"{ROOTFS}/usr/lib/x86_64-linux-gnu/OGRE-2.3"
    gz_vendor = f"{ROOTFS}/opt/ros/kilted/opt/gz_ogre_next_vendor/lib"
    sys_libs  = "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
    existing  = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"]           = f"{sys_libs}:{rootfs_libs}:{ogre}:{gz_vendor}:{existing}"
    env["RMW_IMPLEMENTATION"]         = "rmw_zenoh_cpp"
    env["ZENOH_ROUTER_CHECK_ATTEMPTS"] = "-1"
    env["ZENOH_CONFIG_OVERRIDE"]      = "transport/shared_memory/enabled=false"
    env["OGRE2_RESOURCE_PATH"]        = "/usr/lib/x86_64-linux-gnu/OGRE-2.3/OGRE"
    env.pop("DISPLAY", None)
    rootfs_py = (
        f"{ROOTFS}/usr/lib/python3/dist-packages:"
        f"{ROOTFS}/usr/lib/python3.12/dist-packages"
    )
    env["PYTHONPATH"] = f"{rootfs_py}:{env.get('PYTHONPATH', '')}"
    if extra:
        env.update(extra)
    return env


def _popen(cmd: str, log: Path | None = None, env: dict | None = None) -> subprocess.Popen:
    if log:
        cmd = f"({cmd}) 2>&1 | tee {log}"
    return subprocess.Popen(
        ["bash", "-c", f". {ROS_SETUP} && {cmd}"],
        env=env or _ros_env(),
        start_new_session=True,
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
    for pat in ["aic_model", "rmw_zenohd", "gz_server", "gzserver"]:
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    time.sleep(2)


# ── Scene config ──────────────────────────────────────────────────────────────

def _write_eval_config(scenario: dict, path: Path) -> None:
    sc = scenario
    yaml = f"""# Auto-generated eval config
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
        time_limit: 120

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


# ── Trial result parser ────────────────────────────────────────────────────────

def _parse_trial(log_dir: Path) -> dict:
    result = {"success": False, "time_s": None, "max_force_N": None}

    for log in log_dir.glob("*.log"):
        try:
            text = log.read_text(errors="ignore")
        except Exception:
            continue

        # Insertion success — look for insertion event or success message
        if (re.search(r"insertion.?success", text, re.IGNORECASE)
                or re.search(r"cable.*inserted", text, re.IGNORECASE)
                or "succeeded" in text.lower()):
            result["success"] = True

        # Time to insertion (seconds after trial start)
        m = re.search(r"elapsed[_\s]seconds?[:\s=]+([0-9.]+)", text, re.IGNORECASE)
        if m:
            result["time_s"] = float(m.group(1))

        # Max force
        m = re.search(r"max[_\s]force[:\s=]+([0-9.]+)", text, re.IGNORECASE)
        if m:
            result["max_force_N"] = float(m.group(1))

    # Fallback: look in JSON result files
    for jf in log_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text())
            if "success" in data:
                result["success"] = bool(data["success"])
            if "elapsed_seconds" in data and result["time_s"] is None:
                result["time_s"] = float(data["elapsed_seconds"])
            if "max_force" in data and result["max_force_N"] is None:
                result["max_force_N"] = float(data["max_force"])
        except Exception:
            pass

    return result


# ── One trial runner ──────────────────────────────────────────────────────────

def run_trial(ckpt_path: str, scenario: dict, trial_id: str,
              results_dir: Path, timeout: int = 180) -> dict:
    _cleanup()
    CONFIG_DIR.mkdir(exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / trial_id
    log_dir.mkdir(exist_ok=True)

    cfg_path = CONFIG_DIR / f"{trial_id}.yaml"
    _write_eval_config(scenario, cfg_path)

    env = _ros_env({"AIC_DIFFUSION_CKPT": ckpt_path})

    zenoh = _popen("ros2 run rmw_zenoh_cpp rmw_zenohd",
                   log=log_dir / "zenoh.log", env=env)
    time.sleep(3)

    policy = _popen(
        "ros2 run aic_model aic_model "
        "--ros-args -p use_sim_time:=true "
        "-p policy:=aic_example_policies.ros.RunSimDiffusion.RunSimDiffusion",
        log=log_dir / "policy.log", env=env,
    )
    time.sleep(2)

    sim = _popen(
        "ros2 launch aic_bringup aic_gz_bringup.launch.py "
        "gazebo_gui:=false "
        "ground_truth:=false "
        "spawn_task_board:=false "
        "start_aic_engine:=true "
        "shutdown_on_aic_engine_exit:=true "
        f"aic_engine_config_file:={cfg_path}",
        log=log_dir / "sim.log", env=env,
    )

    start = time.time()
    try:
        sim.wait(timeout=timeout + 40)
    except subprocess.TimeoutExpired:
        print(f"    Trial {trial_id}: sim timeout after {timeout + 40}s")

    elapsed = time.time() - start
    for proc in [policy, sim, zenoh]:
        _kill(proc)
    time.sleep(3)

    result = _parse_trial(log_dir)
    result["elapsed_total_s"] = round(elapsed, 1)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def gen_eval_scenarios(n: int, seed: int = 999) -> list[dict]:
    """Fresh random scenarios for evaluation (different from training seeds)."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        out.append({
            "task_board_x":   round(_BOARD["x"]   + rng.uniform(-0.015, 0.015), 4),
            "task_board_y":   round(_BOARD["y"]   + rng.uniform(-0.015, 0.015), 4),
            "task_board_z":   _BOARD["z"],
            "task_board_yaw": round(_BOARD["yaw"] + rng.uniform(-0.040, 0.040), 4),
            "cable_roll":     round(_CABLE["roll"]  + rng.uniform(-0.025, 0.025), 4),
            "cable_pitch":    round(_CABLE["pitch"] + rng.uniform(-0.025, 0.025), 4),
            "cable_yaw":      round(_CABLE["yaw"]   + rng.uniform(-0.025, 0.025), 4),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials",  type=int, default=5)
    p.add_argument("--ckpt_20ep", required=True)
    p.add_argument("--ckpt_40ep", required=True)
    p.add_argument("--output",    required=True)
    p.add_argument("--seed",      type=int, default=999)
    args = p.parse_args()

    scenarios = gen_eval_scenarios(args.n_trials, seed=args.seed)
    results = {"model_20ep": [], "model_40ep": []}

    print(f"\n{'='*60}")
    print(f"  Success-Rate Evaluation ({args.n_trials} trials per model)")
    print(f"{'='*60}")

    for model_key, ckpt in [("model_20ep", args.ckpt_20ep),
                             ("model_40ep", args.ckpt_40ep)]:
        print(f"\n  Model: {model_key}  checkpoint: {ckpt}")
        for i, sc in enumerate(scenarios):
            trial_id = f"{model_key}_trial{i+1}"
            results_dir = Path(args.output).parent / "eval_trials"
            print(f"    [{i+1}/{args.n_trials}] board=({sc['task_board_x']:.3f},{sc['task_board_y']:.3f}) "
                  f"cable_roll={sc['cable_roll']:.3f}")
            r = run_trial(ckpt, sc, trial_id, results_dir)
            results[model_key].append(r)
            status = "SUCCESS" if r["success"] else "FAILURE"
            print(f"      → {status}  ({r['elapsed_total_s']:.0f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"  {'Metric':<35}  {'20-ep model':>12}  {'40-ep model':>12}")
    print(f"  {'-'*61}")

    for key in ["model_20ep", "model_40ep"]:
        r = results[key]
        results[key + "_summary"] = {
            "success_rate": sum(x["success"] for x in r) / len(r),
            "n_success":    sum(x["success"] for x in r),
            "n_trials":     len(r),
            "mean_time_s":  np.mean([x["time_s"] for x in r if x["time_s"]]) if any(x["time_s"] for x in r) else None,
        }

    m20 = results["model_20ep_summary"]
    m40 = results["model_40ep_summary"]

    def pct(v): return f"{v*100:.0f}%"

    rows = [
        ("Success rate", pct(m20["success_rate"]), pct(m40["success_rate"])),
        ("Successes / Trials",
         f"{m20['n_success']}/{m20['n_trials']}",
         f"{m40['n_success']}/{m40['n_trials']}"),
        ("Mean insertion time (s)",
         f"{m20['mean_time_s']:.1f}" if m20["mean_time_s"] else "N/A",
         f"{m40['mean_time_s']:.1f}" if m40["mean_time_s"] else "N/A"),
    ]
    for label, v20, v40 in rows:
        print(f"  {label:<35}  {v20:>12}  {v40:>12}")

    delta = m40["success_rate"] - m20["success_rate"]
    print(f"\n  Δ success rate (40ep − 20ep): {delta*100:+.0f}%")
    print(f"{'='*60}")

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model_20ep_checkpoint": args.ckpt_20ep,
        "model_40ep_checkpoint": args.ckpt_40ep,
        "n_trials": args.n_trials,
        "seed": args.seed,
        "results": results,
    }, indent=2))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
