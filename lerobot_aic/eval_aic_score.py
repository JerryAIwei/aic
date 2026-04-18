"""
eval_aic_score.py — Full AIC three-tier scoring evaluation for a single checkpoint.

Runs N randomised trials, parses available metrics from engine logs, and
computes the AIC score (scoring.py) for each trial.

Usage:
    python eval_aic_score.py \
        --checkpoint lerobot_aic/checkpoints_diffusion_iter10/best_model \
        --n_trials 10 \
        --output    lerobot_aic/outputs_diffusion_iter10/aic_score.json

Metrics available from logs (no bag required):
    tier1           always 1 when model activates successfully
    tier3           correct/none derived from engine "total score" log
    duration_score  from elapsed_seconds pattern
    force_penalty   estimated from max_force_N (conservative: penalise if > 20 N)
    contact_penalty from off-limit contact pattern in engine log
    smoothness      N/A without EE trajectory bag
    efficiency      N/A without EE trajectory bag
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

from scoring import compute_trial_score, summarize_scores

ROOTFS    = "/opt/aic_rootfs"
ROS_SETUP = "/ws_aic/install/setup.bash"
WORKSPACE = Path("/workspace/aic")
CONFIG_DIR = Path("/tmp/aic_eval_configs")

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
    env["LD_LIBRARY_PATH"]            = f"{sys_libs}:{rootfs_libs}:{ogre}:{gz_vendor}:{existing}"
    env["RMW_IMPLEMENTATION"]          = "rmw_zenoh_cpp"
    env["ZENOH_ROUTER_CHECK_ATTEMPTS"] = "-1"
    env["ZENOH_CONFIG_OVERRIDE"]       = "transport/shared_memory/enabled=false"
    env["OGRE2_RESOURCE_PATH"]         = "/usr/lib/x86_64-linux-gnu/OGRE-2.3/OGRE"
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
    for pat in ["aic_model", "rmw_zenohd", "gz_server", "gzserver",
                "component_container", "aic_engine", "aic_adapter",
                "ros2 launch aic_bringup"]:
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    time.sleep(4)
    aic_results = Path(os.environ.get("AIC_RESULTS_DIR", Path.home() / "aic_results"))
    if aic_results.exists():
        for bag in aic_results.glob("bag_trial_*"):
            try:
                import shutil
                shutil.rmtree(bag)
            except Exception:
                pass


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


# ── Log parser ────────────────────────────────────────────────────────────────

def _parse_trial_logs(log_dir: Path) -> dict:
    """Extract available metrics from trial logs to populate a scoring.py trial dict."""
    raw = {"valid": False, "insertion_result": "none",
           "duration_s": None, "max_force_N": None,
           "has_off_limit_contact": False}

    # Strip ANSI escape codes helper
    _ansi = re.compile(r"\x1b\[[0-9;]*m")

    for log in log_dir.glob("*.log"):
        try:
            text = log.read_text(errors="ignore")
        except Exception:
            continue

        clean = _ansi.sub("", text)

        # Model activated → Tier 1
        if re.search(r"activat|lifecycle.*active|policy.*active", clean, re.IGNORECASE):
            raw["valid"] = True

        # ── Engine structured tier score output ──────────────────────────────
        # The engine logs:
        #   tier_1:
        #     score: 1
        #   tier_2:
        #     score: 0
        #   tier_3:
        #     score: 0 (or 75 for correct insertion)
        # Parse tier_3 score to determine insertion result.
        tier3_m = re.search(
            r"tier_?3\b[^\n]*\n(?:[^\n]*\n)*?[^\n]*score:\s*([0-9.-]+)",
            clean, re.IGNORECASE | re.MULTILINE,
        )
        if tier3_m:
            t3_val = float(tier3_m.group(1))
            raw["engine_tier3_score"] = t3_val
            if t3_val >= 75:
                raw["valid"] = True
                raw["insertion_result"] = "correct"
            elif t3_val == -12:
                raw["insertion_result"] = "wrong"
            elif t3_val >= 38:
                raw["insertion_result"] = "partial"
            elif t3_val > 0:
                raw["insertion_result"] = "proximity"

        tier2_m = re.search(
            r"tier_?2\b[^\n]*\n(?:[^\n]*\n)*?[^\n]*score:\s*([0-9.-]+)",
            clean, re.IGNORECASE | re.MULTILINE,
        )
        if tier2_m:
            raw["engine_tier2_score"] = float(tier2_m.group(1))

        tier1_m = re.search(
            r"tier_?1\b[^\n]*\n(?:[^\n]*\n)*?[^\n]*score:\s*([0-9.-]+)",
            clean, re.IGNORECASE | re.MULTILINE,
        )
        if tier1_m:
            t1_val = float(tier1_m.group(1))
            raw["engine_tier1_score"] = t1_val
            if t1_val >= 1:
                raw["valid"] = True

        # Insertion event topic — only match actual event messages, not config parsing.
        # The engine logs "insertion_event" as a topic name in config output;
        # only trigger on explicit success messages to avoid false positives.
        if re.search(r"cable successfully inserted|insertion complete|insert.*success.*event",
                     clean, re.IGNORECASE):
            raw["valid"] = True
            if raw["insertion_result"] == "none":
                raw["insertion_result"] = "correct"

        # Elapsed time
        m = re.search(r"elapsed[_\s]seconds?[:\s=]+([0-9.]+)", clean, re.IGNORECASE)
        if m and raw["duration_s"] is None:
            raw["duration_s"] = float(m.group(1))

        # Max force
        m = re.search(r"max[_\s]force[:\s=]+([0-9.]+)", clean, re.IGNORECASE)
        if m and raw["max_force_N"] is None:
            raw["max_force_N"] = float(m.group(1))

        # Off-limit contact
        if re.search(r"off.limit contact|off_limit.*detect|contact.*penalty",
                     clean, re.IGNORECASE):
            raw["has_off_limit_contact"] = True

    # JSON fallback
    for jf in log_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text())
            if "success" in data and not raw["valid"]:
                raw["valid"] = bool(data["success"])
                if raw["valid"]:
                    raw["insertion_result"] = "correct"
            if "elapsed_seconds" in data and raw["duration_s"] is None:
                raw["duration_s"] = float(data["elapsed_seconds"])
            if "max_force" in data and raw["max_force_N"] is None:
                raw["max_force_N"] = float(data["max_force"])
        except Exception:
            pass

    return raw


def _build_trial_dict(raw: dict) -> dict:
    """Convert raw log-parsed data into a scoring.py trial dict."""
    trial: dict = {
        "valid":                  raw["valid"],
        "insertion_result":       raw["insertion_result"],
        "duration_s":             raw["duration_s"],
        "has_off_limit_contact":  raw["has_off_limit_contact"],
        # Trajectory data unavailable without bag parsing
        "ee_positions":           None,
        "timestamps":             None,
        "initial_plug_port_dist": None,
        # Engine scores if available (bypass scoring.py computation)
        "_engine_tier1":          raw.get("engine_tier1_score"),
        "_engine_tier2":          raw.get("engine_tier2_score"),
        "_engine_tier3":          raw.get("engine_tier3_score"),
    }

    # Approximate force penalty from max_force:
    # If max_force > 20 N we can't know duration, but conservatively flag it.
    # Build a minimal 2-point "force signal" spanning 2 s if max > threshold.
    if raw["max_force_N"] is not None and raw["max_force_N"] > 20.0:
        trial["force_magnitudes"]  = np.array([raw["max_force_N"], raw["max_force_N"]])
        trial["force_timestamps"]  = np.array([0.0, 2.0])   # assume sustained
    else:
        trial["force_magnitudes"]  = np.array([raw["max_force_N"] or 0.0,
                                               raw["max_force_N"] or 0.0])
        trial["force_timestamps"]  = np.array([0.0, 1.0])

    return trial


# ── One trial runner ──────────────────────────────────────────────────────────

def run_trial(ckpt_path: str, scenario: dict, trial_id: str,
              results_dir: Path, timeout: int = 360) -> dict:
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
    time.sleep(15)

    policy = _popen(
        "ros2 run aic_model aic_model "
        "--ros-args -p use_sim_time:=true "
        "-p policy:=aic_example_policies.ros.RunSimDiffusion",
        log=log_dir / "policy.log", env=env,
    )

    start = time.time()
    try:
        sim.wait(timeout=timeout + 40)
    except subprocess.TimeoutExpired:
        print(f"    Trial {trial_id}: sim timeout after {timeout + 40}s")

    elapsed_wall = time.time() - start
    for proc in [policy, sim, zenoh]:
        _kill(proc)
    time.sleep(3)

    raw   = _parse_trial_logs(log_dir)
    trial = _build_trial_dict(raw)
    score = compute_trial_score(trial)

    # Override with engine's actual tier scores when available (authoritative)
    if raw.get("engine_tier1_score") is not None:
        score["tier1"] = float(raw["engine_tier1_score"])
    if raw.get("engine_tier2_score") is not None:
        score["tier2"] = float(raw["engine_tier2_score"])
    if raw.get("engine_tier3_score") is not None:
        score["tier3"] = float(raw["engine_tier3_score"])
    score["total"] = round(score["tier1"] + score["tier2"] + score["tier3"], 4)

    score["elapsed_wall_s"] = round(elapsed_wall, 1)
    score["raw"]            = {k: (v.tolist() if hasattr(v, "tolist") else v)
                                for k, v in raw.items()}
    return score


# ── Scenario generator ────────────────────────────────────────────────────────

def gen_eval_scenarios(n: int, seed: int = 42) -> list[dict]:
    """
    Generate N randomised evaluation scenarios.
    Uses a fresh seed (different from training seeds 1-9).
    Difficulty: board ±3 cm / ±5° yaw — centre of the iter1–iter8 training range
    (most of the 240 training episodes were collected at ±2.5–5.0 cm).
    ±6.5 cm (the iter10 extreme) is deliberately omitted; those scenes are
    under-represented (only 20/240 episodes) and are better targeted by RL.
    """
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        out.append({
            "task_board_x":   round(_BOARD["x"]   + rng.uniform(-0.030, 0.030), 4),
            "task_board_y":   round(_BOARD["y"]   + rng.uniform(-0.030, 0.030), 4),
            "task_board_z":   _BOARD["z"],
            "task_board_yaw": round(_BOARD["yaw"] + rng.uniform(-0.087, 0.087), 4),
            "cable_roll":     round(_CABLE["roll"]  + rng.uniform(-0.050, 0.050), 4),
            "cable_pitch":    round(_CABLE["pitch"] + rng.uniform(-0.050, 0.050), 4),
            "cable_yaw":      round(_CABLE["yaw"]   + rng.uniform(-0.050, 0.050), 4),
        })
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="AIC three-tier scoring eval")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint dir (best_model/)")
    p.add_argument("--n_trials",   type=int, default=10)
    p.add_argument("--output",     required=True,
                   help="Output JSON path")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--timeout",    type=int, default=360,
                   help="Per-trial timeout (s)")
    args = p.parse_args()

    ckpt = str(Path(args.checkpoint).resolve())
    scenarios = gen_eval_scenarios(args.n_trials, seed=args.seed)
    trial_scores = []

    print(f"\n{'='*65}")
    print(f"  AIC Scoring Evaluation — iter10 model")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Trials     : {args.n_trials}  (seed={args.seed})")
    print(f"  Difficulty : board ±3 cm / ±5°, cable ±2.9°")
    print(f"{'='*65}")

    results_dir = Path(args.output).parent / "eval_trials"

    for i, sc in enumerate(scenarios):
        trial_id = f"iter10_trial{i+1:02d}"
        print(f"\n  [{i+1:2d}/{args.n_trials}] {trial_id}")
        print(f"    board=({sc['task_board_x']:.3f}, {sc['task_board_y']:.3f})  "
              f"yaw={sc['task_board_yaw']:.3f}  "
              f"cable_roll={sc['cable_roll']:.3f}")

        score = run_trial(ckpt, sc, trial_id, results_dir, timeout=args.timeout)
        trial_scores.append(score)

        ins    = score["raw"]["insertion_result"]
        dur    = score["raw"]["duration_s"]
        mf     = score["raw"]["max_force_N"]
        status = "SUCCESS" if ins == "correct" else f"FAIL ({ins})"
        print(f"    → {status}  total={score['total']:.1f} pts  "
              f"(T1={score['tier1']} T2={score['tier2']:.1f} T3={score['tier3']})")
        if dur:  print(f"       duration={dur:.1f}s  duration_score={score['duration_score']}")
        if mf:   print(f"       max_force={mf:.1f}N  force_penalty={score['force_penalty']}")

    summary = summarize_scores(trial_scores)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Summary — {args.n_trials} trials")
    print(f"{'='*65}")
    print(f"  {'Metric':<30}  {'Value':>10}")
    print(f"  {'-'*42}")
    rows = [
        ("Success rate",         f"{summary.get('success_rate', 0)*100:.0f}%  "
                                 f"({summary.get('n_success',0)}/{args.n_trials})"),
        ("Mean total score",     f"{summary.get('mean_total', 0):.1f} / 100"),
        ("Mean Tier 1",          f"{summary.get('mean_tier1', 0):.2f}"),
        ("Mean Tier 2",          f"{summary.get('mean_tier2', 0):.1f}"),
        ("  └ duration score",   f"{summary.get('mean_duration_score', 'N/A')}"),
        ("  └ force penalty",    f"{summary.get('mean_force_penalty',  'N/A')}"),
        ("  └ contact penalty",  f"{summary.get('mean_contact_penalty','N/A')}"),
        ("Mean Tier 3",          f"{summary.get('mean_tier3', 0):.1f}"),
    ]
    for label, val in rows:
        print(f"  {label:<30}  {str(val):>10}")
    print(f"\n  Note: smoothness / efficiency require trajectory bag data")
    print(f"{'='*65}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint":   ckpt,
        "n_trials":     args.n_trials,
        "seed":         args.seed,
        "summary":      summary,
        "trial_scores": trial_scores,
        "scenarios":    scenarios,
    }, indent=2))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
