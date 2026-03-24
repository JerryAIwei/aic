"""
compare_with_act_baseline.py
Run both the trained SimDiffusion policy and the RunACT baseline through
the AIC evaluation engine, then print a side-by-side comparison of:
  - Training metrics (val loss / convergence)
  - Scoring results from $AIC_RESULTS_DIR (success, time, force)

Usage:
    # Run both evaluations (requires simulation environment):
    python compare_with_act_baseline.py --run_eval

    # Compare already-collected results only:
    python compare_with_act_baseline.py
"""

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOTFS = "/opt/aic_rootfs"
ROS_SETUP = "/ws_aic/install/setup.bash"
WORKSPACE = Path("/workspace/aic")

RESULTS_BASE = Path(os.environ.get("AIC_RESULTS_DIR", Path.home() / "aic_results"))

# ── training metrics ──────────────────────────────────────────────────────────

def _load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _best_val(records: list[dict]) -> tuple[float, int]:
    bv, be = float("inf"), 0
    for r in records:
        v = r.get("val_loss", float("inf"))
        if v < bv:
            bv, be = v, r.get("epoch", 0)
    return bv, be


def print_training_comparison():
    sim_metrics  = _load_metrics(HERE / "outputs_diffusion_aic_cable_insertion_sim" / "metrics.json")
    syn_metrics  = _load_metrics(HERE / "outputs_diffusion" / "metrics.json")
    act_metrics  = _load_metrics(HERE / "outputs_act" / "metrics.json")

    print("\n" + "=" * 65)
    print("  Training Metrics Comparison")
    print("=" * 65)
    header = f"  {'Metric':<32}  {'Sim-Diffusion':>13}  {'Syn-Diffusion':>13}"
    print(header)
    print("  " + "-" * 61)

    def row(label, a, b):
        fa = f"{a:.5f}" if isinstance(a, float) else str(a)
        fb = f"{b:.5f}" if isinstance(b, float) else str(b)
        print(f"  {label:<32}  {fa:>13}  {fb:>13}")

    if sim_metrics:
        sv, se = _best_val(sim_metrics)
        n_ep   = sim_metrics[-1].get("epoch", len(sim_metrics))
        row("Epochs trained",    n_ep,       sim_metrics[-1].get("epoch", len(syn_metrics)) if syn_metrics else "N/A")
        row("Best val loss",     sv,         _best_val(syn_metrics)[0] if syn_metrics else float("nan"))
        row("Best val epoch",    se,         _best_val(syn_metrics)[1] if syn_metrics else 0)
        row("Final train loss",  sim_metrics[-1].get("train_loss", float("nan")),
                                 syn_metrics[-1].get("train_loss", float("nan")) if syn_metrics else float("nan"))
    else:
        print("  [sim diffusion metrics not found — run training first]")

    print("\n  Notes")
    print("  • Sim-Diffusion trained on real sim trajectories (RecordCheatCode)")
    print("  • Syn-Diffusion trained on procedurally generated synthetic data")
    print("  • Both use DDPM noise-pred MSE — val loss is comparable")
    if act_metrics:
        bv, _ = _best_val(act_metrics)
        print(f"  • ACT baseline best val loss: {bv:.5f} (CVAE ELBO — different scale)")


# ── scoring evaluation ────────────────────────────────────────────────────────

def _ros_env() -> dict:
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
    env.pop("DISPLAY", None)
    ws_policies = str(WORKSPACE / "aic_example_policies")
    env["PYTHONPATH"] = f"{ws_policies}:{env.get('PYTHONPATH', '')}"
    return env


def run_eval(policy: str, results_subdir: str, timeout: int = 300) -> Path:
    """Run one full eval episode and return the results directory."""
    results_dir = RESULTS_BASE / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    env = _ros_env()
    env["AIC_RESULTS_DIR"] = str(results_dir)

    print(f"\n  Evaluating: {policy}")
    print(f"  Results  → {results_dir}")

    # Zenoh router
    zenoh = subprocess.Popen(
        ["bash", "-c", f". {ROS_SETUP} && ros2 run rmw_zenoh_cpp rmw_zenohd"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    # Policy node
    policy_proc = subprocess.Popen(
        ["bash", "-c",
         f". {ROS_SETUP} && ros2 run aic_model aic_model "
         f"--ros-args -p use_sim_time:=true -p policy:={policy} "
         f"2>&1 | tee {results_dir}/policy.log"],
        env=env,
    )
    time.sleep(2)

    # Simulation + engine (foreground, exits after one trial)
    sim = subprocess.Popen(
        ["bash", "-c",
         f". {ROS_SETUP} && ros2 launch aic_bringup aic_gz_bringup.launch.py "
         f"gazebo_gui:=false ground_truth:=true "
         f"start_aic_engine:=true shutdown_on_aic_engine_exit:=true "
         f"spawn_task_board:=true attach_cable_to_gripper:=true "
         f"2>&1 | tee {results_dir}/sim.log"],
        env=env,
    )

    try:
        sim.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  WARNING: eval timed out after {timeout}s")

    for proc in [policy_proc, zenoh]:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(3)
    return results_dir


# ── scoring results parser ────────────────────────────────────────────────────

def _parse_results(results_dir: Path) -> dict:
    """Extract insertion success and timing from AIC results files."""
    out = {"success": None, "time_s": None, "max_force_N": None}

    # Look for JSON result files
    for jf in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
            if "success" in data:
                out["success"] = data.get("success")
            if "elapsed_seconds" in data:
                out["time_s"] = data.get("elapsed_seconds")
            if "max_force" in data:
                out["max_force_N"] = data.get("max_force")
        except Exception:
            pass

    # Fall back to log scanning for success indicator
    if out["success"] is None:
        for log in results_dir.glob("*.log"):
            text = log.read_text(errors="ignore")
            if "insertion_success" in text.lower() or "succeeded" in text.lower():
                out["success"] = True
                break
            if "failed" in text.lower() or "aborted" in text.lower():
                out["success"] = False

    return out


def print_scoring_comparison(sim_diff_dir: Path, act_dir: Path):
    sd = _parse_results(sim_diff_dir) if sim_diff_dir.exists() else {}
    ac = _parse_results(act_dir)      if act_dir.exists()      else {}

    print("\n" + "=" * 65)
    print("  AIC Scoring Comparison")
    print("=" * 65)
    print(f"  {'Metric':<30}  {'SimDiffusion':>14}  {'RunACT':>14}")
    print("  " + "-" * 61)

    def fmt(v):
        if v is None: return "N/A"
        if isinstance(v, bool): return "YES" if v else "NO"
        if isinstance(v, float): return f"{v:.2f}"
        return str(v)

    for label, sk, ak in [
        ("Insertion success",    "success",     "success"),
        ("Time to insertion (s)", "time_s",      "time_s"),
        ("Max force (N)",         "max_force_N", "max_force_N"),
    ]:
        print(f"  {label:<30}  {fmt(sd.get(sk)):>14}  {fmt(ac.get(ak)):>14}")

    print()
    if not sd and not ac:
        print("  [No scoring results found — run with --run_eval to collect them]")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compare SimDiffusion vs RunACT baseline")
    p.add_argument("--run_eval", action="store_true",
                   help="Actually run the simulation evaluations (requires sim environment)")
    p.add_argument("--timeout",  type=int, default=300,
                   help="Max seconds to wait per evaluation run")
    args = p.parse_args()

    if args.run_eval:
        print("Running simulation evaluations...")
        print("(This launches Gazebo — ensure GPU/display environment is correct)")

        run_eval(
            policy="aic_example_policies.ros.RunSimDiffusion.RunSimDiffusion",
            results_subdir="sim_diffusion_eval",
            timeout=args.timeout,
        )
        run_eval(
            policy="aic_example_policies.ros.RunACT.RunACT",
            results_subdir="run_act_eval",
            timeout=args.timeout,
        )

    print_training_comparison()
    print_scoring_comparison(
        sim_diff_dir=RESULTS_BASE / "sim_diffusion_eval",
        act_dir=RESULTS_BASE / "run_act_eval",
    )

    # Save JSON summary
    summary_path = HERE / "outputs_diffusion_aic_cable_insertion_sim" / "vs_act_comparison.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    sim_metrics = _load_metrics(HERE / "outputs_diffusion_aic_cable_insertion_sim" / "metrics.json")
    act_metrics = _load_metrics(HERE / "outputs_act" / "metrics.json")

    summary = {
        "sim_diffusion": {
            "best_val_loss":  _best_val(sim_metrics)[0] if sim_metrics else None,
            "best_val_epoch": _best_val(sim_metrics)[1] if sim_metrics else None,
            "scoring":        _parse_results(RESULTS_BASE / "sim_diffusion_eval")
                              if (RESULTS_BASE / "sim_diffusion_eval").exists() else {},
        },
        "run_act_baseline": {
            "best_val_loss":  _best_val(act_metrics)[0] if act_metrics else None,
            "scoring":        _parse_results(RESULTS_BASE / "run_act_eval")
                              if (RESULTS_BASE / "run_act_eval").exists() else {},
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
