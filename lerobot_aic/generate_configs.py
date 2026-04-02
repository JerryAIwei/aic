"""
generate_configs.py
Generate diverse aic_engine YAML configs for manual demo collection.

Run this LOCALLY to create N config files, then use each config
inside distrobox to record one episode per config.

Usage:
  python generate_configs.py --n 20                    # 20 configs in /tmp/aic_configs/
  python generate_configs.py --n 10 --seed 99          # different seed = different variations
  python generate_configs.py --n 5 --output_dir ./configs  # custom output dir
"""

import argparse
from pathlib import Path

# Reuse scenario generation and config writing from collect_sim_demos
from collect_sim_demos import gen_scenarios, write_trial_config


def main():
    p = argparse.ArgumentParser(description="Generate diverse aic_engine configs for demo collection")
    p.add_argument("--n", type=int, default=20, help="Number of configs to generate")
    p.add_argument("--seed", type=int, default=42, help="Random seed for scene variation")
    p.add_argument("--output_dir", type=str, default="/tmp/aic_configs",
                   help="Directory to write config YAML files")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenarios = gen_scenarios(args.n, seed=args.seed)

    print(f"Generating {args.n} diverse configs (seed={args.seed}) → {out}/\n")
    for i, sc in enumerate(scenarios):
        cfg_path = out / f"trial_{i}.yaml"
        write_trial_config(sc, cfg_path)
        print(f"  [{i:2d}] board=({sc['task_board_x']:+.4f}, {sc['task_board_y']:+.4f}, yaw={sc['task_board_yaw']:.4f})  "
              f"cable=({sc['cable_roll']:.4f}, {sc['cable_pitch']:.4f}, {sc['cable_yaw']:.4f})  → {cfg_path.name}")

    print(f"\n{'─' * 60}")
    print(f"Generated {args.n} configs in {out}/")
    print(f"\nTo record episode i inside distrobox:")
    print(f"  # Terminal 1 (distrobox):")
    print(f"  /entrypoint.sh ground_truth:=true start_aic_engine:=true \\")
    print(f"    aic_engine_config_file:={out}/trial_<i>.yaml")
    print(f"\n  # Terminal 2 (distrobox):")
    print(f"  pixi run ros2 run aic_model aic_model \\")
    print(f"    --ros-args -p use_sim_time:=true \\")
    print(f"    -p policy:=aic_example_policies.ros.RecordCheatCode")
    print(f"\n  After 'insert_cable() done' appears, Ctrl+C both terminals.")
    print(f"  Episode saved to /tmp/aic_recordings/ep_<timestamp>.npz")
    print(f"\nAfter collecting all episodes, build dataset LOCALLY:")
    print(f"  python collect_sim_demos.py --build_only --data_dir /tmp/aic_recordings")


if __name__ == "__main__":
    main()
