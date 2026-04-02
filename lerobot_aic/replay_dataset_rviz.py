#!/usr/bin/env python3
"""
replay_dataset_rviz.py
Replays a LeRobot dataset (or raw synthetic episodes) as /joint_states
messages so the trajectory can be visualised in RViz.

Requirements:
  - ROS2 (rclpy) available (run inside distrobox or Docker)
  - Robot URDF loaded (via start_simulator.sh or robot_state_publisher)
  - RViz open with the aic.rviz config

Usage (inside distrobox/docker):
    # Replay the LeRobot dataset:
    python3 lerobot_aic/replay_dataset_rviz.py

    # Replay a specific episode:
    python3 lerobot_aic/replay_dataset_rviz.py --episode 5

    # Generate a raw episode on-the-fly (no dataset needed):
    python3 lerobot_aic/replay_dataset_rviz.py --synthetic --strategy curved_approach

    # Faster/slower playback:
    python3 lerobot_aic/replay_dataset_rviz.py --speed 2.0
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np

# ─── Joint names must match the URDF ─────────────────────────────────────────
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
FPS = 20


def get_episode_from_dataset(episode_idx: int):
    """Load joint positions from a LeRobot dataset episode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for repo_id in ["local/aic_cable_insertion_large",
                     "local/aic_cable_insertion_sim",
                     "local/aic_cable_insertion"]:
        try:
            ds = LeRobotDataset(repo_id=repo_id, video_backend="pyav")
            break
        except Exception:
            continue
    else:
        print("No LeRobot dataset found. Use --synthetic to generate on-the-fly.")
        sys.exit(1)

    print(f"Dataset: {repo_id}  ({len(ds)} frames)")

    # Find frame indices for the requested episode
    ep_indices = []
    for i in range(len(ds)):
        frame = ds[i]
        state = frame["observation.state"].numpy()
        ep_indices.append(state)
        # Episodes are sequential; each has ~100 frames
        if len(ep_indices) > 0 and i > 0:
            # Check if we've collected enough for the target episode
            pass

    # Simpler: just grab frames by offset
    # Each episode is ~100 steps. Joint positions are state[19:25] (6 joints)
    ep_start = episode_idx * 100
    ep_end   = min(ep_start + 100, len(ds))
    if ep_start >= len(ds):
        print(f"Episode {episode_idx} out of range (max: {len(ds) // 100 - 1})")
        sys.exit(1)

    joints = []
    for i in range(ep_start, ep_end):
        state = ds[i]["observation.state"].numpy()
        # Joint positions are indices 19-24 (6 joints), index 25 is gripper
        joints.append(state[19:25])

    print(f"Episode {episode_idx}: {len(joints)} steps")
    return np.array(joints)


def get_synthetic_episode(strategy: str, seed: int = 42):
    """Generate a synthetic episode without needing a dataset."""
    # Import from the dataset creation script
    sys.path.insert(0, str(Path(__file__).parent))
    from create_larger_dataset import generate_episode

    states, actions, _, _, _, task = generate_episode(
        seed=seed, strategy=strategy, episode_len=100
    )
    # Joint positions are state[19:25]
    joints = states[:, 19:25]
    print(f"Synthetic episode: {strategy}, {len(joints)} steps")
    return joints


def replay_ros2(joints: np.ndarray, speed: float, loop: bool):
    """Publish joint states to ROS2 /joint_states topic."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("dataset_replay")
    pub  = node.create_publisher(JointState, "/joint_states", 10)

    dt = 1.0 / (FPS * speed)

    print(f"Publishing to /joint_states at {FPS * speed:.1f} Hz (speed={speed}x)")
    print("Open RViz to visualize. Press Ctrl+C to stop.\n")

    try:
        iteration = 0
        while True:
            iteration += 1
            for step, q in enumerate(joints):
                msg = JointState()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.name     = JOINT_NAMES
                msg.position = q.tolist()
                msg.velocity = [0.0] * 6
                msg.effort   = [0.0] * 6
                pub.publish(msg)

                # Print progress
                if step % 20 == 0:
                    pos_str = ", ".join(f"{v:.2f}" for v in q[:3])
                    print(f"  Step {step:>3}/{len(joints)}  joints=[{pos_str}, ...]")

                time.sleep(dt)

            if not loop:
                break
            print(f"\n  Loop {iteration} complete. Restarting…\n")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def replay_stdout(joints: np.ndarray, speed: float):
    """Print joint states to stdout (no ROS2 needed) for debugging."""
    dt = 1.0 / (FPS * speed)
    print(f"Replaying {len(joints)} steps at {FPS * speed:.1f} Hz (stdout mode)\n")
    print(f"{'Step':>5}  {'shoulder_pan':>12} {'shoulder_lift':>13} {'elbow':>8} "
          f"{'wrist_1':>8} {'wrist_2':>8} {'wrist_3':>8}")
    print("─" * 78)

    for step, q in enumerate(joints):
        vals = "  ".join(f"{v:>8.3f}" for v in q)
        print(f"{step:>5}  {vals}")
        time.sleep(dt)

    print(f"\nDone. {len(joints)} steps replayed.")


def main():
    parser = argparse.ArgumentParser(
        description="Replay synthetic/dataset trajectories in RViz"
    )
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to replay (default: 0)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate on-the-fly (no dataset needed)")
    parser.add_argument("--strategy", type=str, default="straight_approach",
                        choices=["straight_approach", "curved_approach",
                                 "slow_insertion", "angled_approach", "recovery"],
                        help="Trajectory strategy for --synthetic")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --synthetic")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the replay continuously")
    parser.add_argument("--no-ros", action="store_true",
                        help="Print to stdout instead of publishing to ROS2")
    args = parser.parse_args()

    # Get joint trajectory
    if args.synthetic:
        joints = get_synthetic_episode(args.strategy, args.seed)
    else:
        joints = get_episode_from_dataset(args.episode)

    # Replay
    if args.no_ros:
        replay_stdout(joints, args.speed)
    else:
        replay_ros2(joints, args.speed, args.loop)


if __name__ == "__main__":
    main()
