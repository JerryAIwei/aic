"""
RunMLPPolicy.py
Runs the MLP actor-critic policy trained by train_rl.py (BC pre-train + PPO fine-tune).

State-only inference — no cameras → <1 ms per step, full 20 Hz without CUDA warmup issues.

Configuration (written by train_rl.py before each episode):
  /tmp/mlp_rl/config.json  — state_mean, state_std, port_pos
  /tmp/mlp_rl/weights.pt   — ActorCritic state_dict

Trajectory is saved to /tmp/aic_recordings/ep_{timestamp}_mlp.npz after each episode
and the path is written to /tmp/aic_recordings/latest.txt so train_rl.py can pick it up.

Policy parameter for aic_model:
  aic_example_policies.ros.RunMLPPolicy.RunMLPPolicy
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion

MLP_DIR  = Path(os.environ.get("MLP_RL_DIR", "/tmp/mlp_rl"))
SAVE_DIR = Path("/tmp/aic_recordings")
STATE_DIM  = 26
ACTION_DIM = 6


# ── Network (must stay in sync with train_rl.py ActorCritic) ─────────────────

class ActorCritic(nn.Module):
    """Shared-trunk actor-critic.  Only the actor is used at inference time."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -1.5))
        self.critic        = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        h    = self.trunk(state)
        mean = self.actor_mean(h)
        std  = self.actor_log_std.exp().expand_as(mean)
        val  = self.critic(h).squeeze(-1)
        return mean, std, val


# ── State extraction ──────────────────────────────────────────────────────────

def _obs_to_state(obs) -> np.ndarray:
    """Extract 26-D state vector from Observation message (matches train_rl.py ordering)."""
    cs  = obs.controller_state
    tcp = cs.tcp_pose
    vel = cs.tcp_velocity
    return np.array([
        tcp.position.x,    tcp.position.y,    tcp.position.z,
        tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
        vel.linear.x,  vel.linear.y,  vel.linear.z,
        vel.angular.x, vel.angular.y, vel.angular.z,
        *list(cs.tcp_error),
        *list(obs.joint_states.position[:7]),
    ], dtype=np.float32)


# ── Policy class ──────────────────────────────────────────────────────────────

class RunMLPPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path  = MLP_DIR / "config.json"
        weights_path = MLP_DIR / "weights.pt"

        if not config_path.exists():
            raise FileNotFoundError(
                f"MLP config not found: {config_path}\n"
                "Run: python lerobot_aic/train_rl.py --rl_iter 0   (BC-only pass)"
            )
        if not weights_path.exists():
            raise FileNotFoundError(f"MLP weights not found: {weights_path}")

        config = json.loads(config_path.read_text())
        self.state_mean = np.array(config["state_mean"], dtype=np.float32)
        self.state_std  = np.array(config["state_std"],  dtype=np.float32)

        self.model = ActorCritic(STATE_DIM, ACTION_DIM, hidden=256)
        self.model.load_state_dict(
            torch.load(str(weights_path), map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.model.to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        self.get_logger().info(
            f"RunMLPPolicy loaded ({n_params:,} params) on {self.device}  "
            f"from {weights_path}"
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info("RunMLPPolicy.insert_cable() start")

        # Wait for first valid observation
        obs = get_observation()
        while obs is None:
            self.sleep_for(0.05)
            obs = get_observation()
        self.get_logger().info("RunMLPPolicy: first observation received, starting loop")

        max_steps = 2400    # 120 s at 20 Hz
        dt = 1.0 / 20.0
        send_feedback("RunMLPPolicy: running BC+PPO MLP policy")

        states_log  = []
        actions_log = []

        for step in range(max_steps):
            obs   = get_observation()
            state = _obs_to_state(obs)
            states_log.append(state)

            # Normalize and run actor
            s_norm = (state - self.state_mean) / (self.state_std + 1e-8)
            s_t    = torch.from_numpy(s_norm).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                mean, _, _ = self.model(s_t)
                action = mean.squeeze(0).cpu().numpy()   # (6,) deterministic mean

            actions_log.append(action)

            # Integrate velocity → pose target (same as RunSimDiffusion)
            cs  = obs.controller_state
            cur = cs.tcp_pose
            vx, vy, vz, wx, wy, wz = action.tolist()

            omega = np.array([wx, wy, wz], dtype=np.float64)
            angle = float(np.linalg.norm(omega)) * dt
            q = np.array([cur.orientation.x, cur.orientation.y,
                          cur.orientation.z, cur.orientation.w], dtype=np.float64)
            if angle > 1e-8:
                axis = omega / np.linalg.norm(omega)
                sh, ch = np.sin(angle / 2.0), np.cos(angle / 2.0)
                dq = np.array([axis[0]*sh, axis[1]*sh, axis[2]*sh, ch])
                ax, ay, az, aw = dq
                bx, by, bz, bw = q
                q = np.array([
                    aw*bx + ax*bw + ay*bz - az*by,
                    aw*by - ax*bz + ay*bw + az*bx,
                    aw*bz + ax*by - ay*bx + az*bw,
                    aw*bw - ax*bx - ay*by - az*bz,
                ])
                q /= np.linalg.norm(q)

            target_pose = Pose(
                position=Point(
                    x=cur.position.x + vx * dt,
                    y=cur.position.y + vy * dt,
                    z=cur.position.z + vz * dt,
                ),
                orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
            )
            self.set_pose_target(move_robot=move_robot, pose=target_pose)
            time.sleep(dt)   # real-time pace (not sim-clock, avoids CUDA-warmup deadlock)

            if step % 100 == 0:
                tcp = state[:3]
                send_feedback(
                    f"RunMLPPolicy: step {step}/{max_steps}  "
                    f"tcp=({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f})"
                )

        self._save_trajectory(states_log, actions_log)
        self.get_logger().info("RunMLPPolicy.insert_cable() finished")
        return True

    def _save_trajectory(self, states_log: list, actions_log: list) -> None:
        """Save (states, actions) to npz and write path to latest.txt."""
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        save_path = SAVE_DIR / f"ep_{ts}_mlp.npz"
        np.savez(
            str(save_path),
            states=np.array(states_log,  dtype=np.float32),
            actions=np.array(actions_log, dtype=np.float32),
        )
        (SAVE_DIR / "latest.txt").write_text(str(save_path))
        self.get_logger().info(
            f"Trajectory saved: {save_path}  ({len(states_log)} steps)"
        )
