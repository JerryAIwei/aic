"""
RunSimDiffusion.py
Runs the Diffusion Policy trained on sim-collected data
(checkpoints_diffusion_aic_cable_insertion_sim/best_model).

Architecture matches train_diffusion.py:
  - ResNet-18 × 3 cameras → spatial softmax
  - 1D U-Net (256, 512, 1024) denoiser
  - DDPM 100-step, ε-prediction
  - n_obs_steps=2, horizon=16, n_action_steps=8
  - action: 6D cartesian velocity

At each inference call the policy:
  1. Buffers the last 2 observations
  2. Normalises state and images
  3. Runs DDPM denoising to predict 16 future actions
  4. Executes actions 1–8 (one per control step at 20 Hz) before re-planning

Policy parameter for aic_model:
  aic_example_policies.ros.RunSimDiffusion.RunSimDiffusion
"""

import collections
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from std_msgs.msg import Header

# ── checkpoint location ───────────────────────────────────────────────────────
# Allow override via environment variable; default to the workspace path
CKPT_DIR = Path(
    os.environ.get(
        "AIC_DIFFUSION_CKPT",
        "/workspace/aic/lerobot_aic/checkpoints_diffusion_aic_cable_insertion_sim/best_model",
    )
)

IMG_H, IMG_W = 128, 144   # must match recording resolution
N_OBS   = 2
N_ACTION = 8


def _ros_img_to_tensor(img_msg, device) -> torch.Tensor:
    """sensor_msgs/Image → (1, 3, H, W) float32 [0,1] tensor on device."""
    arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.width, 3
    )
    arr = cv2.resize(arr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return t.to(device)


def _obs_to_state_tensor(obs, device) -> torch.Tensor:
    """Extract 26-D state → (1, 26) float32 tensor."""
    cs = obs.controller_state
    tcp = cs.tcp_pose
    vel = cs.tcp_velocity
    state = np.array(
        [
            tcp.position.x, tcp.position.y, tcp.position.z,
            tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
            vel.linear.x, vel.linear.y, vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
            *cs.tcp_error,
            *list(obs.joint_states.position[:7]),
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(state).unsqueeze(0).to(device)


class RunSimDiffusion(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"RunSimDiffusion loading checkpoint from {CKPT_DIR}")

        if not CKPT_DIR.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CKPT_DIR}\n"
                "Run: python lerobot_aic/train_diffusion.py "
                "--repo_id local/aic_cable_insertion_sim --run_name aic_cable_insertion_sim"
            )

        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        self.model = DiffusionPolicy.from_pretrained(str(CKPT_DIR))
        self.model.eval()
        self.model.to(self.device)

        # Observation ring-buffer: deque of length N_OBS
        self._obs_buf: collections.deque = collections.deque(maxlen=N_OBS)
        # Action chunk buffer: remaining pre-planned actions
        self._action_queue: collections.deque = collections.deque()

        n_params = sum(p.numel() for p in self.model.parameters())
        self.get_logger().info(f"DiffusionPolicy loaded on {self.device}  ({n_params:,} params)")

    # ── observation preprocessing ─────────────────────────────────────────────

    def _preprocess_obs(self, obs_list: list) -> dict:
        """Stack N_OBS observations into model input tensors (on device)."""
        assert len(obs_list) == N_OBS

        # Images: (1, N_OBS, 3, H, W)
        def stack_imgs(key):
            frames = [_ros_img_to_tensor(getattr(o, key), self.device) for o in obs_list]
            return torch.stack(frames, dim=1)   # (1, N_OBS, 3, H, W)

        # State: (1, N_OBS, 26)
        states = torch.stack(
            [_obs_to_state_tensor(o, self.device) for o in obs_list], dim=1
        )  # (1, N_OBS, 26)

        return {
            "observation.state":                   states,
            "observation.images.center_camera":    stack_imgs("center_image"),
            "observation.images.left_camera":      stack_imgs("left_image"),
            "observation.images.right_camera":     stack_imgs("right_image"),
        }

    # ── action execution ──────────────────────────────────────────────────────

    def _action_to_motion_update(self, action_vec: np.ndarray) -> MotionUpdate:
        """Convert 6D velocity action to a MotionUpdate pose command.

        We send the current TCP pose displaced by the predicted velocity × dt,
        which translates the velocity prediction into a pose target for the
        Cartesian impedance controller.
        """
        vx, vy, vz, wx, wy, wz = action_vec.tolist()
        dt = 1.0 / 20.0   # 20 Hz

        # We will overwrite pose in insert_cable() with current + delta,
        # so just pack the delta here and let insert_cable resolve TCP pose.
        # This field is used as a velocity command via displacement.
        return (vx, vy, vz, wx, wy, wz, dt)

    # ── main policy loop ──────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info("RunSimDiffusion.insert_cable() start")
        self._obs_buf.clear()
        self._action_queue.clear()
        self.model.reset()

        max_steps = 600     # 30 s at 20 Hz
        send_feedback("RunSimDiffusion: running diffusion policy")

        for step in range(max_steps):
            obs = get_observation()
            self._obs_buf.append(obs)

            # Pad buffer to N_OBS by repeating first observation
            obs_list = list(self._obs_buf)
            while len(obs_list) < N_OBS:
                obs_list.insert(0, obs_list[0])

            # Re-plan when action queue is empty
            if not self._action_queue:
                batch = self._preprocess_obs(obs_list)
                with torch.no_grad():
                    actions = self.model.select_action(batch)
                # actions: (1, n_action_steps, 6) or (n_action_steps, 6)
                if actions.dim() == 3:
                    actions = actions[0]   # (n_action_steps, 6)
                for a in actions.cpu().numpy():
                    self._action_queue.append(a)
                self.get_logger().info(
                    f"step {step}: re-planned {len(actions)} actions"
                )

            action = self._action_queue.popleft()   # (6,)

            # Convert velocity → pose target (integrate current TCP pose + delta)
            cs = obs.controller_state
            dt = 1.0 / 20.0
            vx, vy, vz, wx, wy, wz = action.tolist()
            cur = cs.tcp_pose
            target_pose = Pose(
                position=Point(
                    x=cur.position.x + vx * dt,
                    y=cur.position.y + vy * dt,
                    z=cur.position.z + vz * dt,
                ),
                orientation=cur.orientation,   # keep current orientation for now
            )
            self.set_pose_target(move_robot=move_robot, pose=target_pose)
            self.sleep_for(dt)

            if step % 50 == 0:
                send_feedback(f"RunSimDiffusion: step {step}/{max_steps}")

        self.get_logger().info("RunSimDiffusion.insert_cable() finished")
        return True
