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

# ── checkpoint & dataset locations ───────────────────────────────────────────
# Allow override via environment variables; defaults to the iter10 model
CKPT_DIR = Path(
    os.environ.get(
        "AIC_DIFFUSION_CKPT",
        "/workspace/aic/lerobot_aic/checkpoints_diffusion_iter10/best_model",
    )
)
DATASET_ID = os.environ.get(
    "AIC_DIFFUSION_DATASET",
    "local/aic_cable_insertion_iter10",
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
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.types import PolicyAction
        self._PolicyAction = PolicyAction

        self.model = DiffusionPolicy.from_pretrained(str(CKPT_DIR))
        self.model.eval()
        self.model.to(self.device)

        # Load normalization preprocessor from training dataset stats.
        # The model was trained with MIN_MAX normalization for STATE/ACTION and
        # MEAN_STD for images. Without this, select_action receives raw sensor
        # values that are completely out of the model's expected input range.
        self.get_logger().info(f"Loading dataset normalization stats from {DATASET_ID}")
        ds = LeRobotDataset(DATASET_ID)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.model.config, dataset_stats=ds.meta.stats
        )
        self.get_logger().info("Normalization preprocessor ready")

        n_params = sum(p.numel() for p in self.model.parameters())
        self.get_logger().info(f"DiffusionPolicy loaded on {self.device}  ({n_params:,} params)")

    # ── observation preprocessing ─────────────────────────────────────────────

    def _preprocess_obs(self, obs) -> dict:
        """Convert a single observation into model input tensors (on device).

        select_action() manages its own obs history queue internally, so we
        provide single-timestep tensors: (1, C, H, W) for images, (1, 26)
        for state.  select_action() stacks them into (B, n_obs, ...) internally.
        """
        return {
            "observation.state":                   _obs_to_state_tensor(obs, self.device),
            "observation.images.center_camera":    _ros_img_to_tensor(obs.center_image, self.device),
            "observation.images.left_camera":      _ros_img_to_tensor(obs.left_image, self.device),
            "observation.images.right_camera":     _ros_img_to_tensor(obs.right_image, self.device),
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
        self.model.reset()   # clear model's internal obs/action queues

        max_steps = 2400    # 120 s at 20 Hz — matches engine time_limit
        send_feedback("RunSimDiffusion: running diffusion policy")

        for step in range(max_steps):
            obs = get_observation()

            # select_action() handles obs history and action chunk caching
            # internally; pass a single-timestep batch each control step.
            # Apply normalization before feeding to model (MIN_MAX for state,
            # MEAN_STD for images) — the model was trained on normalized inputs.
            raw_batch = self._preprocess_obs(obs)
            norm_batch = self.preprocessor(raw_batch)
            with torch.no_grad():
                norm_action = self.model.select_action(norm_batch)
            # select_action returns shape (1, 6) or (6,); squeeze to (6,).
            # postprocessor expects a 1-D PolicyAction (Tensor subclass).
            action = self.postprocessor(
                norm_action.reshape(6).as_subclass(self._PolicyAction)
            ).cpu().numpy()   # (6,) in m/s

            # Convert velocity → pose target (integrate current TCP pose + delta)
            cs = obs.controller_state
            dt = 1.0 / 20.0
            vx, vy, vz, wx, wy, wz = action.tolist()
            cur = cs.tcp_pose

            # Integrate angular velocity into orientation via axis-angle rotation.
            # tcp_velocity is expressed in the world (base) frame.
            omega = np.array([wx, wy, wz], dtype=np.float64)
            angle = float(np.linalg.norm(omega)) * dt
            q = np.array([cur.orientation.x, cur.orientation.y,
                          cur.orientation.z, cur.orientation.w], dtype=np.float64)
            if angle > 1e-8:
                axis = omega / np.linalg.norm(omega)
                sh, ch = np.sin(angle / 2.0), np.cos(angle / 2.0)
                dq = np.array([axis[0]*sh, axis[1]*sh, axis[2]*sh, ch])
                # World-frame rotation: new_q = dq ⊗ q_old
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
            self.sleep_for(dt)

            if step % 50 == 0:
                send_feedback(f"RunSimDiffusion: step {step}/{max_steps}")

        self.get_logger().info("RunSimDiffusion.insert_cable() finished")
        return True
