"""
RunVisionDiffusion.py
Runs the vision-guided Diffusion Policy fine-tuned on combined iter10 +
high-variation (±10 cm board) data with image augmentation.

This policy differs from RunSimDiffusion in checkpoint and dataset sources:
  - Checkpoint: checkpoints_diffusion_vision_finetune_combined/best_model
  - Dataset stats: local/aic_cable_insertion_iter10 (primary)
  - Training: fine-tuned from iter10 with ColorJitter + spatial-shift augmentation
  - Board variation: ±10 cm XY (5× larger than original ±2 cm)

Architecture (same as RunSimDiffusion):
  - ResNet-18 × 3 cameras → spatial softmax
  - 1D U-Net (256, 512, 1024) denoiser
  - DDPM 100-step, ε-prediction
  - n_obs_steps=2, horizon=16, n_action_steps=8
  - action: 6D cartesian velocity

Policy parameter for aic_model:
  aic_example_policies.ros.RunVisionDiffusion.RunVisionDiffusion
"""

import os
import time
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
CKPT_DIR = Path(
    os.environ.get(
        "AIC_VISION_CKPT",
        "/workspace/aic/lerobot_aic/checkpoints_diffusion_vision_10cm_v2/best_model",
    )
)
# Use 10cm dataset stats for normalization (matches training distribution).
DATASET_ID = os.environ.get(
    "AIC_VISION_DATASET",
    "local/aic_cable_insertion_10cm",
)

IMG_H, IMG_W = 128, 144
N_OBS    = 2
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


class RunVisionDiffusion(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"RunVisionDiffusion loading checkpoint from {CKPT_DIR}")

        if not CKPT_DIR.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CKPT_DIR}\n"
                "Run the vision fine-tuning pipeline:\n"
                "  python lerobot_aic/train_vision_fast.py "
                "--npz_dir /tmp/aic_recordings_iter10_backup --steps 6000 "
                "--run_name vision_fast_v2"
            )

        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.types import PolicyAction
        self._PolicyAction = PolicyAction

        self.model = DiffusionPolicy.from_pretrained(str(CKPT_DIR))
        self.model.eval()
        self.model.to(self.device)

        self.get_logger().info(f"Loading normalization stats from {DATASET_ID}")
        ds = LeRobotDataset(DATASET_ID)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.model.config, dataset_stats=ds.meta.stats
        )
        self.get_logger().info("Normalization preprocessor ready")

        n_params = sum(p.numel() for p in self.model.parameters())
        self.get_logger().info(
            f"RunVisionDiffusion loaded on {self.device}  ({n_params:,} params)"
        )

    def _preprocess_obs(self, obs) -> dict:
        return {
            "observation.state":                   _obs_to_state_tensor(obs, self.device),
            "observation.images.center_camera":    _ros_img_to_tensor(obs.center_image, self.device),
            "observation.images.left_camera":      _ros_img_to_tensor(obs.left_image, self.device),
            "observation.images.right_camera":     _ros_img_to_tensor(obs.right_image, self.device),
        }

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info("RunVisionDiffusion.insert_cable() start")
        self.model.reset()

        # Wait for the first valid observation before starting the policy loop.
        obs = get_observation()
        while obs is None:
            self.sleep_for(0.05)
            obs = get_observation()
        self.get_logger().info("RunVisionDiffusion: first observation received, starting loop")

        max_steps = 2400    # 120 s at 20 Hz
        send_feedback("RunVisionDiffusion: running vision-guided diffusion policy")

        for step in range(max_steps):
            obs = get_observation()

            raw_batch  = self._preprocess_obs(obs)
            norm_batch = self.preprocessor(raw_batch)
            with torch.no_grad():
                norm_action = self.model.select_action(norm_batch)
            action = self.postprocessor(
                norm_action.reshape(6).as_subclass(self._PolicyAction)
            ).cpu().numpy()

            cs = obs.controller_state
            dt = 1.0 / 20.0
            vx, vy, vz, wx, wy, wz = action.tolist()
            cur = cs.tcp_pose

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
            time.sleep(dt)   # real-time pace; sim-clock sleep_for blocks after CUDA warmup

            if step % 50 == 0:
                send_feedback(f"RunVisionDiffusion: step {step}/{max_steps}")

        self.get_logger().info("RunVisionDiffusion.insert_cable() finished")
        return True
