"""
RecordCheatCode.py
CheatCode policy that records demonstrations for imitation learning.

Runs the same ground-truth-guided insertion motion as CheatCode, but
calls get_observation() at every control step and saves:
  states[T, 26]       tcp_pose(7) + tcp_vel(6) + tcp_err(6) + joints(7)
  actions[T, 6]       tcp linear+angular velocity (what the robot is doing)
  center[T, 128, 144, 3]  center camera uint8 RGB
  left[T, 128, 144, 3]
  right[T, 128, 144, 3]

Episodes are saved to /tmp/aic_recordings/ep_<timestamp>.npz.
The orchestrator (collect_sim_demos.py) polls /tmp/aic_recordings/latest.txt
to know when an episode is ready.

Policy parameter for aic_model:
  aic_example_policies.ros.RecordCheatCode.RecordCheatCode
"""

import time
from pathlib import Path

import cv2
import numpy as np

from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from tf2_ros import TransformException

# Target image resolution (1024×1152 cameras → 128×144, scale 0.125)
IMG_H, IMG_W = 128, 144
SAVE_DIR = Path("/tmp/aic_recordings")


def _ros_img_to_numpy(img_msg) -> np.ndarray | None:
    """Convert sensor_msgs/Image → (IMG_H, IMG_W, 3) uint8 RGB array, or None if empty."""
    if not img_msg.data or img_msg.height == 0 or img_msg.width == 0:
        return None
    arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.width, 3
    )
    return cv2.resize(arr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def _extract_state(obs) -> np.ndarray:
    """Extract 26-D state vector from an Observation message."""
    cs = obs.controller_state
    tcp = cs.tcp_pose
    vel = cs.tcp_velocity
    joints = list(obs.joint_states.position)[:7]
    return np.array(
        [
            tcp.position.x, tcp.position.y, tcp.position.z,
            tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
            vel.linear.x, vel.linear.y, vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
            *cs.tcp_error,
            *joints,
        ],
        dtype=np.float32,
    )


def _extract_action(obs) -> np.ndarray:
    """Extract 6-D cartesian-velocity action from an Observation message."""
    vel = obs.controller_state.tcp_velocity
    return np.array(
        [vel.linear.x, vel.linear.y, vel.linear.z,
         vel.angular.x, vel.angular.y, vel.angular.z],
        dtype=np.float32,
    )


class RecordCheatCode(CheatCode):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        self.get_logger().info("RecordCheatCode ready — will record to " + str(SAVE_DIR))

    # ── recording helpers ────────────────────────────────────────────────────

    def _record(self, get_observation, states, actions, imgs_c, imgs_l, imgs_r) -> bool:
        """Call get_observation() once, append to buffers. Returns False if images empty."""
        obs = get_observation()
        c = _ros_img_to_numpy(obs.center_image)
        l = _ros_img_to_numpy(obs.left_image)
        r = _ros_img_to_numpy(obs.right_image)
        if c is None or l is None or r is None:
            return False   # camera not yet streaming — skip this step
        states.append(_extract_state(obs))
        actions.append(_extract_action(obs))
        imgs_c.append(c)
        imgs_l.append(l)
        imgs_r.append(r)
        return True

    def _save_episode(self, states, actions, imgs_c, imgs_l, imgs_r) -> Path:
        """Compress and save episode arrays; write path to latest.txt."""
        ts = int(time.time() * 1000)
        save_path = SAVE_DIR / f"ep_{ts}.npz"
        np.savez_compressed(
            str(save_path),
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            center=np.array(imgs_c, dtype=np.uint8),
            left=np.array(imgs_l, dtype=np.uint8),
            right=np.array(imgs_r, dtype=np.uint8),
            success=np.array([True]),
        )
        (SAVE_DIR / "latest.txt").write_text(str(save_path))
        self.get_logger().info(
            f"Episode saved: {save_path}  ({len(states)} steps, "
            f"center shape {np.array(imgs_c).shape})"
        )
        return save_path

    # ── main policy logic (mirrors CheatCode + observation recording) ────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info("RecordCheatCode.insert_cable()")
        self._task = task

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame, timeout_sec=60.0):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time()
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False

        port_transform = port_tf_stamped.transform
        z_offset = 0.2

        states, actions, imgs_c, imgs_l, imgs_r = [], [], [], [], []

        # ── approach phase (0 → 5 s) ────────────────────────────────────────
        for t in range(100):
            frac = t / 100.0
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=frac,
                        position_fraction=frac,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"Approach TF fail: {ex}")
            self._record(get_observation, states, actions, imgs_c, imgs_l, imgs_r)
            self.sleep_for(0.05)

        # ── insertion phase ──────────────────────────────────────────────────
        while z_offset >= -0.015:
            z_offset -= 0.0005
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(port_transform, z_offset=z_offset),
                )
            except TransformException as ex:
                self.get_logger().warn(f"Insertion TF fail: {ex}")
            self._record(get_observation, states, actions, imgs_c, imgs_l, imgs_r)
            self.sleep_for(0.05)

        # ── save ─────────────────────────────────────────────────────────────
        self._save_episode(states, actions, imgs_c, imgs_l, imgs_r)

        self.get_logger().info("Stabilising for 5 s …")
        self.sleep_for(5.0)
        self.get_logger().info("RecordCheatCode.insert_cable() done")
        return True
