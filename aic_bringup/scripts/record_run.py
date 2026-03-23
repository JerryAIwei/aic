#!/usr/bin/env python3
"""
record_run.py — Records every InsertCable trial as an MP4 video.

Subscribes to the three wrist cameras and monitors the /insert_cable action
status. Automatically starts recording when a trial begins (goal EXECUTING)
and saves a side-by-side composite video to $AIC_RESULTS_DIR when it ends.

Output filename: <ISO-timestamp>_<success|failed>.mp4
Output location: $AIC_RESULTS_DIR  (default: ~/aic_results/videos/)

Usage (run in a separate terminal while sim + policy are running):
    pixi run python aic_bringup/scripts/record_run.py
"""

import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SCALE = 0.25  # Downscale factor — matches RunACT.py image_scaling
FPS = 20.0  # Camera publish rate
CAMERA_TOPICS = [
    "/left_camera/image",
    "/center_camera/image",
    "/right_camera/image",
]
ACTION_STATUS_TOPIC = "/insert_cable/_action/status"
TERMINAL_STATUSES = frozenset(
    [GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED]
)


def _results_dir() -> Path:
    env = os.environ.get("AIC_RESULTS_DIR", "")
    base = Path(env) if env else Path.home() / "aic_results"
    videos = base / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    return videos


def _ros_image_to_bgr(msg: Image) -> np.ndarray:
    """Convert a ROS sensor_msgs/Image to an OpenCV BGR numpy array."""
    channels = len(msg.data) // (msg.height * msg.width)
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding.lower() in ("rgb8", "rgb"):
        return cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2BGR)
    return data[:, :, :3].copy()  # assume bgr8


# --------------------------------------------------------------------------- #
# Recorder node
# --------------------------------------------------------------------------- #
class RunRecorder(Node):
    def __init__(self):
        super().__init__("run_recorder")

        self._lock = threading.Lock()
        self._latest_frames: dict[str, Image | None] = {t: None for t in CAMERA_TOPICS}
        self._out_dir = _results_dir()

        # Recording state
        self._recording = False
        self._writer: cv2.VideoWriter | None = None
        self._trial_start_time: datetime | None = None
        self._current_goal: bytes | None = None  # UUID bytes of goal being recorded
        self._known_goals: dict[bytes, int] = {}  # UUID -> last seen status

        self.get_logger().info(f"Videos will be saved to: {self._out_dir}")

        # Subscribe to all three cameras
        for topic in CAMERA_TOPICS:
            self.create_subscription(Image, topic, self._make_camera_cb(topic), 5)

        # Monitor action status
        self.create_subscription(
            GoalStatusArray, ACTION_STATUS_TOPIC, self._status_cb, 10
        )

        # Timer writes composite frames at FPS when recording
        self.create_timer(1.0 / FPS, self._write_frame)

    # ----------------------------------------------------------------------- #
    # Camera callbacks — just cache the latest frame per topic
    # ----------------------------------------------------------------------- #
    def _make_camera_cb(self, topic: str):
        def cb(msg: Image):
            with self._lock:
                self._latest_frames[topic] = msg

        return cb

    # ----------------------------------------------------------------------- #
    # Action status callback
    # ----------------------------------------------------------------------- #
    def _status_cb(self, msg: GoalStatusArray):
        current: dict[bytes, int] = {
            s.goal_info.goal_id.uuid.tobytes(): s.status for s in msg.status_list
        }

        for uid, status in current.items():
            prev = self._known_goals.get(uid)

            # New goal just started executing → begin recording
            if status == GoalStatus.STATUS_EXECUTING and prev != GoalStatus.STATUS_EXECUTING:
                if not self._recording:
                    self._start_recording(uid)

            # Goal we're recording just reached a terminal state
            if uid == self._current_goal and status in TERMINAL_STATUSES:
                if self._recording:
                    self._stop_recording(success=(status == GoalStatus.STATUS_SUCCEEDED))

        # Goal we're recording has disappeared from the status list entirely
        if self._recording and self._current_goal and self._current_goal not in current:
            self._stop_recording(success=False)

        self._known_goals = current

    # ----------------------------------------------------------------------- #
    # Recording lifecycle
    # ----------------------------------------------------------------------- #
    def _start_recording(self, goal_uid: bytes):
        self._recording = True
        self._current_goal = goal_uid
        self._trial_start_time = datetime.now()
        self.get_logger().info("Trial started — recording...")

    def _stop_recording(self, success: bool):
        self._recording = False
        outcome = "success" if success else "failed"
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
        self.get_logger().info(f"Trial ended ({outcome}) — video saved to {self._out_dir}")
        # Tag the filename retroactively by renaming
        ts = self._trial_start_time.strftime("%Y-%m-%dT%H-%M-%S") if self._trial_start_time else "unknown"
        pending = self._out_dir / f"{ts}_trial_pending.mp4"
        final = self._out_dir / f"{ts}_{outcome}.mp4"
        if pending.exists():
            pending.rename(final)
            self.get_logger().info(f"Saved: {final.name}")
        self._current_goal = None

    # ----------------------------------------------------------------------- #
    # Frame writer (called at FPS by timer)
    # ----------------------------------------------------------------------- #
    def _write_frame(self):
        if not self._recording:
            return

        with self._lock:
            raw_frames = [self._latest_frames[t] for t in CAMERA_TOPICS]

        if any(f is None for f in raw_frames):
            return  # cameras not ready yet

        bgr_frames = [_ros_image_to_bgr(f) for f in raw_frames]

        scaled = []
        for f in bgr_frames:
            h, w = f.shape[:2]
            scaled.append(
                cv2.resize(
                    f,
                    (int(w * SCALE), int(h * SCALE)),
                    interpolation=cv2.INTER_AREA,
                )
            )

        composite = np.concatenate(scaled, axis=1)  # left | center | right
        frame_h, frame_w = composite.shape[:2]

        # Overlay: timestamp + recording indicator
        elapsed = (datetime.now() - self._trial_start_time).total_seconds()
        cv2.putText(
            composite,
            f"REC  {elapsed:.1f}s",
            (10, frame_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        with self._lock:
            if self._writer is None and self._trial_start_time is not None:
                ts = self._trial_start_time.strftime("%Y-%m-%dT%H-%M-%S")
                out_path = str(self._out_dir / f"{ts}_trial_pending.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(out_path, fourcc, FPS, (frame_w, frame_h))
                self.get_logger().info(f"Writing to {out_path}")
            if self._writer is not None:
                self._writer.write(composite)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(args=None):
    try:
        with rclpy.init(args=args):
            node = RunRecorder()
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Flush any in-progress recording on shutdown
        rclpy.try_shutdown()


if __name__ == "__main__":
    main(sys.argv)
