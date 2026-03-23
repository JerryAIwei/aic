#!/usr/bin/env python3
"""
Unit tests for record_run.py — no live ROS required.

Tests the image conversion helper, composite frame assembly, and the
goal-tracking state machine using mock ROS messages.
"""

import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest


# ── helpers imported directly (no rclpy.init needed) ────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from record_run import _ros_image_to_bgr, SCALE, CAMERA_TOPICS


# ── fixtures ─────────────────────────────────────────────────────────────────
def _make_image_msg(h=100, w=100, encoding="rgb8"):
    """Return a minimal mock sensor_msgs/Image."""
    msg = MagicMock()
    msg.height = h
    msg.width = w
    msg.encoding = encoding
    # Red frame in RGB
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # R channel
    msg.data = frame.tobytes()
    return msg


def _make_status_msg(uid: bytes, status: int):
    """Return a minimal mock action_msgs/GoalStatusArray with one entry."""
    from action_msgs.msg import GoalStatus, GoalStatusArray

    gs = GoalStatus()
    gs.goal_info.goal_id.uuid = list(uid)
    gs.status = status
    arr = GoalStatusArray()
    arr.status_list = [gs]
    return arr


# ── image conversion ─────────────────────────────────────────────────────────
class TestRosImageToBgr:
    def test_rgb8_converts_to_bgr(self):
        msg = _make_image_msg(encoding="rgb8")
        bgr = _ros_image_to_bgr(msg)
        # Red pixel in RGB (R=255,G=0,B=0) → BGR (B=0,G=0,R=255)
        assert bgr[0, 0, 0] == 0    # B channel
        assert bgr[0, 0, 2] == 255  # R channel

    def test_bgr8_passthrough(self):
        msg = _make_image_msg(encoding="bgr8")
        bgr = _ros_image_to_bgr(msg)
        # No conversion — original R channel still in position 0
        assert bgr[0, 0, 0] == 255

    def test_output_shape(self):
        msg = _make_image_msg(h=48, w=64, encoding="rgb8")
        bgr = _ros_image_to_bgr(msg)
        assert bgr.shape == (48, 64, 3)


# ── composite assembly ───────────────────────────────────────────────────────
class TestCompositeFrame:
    def test_three_cameras_side_by_side(self):
        h, w = 40, 60
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(3)]
        composite = np.concatenate(frames, axis=1)
        assert composite.shape == (h, w * 3, 3)

    def test_scale_reduces_size(self):
        h, w = 1024, 1152
        scaled_w = int(w * SCALE)
        scaled_h = int(h * SCALE)
        assert scaled_w == 288
        assert scaled_h == 256


# ── recorder state machine ───────────────────────────────────────────────────
class TestRecorderStateMachine:
    """Test start/stop logic using mock GoalStatusArray messages."""

    def _make_recorder(self, tmp_path):
        """Instantiate RunRecorder with rclpy patched out."""
        with patch("rclpy.init"), patch("rclpy.spin"), patch("rclpy.try_shutdown"):
            import record_run

            with patch.object(record_run, "_results_dir", return_value=tmp_path):
                with patch("rclpy.node.Node.__init__", return_value=None):
                    rec = record_run.RunRecorder.__new__(record_run.RunRecorder)
                    # Manually init state (bypassing Node.__init__)
                    rec._lock = threading.Lock()
                    rec._latest_frames = {t: None for t in CAMERA_TOPICS}
                    rec._out_dir = tmp_path
                    rec._recording = False
                    rec._writer = None
                    rec._trial_start_time = None
                    rec._current_goal = None
                    rec._known_goals = {}
                    # Mock logger
                    rec.get_logger = lambda: MagicMock()
                    return rec

    def test_starts_recording_on_executing(self, tmp_path):
        rec = self._make_recorder(tmp_path)
        uid = bytes(range(16))
        from action_msgs.msg import GoalStatus

        msg = _make_status_msg(uid, GoalStatus.STATUS_EXECUTING)
        rec._status_cb(msg)
        assert rec._recording is True
        assert rec._current_goal == uid

    def test_stops_recording_on_succeeded(self, tmp_path):
        rec = self._make_recorder(tmp_path)
        uid = bytes(range(16))
        from action_msgs.msg import GoalStatus

        # Start
        rec._status_cb(_make_status_msg(uid, GoalStatus.STATUS_EXECUTING))
        assert rec._recording is True

        # Succeed — writer is None so no file rename needed
        rec._stop_recording = MagicMock(wraps=rec._stop_recording)
        rec._status_cb(_make_status_msg(uid, GoalStatus.STATUS_SUCCEEDED))
        rec._stop_recording.assert_called_once_with(success=True)
        assert rec._recording is False

    def test_stops_recording_on_aborted(self, tmp_path):
        rec = self._make_recorder(tmp_path)
        uid = bytes(range(16))
        from action_msgs.msg import GoalStatus

        rec._status_cb(_make_status_msg(uid, GoalStatus.STATUS_EXECUTING))
        rec._status_cb(_make_status_msg(uid, GoalStatus.STATUS_ABORTED))
        assert rec._recording is False

    def test_goal_disappear_stops_recording(self, tmp_path):
        rec = self._make_recorder(tmp_path)
        uid = bytes(range(16))
        from action_msgs.msg import GoalStatus, GoalStatusArray

        rec._status_cb(_make_status_msg(uid, GoalStatus.STATUS_EXECUTING))
        assert rec._recording is True

        # Send empty status array (goal gone)
        rec._status_cb(GoalStatusArray())
        assert rec._recording is False

    def test_does_not_double_start(self, tmp_path):
        rec = self._make_recorder(tmp_path)
        uid = bytes(range(16))
        from action_msgs.msg import GoalStatus

        rec._start_recording = MagicMock(wraps=rec._start_recording)
        rec._status_cb(_make_status_msg(uid, GoalStatus.STATUS_EXECUTING))
        rec._status_cb(_make_status_msg(uid, GoalStatus.STATUS_EXECUTING))
        assert rec._start_recording.call_count == 1


# ── video writer integration ──────────────────────────────────────────────────
class TestVideoWriter:
    def test_write_frame_creates_file(self, tmp_path):
        """_write_frame should lazily create a VideoWriter and write frames."""
        import record_run
        from datetime import datetime

        rec = record_run.RunRecorder.__new__(record_run.RunRecorder)
        rec._lock = threading.Lock()
        rec._out_dir = tmp_path
        rec._recording = True
        rec._writer = None
        rec._trial_start_time = datetime(2026, 3, 23, 12, 0, 0)
        rec.get_logger = lambda: MagicMock()

        # Provide mock camera frames (small, 3-channel BGR)
        h, w = 40, 60
        fake_img = _make_image_msg(h=h, w=w, encoding="bgr8")
        rec._latest_frames = {t: fake_img for t in CAMERA_TOPICS}

        # Patch Node.__init__ to avoid real ROS
        with patch("rclpy.node.Node.__init__", return_value=None):
            rec._write_frame()

        assert rec._writer is not None
        rec._writer.release()

        # Pending file should exist
        pending = list(tmp_path.glob("*_trial_pending.mp4"))
        assert len(pending) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
