#!/bin/bash
# Stop all AIC-related processes
# Usage: ./stop_simulator.sh

echo "Stopping AIC processes..."

# ROS 2 nodes
pkill -f "ros2 launch aic_bringup" 2>/dev/null && echo "  killed: aic_bringup launch" || true
pkill -f "ros2 run aic_model"      2>/dev/null && echo "  killed: aic_model"          || true
pkill -f "ros2 run rmw_zenoh_cpp"  2>/dev/null && echo "  killed: zenoh router"       || true
pkill -f "rmw_zenohd"              2>/dev/null && echo "  killed: rmw_zenohd"         || true

# Gazebo
pkill -f "gz sim"                  2>/dev/null && echo "  killed: gz sim"             || true
pkill -f "gzserver"                2>/dev/null && echo "  killed: gzserver"           || true
pkill -f "gzclient"                2>/dev/null && echo "  killed: gzclient"           || true

# ROS 2 daemon (clears stale node/topic cache)
ros2 daemon stop 2>/dev/null && echo "  stopped: ros2 daemon" || true

# Distrobox / Docker (if running aic_eval container)
if command -v distrobox &>/dev/null; then
  distrobox stop aic_eval 2>/dev/null && echo "  stopped: distrobox aic_eval" || true
fi

echo "Done."
