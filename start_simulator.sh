#!/bin/bash
# Start the AIC simulator
# Usage: ./start_simulator.sh [MODE] [OPTIONS]
#
# Modes:
#   basic       - Robot only, no task board (default)
#   dev         - Robot + task board + ground truth (policy development)
#   eval        - Full evaluation with engine + scoring
#   docker      - Run via Docker (matches cloud evaluation environment)
#   gpu         - Run directly with NVIDIA GPU (inside Docker, no distrobox needed)
#
# Examples:
#   ./start_simulator.sh
#   ./start_simulator.sh dev
#   ./start_simulator.sh eval --policy aic_example_policies.ros.WaveArm
#   ./start_simulator.sh gpu
#   ./start_simulator.sh gpu dev
#   ./start_simulator.sh gpu eval --policy aic_example_policies.ros.CheatCode

set -e

MODE="${1:-basic}"
GPU_MODE=false
POLICY="aic_example_policies.ros.WaveArm"
RESULTS_DIR="${AIC_RESULTS_DIR:-$HOME/aic_results}"
WS="${AIC_WS:-$HOME/ws_aic}"
ROOTFS="/opt/aic_rootfs"

# Check for "gpu" as first argument
if [[ "$MODE" == "gpu" ]]; then
  GPU_MODE=true
  MODE="${2:-basic}"
  shift 2 || shift 1 || true
else
  shift || true
fi

# Parse extra flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) POLICY="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --ws) WS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── helpers ──────────────────────────────────────────────────────────────────

setup_env() {
  if [[ ! -f "$WS/install/setup.bash" ]]; then
    echo "ERROR: Built workspace not found at $WS/install/setup.bash"
    echo ""
    echo "Build it first:"
    echo "  cd $WS"
    echo "  GZ_BUILD_FROM_SOURCE=1 colcon build \\"
    echo "    --cmake-args -DCMAKE_BUILD_TYPE=Release \\"
    echo "    --merge-install --symlink-install \\"
    echo "    --packages-ignore lerobot_robot_aic"
    echo ""
    echo "Or if using Pixi (from the aic repo dir):"
    echo "  cd /workspace/aic && pixi install"
    echo ""
    echo "Set AIC_WS or pass --ws <path> to point to a different workspace."
    exit 1
  fi
  source "$WS/install/setup.bash"
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
  export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true;transport/shared_memory/transport_optimization/pool_size=536870912'
}

# GPU mode: run using the pre-extracted rootfs at /opt/aic_rootfs
# Injects NVIDIA GPU libs from the current Docker container into the rootfs
# environment — no distrobox, no nested Docker needed.
setup_gpu_env() {
  if [[ ! -d "$ROOTFS/ws_aic/install" ]]; then
    echo "ERROR: rootfs not found at $ROOTFS"
    echo "The pre-extracted aic_eval rootfs is expected at /opt/aic_rootfs."
    exit 1
  fi

  # Create path symlinks so setup.bash paths (/opt/ros/kilted, /ws_aic) resolve
  # from the current container's root into the rootfs.
  [[ -L /opt/ros ]] || ln -sf "$ROOTFS/opt/ros" /opt/ros
  [[ -L /ws_aic  ]] || ln -sf "$ROOTFS/ws_aic"  /ws_aic

  # NVIDIA GPU libs from the current container come first in LD_LIBRARY_PATH.
  # This ensures libEGL_nvidia.so.0 (GPU) is found before libEGL_mesa.so.0 (CPU).
  NVIDIA_LIBS="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
  ROOTFS_LIBS="$ROOTFS/usr/lib/x86_64-linux-gnu:$ROOTFS/lib/x86_64-linux-gnu:$ROOTFS/usr/lib"
  # OgreNext libs are in a subdirectory that must be explicitly added
  OGRE_LIBS="$ROOTFS/usr/lib/x86_64-linux-gnu/OGRE-2.3"
  GZ_VENDOR_LIBS="$ROOTFS/opt/ros/kilted/opt/gz_ogre_next_vendor/lib"
  export LD_LIBRARY_PATH="$NVIDIA_LIBS:$ROOTFS_LIBS:$OGRE_LIBS:$GZ_VENDOR_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  # OGRE-2.3/OGRE/ contains the render system plugins (RenderSystem_GL3Plus.so)
  # The compiled-in path is /usr/lib/x86_64-linux-gnu/OGRE-2.3, which we symlink below.
  # OGRE2_RESOURCE_PATH adds the plugins subdirectory where Gazebo finds GL3Plus.
  export OGRE2_RESOURCE_PATH="/usr/lib/x86_64-linux-gnu/OGRE-2.3/OGRE"

  # Symlink OGRE-2.3 into the container lib dir (compiled-in path in gz-rendering)
  [[ -L /usr/lib/x86_64-linux-gnu/OGRE-2.3 ]] || \
    ln -sfn "$ROOTFS/usr/lib/x86_64-linux-gnu/OGRE-2.3" /usr/lib/x86_64-linux-gnu/OGRE-2.3

  # Add rootfs Python dist-packages (needed for 'em' module used by launch files)
  export PYTHONPATH="$ROOTFS/usr/lib/python3/dist-packages:$ROOTFS/usr/lib/python3.12/dist-packages${PYTHONPATH:+:$PYTHONPATH}"

  # Point EGL vendor discovery to our container's NVIDIA vendor JSON
  export __EGL_VENDOR_LIBRARY_DIRS="/usr/share/glvnd/egl_vendor.d"

  # Headless EGL — no X11 display needed
  unset DISPLAY

  # Source the rootfs workspace (paths now resolve via symlinks)
  source /ws_aic/install/setup.bash

  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  export ZENOH_ROUTER_CHECK_ATTEMPTS=-1
  # Shared memory disabled: in headless EGL mode the Zenoh SHM transport
  # can cause issues; TCP transport is reliable.
  export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'

  echo "==> GPU mode: NVIDIA EGL headless (no X11)"
  echo "    EGL vendor: $(python3 -c "
import ctypes
egl = ctypes.CDLL('libEGL.so.1')
egl.eglGetDisplay.restype = ctypes.c_void_p
egl.eglQueryString.restype = ctypes.c_char_p
disp = egl.eglGetDisplay(0)
m, n = ctypes.c_int(), ctypes.c_int()
egl.eglInitialize(disp, ctypes.byref(m), ctypes.byref(n))
v = egl.eglQueryString(disp, 0x3053)  # EGL_VENDOR
print(v.decode() if v else 'unknown')
" 2>/dev/null || echo 'check failed')"
}

cleanup() {
  echo ""
  echo "Shutting down..."
  # Kill all background jobs spawned by this script
  jobs -p | xargs -r kill 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── modes ────────────────────────────────────────────────────────────────────

run_basic() {
  echo "==> Starting basic simulation (robot only)..."
  if $GPU_MODE; then setup_gpu_env; else setup_env; fi
  ros2 launch aic_bringup aic_gz_bringup.launch.py
}

run_dev() {
  echo "==> Starting development simulation (robot + task board + ground truth)..."
  if $GPU_MODE; then setup_gpu_env; else setup_env; fi
  ros2 launch aic_bringup aic_gz_bringup.launch.py \
    ground_truth:=true \
    spawn_task_board:=true
}

run_eval() {
  echo "==> Starting full evaluation (engine + scoring)..."
  echo "    Policy:      $POLICY"
  echo "    Results dir: $RESULTS_DIR"
  if $GPU_MODE; then setup_gpu_env; else setup_env; fi

  # Terminal 1: Zenoh router
  echo "--> Launching Zenoh router..."
  ros2 run rmw_zenoh_cpp rmw_zenohd &
  ZENOH_PID=$!
  sleep 2

  # Terminal 2: Policy model
  echo "--> Launching aic_model with policy: $POLICY"
  ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true -p policy:="$POLICY" &
  MODEL_PID=$!
  sleep 1

  # Terminal 3: Simulation + Engine (foreground)
  echo "--> Launching simulation + engine (foreground)..."
  AIC_RESULTS_DIR="$RESULTS_DIR" \
  ros2 launch aic_bringup aic_gz_bringup.launch.py \
    ground_truth:=false \
    start_aic_engine:=true
}

run_docker() {
  echo "==> Starting simulator via Docker..."

  if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is not installed."
    exit 1
  fi

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # Check for NVIDIA GPU
  NVIDIA_FLAG=""
  if command -v nvidia-smi &>/dev/null; then
    echo "    NVIDIA GPU detected — enabling GPU support."
    NVIDIA_FLAG="--nvidia"
  else
    echo "    No NVIDIA GPU detected — running without GPU acceleration."
  fi

  export DBX_CONTAINER_MANAGER=docker

  echo "--> Pulling latest aic_eval image..."
  docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest

  echo "--> Creating distrobox container..."
  distrobox create -r $NVIDIA_FLAG \
    -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval 2>/dev/null || true

  echo "--> Entering container and starting simulator..."
  distrobox enter -r aic_eval -- /entrypoint.sh \
    ground_truth:=false \
    start_aic_engine:=true
}

# ── dispatch ─────────────────────────────────────────────────────────────────

case "$MODE" in
  basic)  run_basic  ;;
  dev)    run_dev    ;;
  eval)   run_eval   ;;
  docker) run_docker ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Valid modes: basic, dev, eval, docker"
    echo "Prefix with 'gpu' for GPU-accelerated headless rendering:"
    echo "  ./start_simulator.sh gpu [mode] [options]"
    exit 1
    ;;
esac
