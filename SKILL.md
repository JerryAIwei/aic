# SKILL.md — Running AIC Simulation with GPU Inside Docker

This document captures everything learned about running the AIC Gazebo simulation
with NVIDIA GPU acceleration from within a Docker container, without distrobox or
nested Docker.

---

## Environment

- **We are inside Docker**: `/workspace/aic` is mounted into a container that has
  NVIDIA driver 550.127.05 libs injected by the Docker runtime.
- **Pre-extracted rootfs**: The `aic_eval` Docker image is pre-extracted at
  `/opt/aic_rootfs` — a full Ubuntu 24.04 filesystem with ROS Kilted and the AIC
  workspace at `/opt/aic_rootfs/ws_aic`.
- **GPU**: NVIDIA RTX 4000 Ada Generation (20 GB), driver 550.127.05, CUDA 12.4.

---

## The Problem

The cloud evaluation container (`aic_eval`) expects to be run with `distrobox --nvidia`
or `docker run --gpus all`, which injects NVIDIA device nodes and library paths at
runtime. Inside an already-Docker-ized development environment, nested Docker /
distrobox is unavailable (no `CAP_SYS_ADMIN` for bind-mounts, no user namespaces).

Without GPU access, Gazebo uses Mesa software rendering at ~0.6% real-time factor
(simulation takes 160× longer than real time). This makes policy evaluation impractical.

---

## The Solution

**Run the rootfs ROS workspace directly from our container** — no chroot needed.

Both environments are Ubuntu 24.04, so the rootfs binaries run fine in our container.
The key steps:

### 1. Create path symlinks

The rootfs workspace `setup.bash` hardcodes `/opt/ros/kilted` and `/ws_aic`:

```bash
ln -sfn /opt/aic_rootfs/opt/ros /opt/ros
ln -sfn /opt/aic_rootfs/ws_aic  /ws_aic
ln -sfn /opt/aic_rootfs/usr/lib/x86_64-linux-gnu/OGRE-2.3 \
        /usr/lib/x86_64-linux-gnu/OGRE-2.3
```

The OGRE-2.3 symlink is required because the compiled-in plugin search path in
`libgz-rendering-ogre2.so` is `/usr/lib/x86_64-linux-gnu/OGRE-2.3`.

### 2. Set LD_LIBRARY_PATH with NVIDIA libs first

```bash
ROOTFS=/opt/aic_rootfs
NVIDIA_LIBS="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"       # GPU EGL/GL
ROOTFS_LIBS="$ROOTFS/usr/lib/x86_64-linux-gnu:$ROOTFS/lib/x86_64-linux-gnu:$ROOTFS/usr/lib"
OGRE_LIBS="$ROOTFS/usr/lib/x86_64-linux-gnu/OGRE-2.3"
GZ_VENDOR_LIBS="$ROOTFS/opt/ros/kilted/opt/gz_ogre_next_vendor/lib"

export LD_LIBRARY_PATH="$NVIDIA_LIBS:$ROOTFS_LIBS:$OGRE_LIBS:$GZ_VENDOR_LIBS"
```

NVIDIA libs must come first so `libEGL_nvidia.so.0` is preferred over `libEGL_mesa.so.0`.

### 3. Point EGL to NVIDIA vendor config

```bash
export __EGL_VENDOR_LIBRARY_DIRS="/usr/share/glvnd/egl_vendor.d"
```

This directory has both `10_nvidia.json` (NVIDIA EGL) and `50_mesa.json` (Mesa EGL).
With NVIDIA libs in `LD_LIBRARY_PATH`, the NVIDIA vendor is selected.

### 4. Expose the Ogre plugin directory

The Ogre render system plugin (`RenderSystem_GL3Plus.so`) is at:
`/usr/lib/x86_64-linux-gnu/OGRE-2.3/OGRE/` (subdirectory).
Gazebo's `Ogre2RenderEngine::LoadPlugins()` looks in `ogrePaths` entries directly;
set the env var to add the subdirectory:

```bash
export OGRE2_RESOURCE_PATH="/usr/lib/x86_64-linux-gnu/OGRE-2.3/OGRE"
```

### 5. Add rootfs Python dist-packages

Launch files use `empy` (Python template engine) which is only in the rootfs:

```bash
export PYTHONPATH="$ROOTFS/usr/lib/python3/dist-packages:$ROOTFS/usr/lib/python3.12/dist-packages"
```

### 6. Headless mode (no X11)

Gazebo detects that `$DISPLAY` is empty and automatically switches to EGL headless mode:
```
[warning] Unable to open display: . Trying to run in headless mode.
```

This is the correct behaviour. Do NOT start Xvfb — it uses Mesa software rendering.

### 7. Disable Zenoh shared memory

Zenoh shared memory transport can cause issues in this configuration:

```bash
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
```

---

## Result

| Metric | Before (Mesa/Xvfb) | After (NVIDIA EGL) |
|---|---|---|
| GPU utilization | 0% | 20–21% |
| Real-time factor | ~0.6% | ~100% (real-time) |
| Render backend | Mesa software | NVIDIA EGL headless |

---

## Quick Start

Use the provided startup script:

```bash
# Dev mode (robot + task board + ground truth), GPU accelerated
./start_simulator.sh gpu dev

# Full evaluation mode with a policy
./start_simulator.sh gpu eval --policy aic_example_policies.ros.CheatCode
```

Or set up the environment manually (for custom workflows):

```bash
ROOTFS=/opt/aic_rootfs

# Symlinks (one-time, persist until container restart)
ln -sfn $ROOTFS/opt/ros /opt/ros
ln -sfn $ROOTFS/ws_aic  /ws_aic
ln -sfn $ROOTFS/usr/lib/x86_64-linux-gnu/OGRE-2.3 /usr/lib/x86_64-linux-gnu/OGRE-2.3

# Environment
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$ROOTFS/usr/lib/x86_64-linux-gnu:$ROOTFS/lib/x86_64-linux-gnu:$ROOTFS/usr/lib:$ROOTFS/usr/lib/x86_64-linux-gnu/OGRE-2.3:$ROOTFS/opt/ros/kilted/opt/gz_ogre_next_vendor/lib"
export __EGL_VENDOR_LIBRARY_DIRS="/usr/share/glvnd/egl_vendor.d"
export OGRE2_RESOURCE_PATH="/usr/lib/x86_64-linux-gnu/OGRE-2.3/OGRE"
export PYTHONPATH="$ROOTFS/usr/lib/python3/dist-packages:$ROOTFS/usr/lib/python3.12/dist-packages"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
unset DISPLAY

# Source workspace
source /ws_aic/install/setup.bash

# Start Zenoh router
ros2 run rmw_zenoh_cpp rmw_zenohd &

# Launch simulation
ros2 launch aic_bringup aic_gz_bringup.launch.py ground_truth:=true spawn_task_board:=true
```

---

## Connecting pixi tools to the GPU simulation

The `pixi` environment (used for `lerobot-record`, `record_run.py`, policies) connects
to the same Zenoh router via:

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/127.0.0.1:7447"];transport/shared_memory/enabled=false'
```

This is already in `pixi_env_setup.sh`. Run it before any `pixi run` command:

```bash
source pixi_env_setup.sh
pixi run python aic_bringup/scripts/record_run.py
```

---

## What Doesn't Work

- **Xvfb + chroot approach**: `chroot /opt/aic_rootfs` works for physics but uses Mesa
  software rendering via Xvfb (0.6% RT factor). Abandoned.
- **Docker-in-Docker**: Fails with `overlayfs operation not permitted` inside the container.
- **bind-mount of /dev/nvidia***: Requires `CAP_SYS_ADMIN` which is not available.
- **mknod for device nodes**: Requires `CAP_MKNOD` which is not available.
- **RViz2**: Crashes (`no Qt platform plugin "xcb"`) because there's no X display.
  This is expected and harmless — RViz2 is optional.

---

## NVIDIA Libs Copied Into rootfs

These libs were copied from the container into `/opt/aic_rootfs/` to make them
available if running inside the chroot (not needed for the no-chroot GPU approach):

- `libEGL_nvidia.so.550.127.05` + `.0` symlink
- `libGLX_nvidia.so.550.127.05` + `.0` symlink
- `libnvidia-glsi/eglcore/glcore/glvkspirv/gpucomp/tls/allocator.so.550.127.05`
- `libcuda.so.550.127.05`

And the NVIDIA EGL vendor JSON was copied to `/opt/aic_rootfs/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
