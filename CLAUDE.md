# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The **AI for Industry Challenge (AIC)** is a robotics competition toolkit for solving cable insertion tasks. The system is split into:

- **Evaluation infrastructure** (provided): `aic_engine`, `aic_bringup`, `aic_controller`, `aic_adapter`, `aic_scoring`, `aic_gazebo`, `aic_description`, `aic_assets`
- **Participant framework**: `aic_model` (lifecycle node), `aic_example_policies` (reference implementations)
- **Utilities**: `aic_utils` (LeRobot/MuJoCo/Isaac/teleoperation integrations)

**ROS 2 Kilted Kaiju** is the mandated distribution. **rmw_zenoh_cpp** is the DDS middleware.

## Environment Setup

This project uses **Pixi** for environment management:

```bash
pixi install --locked       # Install all dependencies
source pixi_env_setup.sh    # Activate environment (sets RMW_IMPLEMENTATION, ZENOH config, etc.)
```

## Build

```bash
# Build all ROS 2 packages
pixi run colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Or directly (requires activated pixi environment)
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## Lint & Format

```bash
# Python formatting
black --check .          # Check
black .                  # Fix

# Python import sorting (only aic_utils/lerobot_robot_aic)
isort -c aic_utils/lerobot_robot_aic   # Check
isort aic_utils/lerobot_robot_aic      # Fix

# Python type checking
pyright

# C++ formatting (Google style via .clang-format)
clang-format-19 --dry-run src/**/*.{cpp,hpp,h}   # Check
clang-format-19 -i src/**/*.{cpp,hpp,h}           # Fix
```

CI enforces: clang-format v19 for C++, black for Python, isort + pyright for `aic_utils`.

## Tests

Tests are Python scripts in `aic_model/test/` and `aic_example_policies/test/`. Run with:

```bash
pytest aic_model/test/
pytest aic_example_policies/test/
```

Integration testing requires launching the simulator (see below).

## Running the Simulation

```bash
# Full simulation with ground truth (development mode)
ros2 launch aic_bringup aic_gz_bringup.launch.py ground_truth:=true spawn_task_board:=true

# Run a policy node (separate terminal, after sourcing pixi env)
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode

# Full evaluation with engine
ros2 launch aic_bringup aic_gz_bringup.launch.py ground_truth:=false start_aic_engine:=true
```

## Architecture: Key Design Patterns

### Policy Framework (`aic_model/`)
- `aic_model.py` is a **ROS 2 Lifecycle Node** that dynamically imports a policy module at runtime via the `policy` parameter (e.g., `my_package.MyPolicy`).
- `policy.py` defines the abstract `Policy` base class. Participants implement `insert_cable()` using three callbacks:
  - `move_robot(MotionUpdate)` — send Cartesian/joint commands
  - `get_observation()` → `Observation` — camera images, wrench, joint states
  - `send_feedback(str)` — report progress to the engine
- All nodes use `use_sim_time:=true`.

### Trial Orchestration (`aic_engine/`)
- C++ state machine: Uninitialized → Initialized → Running → Completed
- Validates lifecycle state of `aic_model`, spawns task board, sends `InsertCable` action goal, monitors completion, collects scoring data.

### Controller (`aic_controller/`)
- ROS 2 `ros2_control` controller implementing **Cartesian impedance** and **joint impedance** control.
- Key actions: `CartesianImpedanceAction`, `JointImpedanceAction`, `GravityCompensationAction`.
- Uses KDL, Sophus, Eigen3.

### ROS 2 Interfaces (`aic_interfaces/`)
Four sub-packages define the API:
- `aic_control_interfaces` — `MotionUpdate.msg`, `JointMotionUpdate.msg`, `ChangeTargetMode.srv`, `ResetJoints.srv`
- `aic_model_interfaces` — `Observation.msg` (3 cameras, wrist wrench, joint states, controller state)
- `aic_task_interfaces` — `InsertCable.action`, `Task.msg`
- `aic_engine_interfaces` — engine-specific services

### Middleware Configuration
Set in `pixi_env_setup.sh`:
```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/127.0.0.1:7447"];transport/shared_memory/enabled=false'
```
Docker containers use the same Zenoh TCP config (no shared memory).

## Submission

Participants package policies as Docker containers:
```bash
docker build -f docker/aic_model/Dockerfile -t my-policy:latest .
```
Base image: `ros:kilted-ros-core`. The container runs `aic_model` and connects to the organizer's evaluation infrastructure via Zenoh.

## External Dependencies

`aic.repos` pins 40+ external ROS/Gazebo repositories (specific commits). Run `vcs import` to fetch them for local builds that require non-Pixi dependencies.

Results are saved to `$AIC_RESULTS_DIR` (default: `$HOME/aic_results`).
