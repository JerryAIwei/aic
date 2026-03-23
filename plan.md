# Training Plan: Cable Insertion Policy

## sugestion
- build a tool to record every policy run as a video to make debug on cloud easier

## Goal

Train a policy that achieves high scores across all three qualification trials:
- **Trial 1 & 2**: Insert `SFP_MODULE` plug into a randomized NIC card port
- **Trial 3**: Insert `SC_PLUG` into a randomized SC port
- **Max score per trial**: 100 pts (75 insertion + 24 performance + 1 validity)

Key constraints: robot starts a few cm from target, task board pose is randomized, grasp has ~2mm/0.04rad deviation.

---

## Recommended Approach: ACT via Imitation Learning

The existing codebase already includes:
- `RunACT.py` — a working ACT inference policy (LeRobot's ACT model)
- `lerobot-record` — teleoperation + dataset recording pipeline
- `lerobot-train` — training pipeline
- `lerobot-teleoperate` — keyboard/SpaceMouse teleoperation

This makes **ACT (Action Chunking with Transformers)** the lowest-friction path to a working policy.

---

## Phase 0: Baseline Verification

**Goal**: Confirm the simulation and tooling work end-to-end before any training.

- [ ] Launch simulation in `dev` mode and verify all sensors are visible
  ```bash
  ros2 launch aic_bringup aic_gz_bringup.launch.py ground_truth:=true spawn_task_board:=true
  ```
- [ ] Run `CheatCode` policy (ground truth) to confirm a perfect run is achievable and understand the insertion motion
  ```bash
  ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode
  ```
- [ ] Run `RunACT` with the pretrained HuggingFace checkpoint (`grkw/aic_act_policy`) to establish a baseline score
- [ ] Record what the `CheatCode` policy's TCP trajectory looks like (use PlotJuggler or `ros2 bag record`) — this is the "oracle" motion to imitate

---

## Phase 1: Data Collection

**Goal**: Collect ~50–200 high-quality teleoperation demonstrations per plug type.

### 1.1 Set Up Teleoperation

Use LeRobot's keyboard or SpaceMouse teleoperation in Cartesian space:
```bash
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=gripper/tcp \
  --display_data=true
```

Prefer `gripper/tcp` frame for local insertion control (easier to align with port).

### 1.2 Record Dataset

```bash
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=gripper/tcp \
  --dataset.repo_id=local/aic_sfp_insertion \
  --dataset.single_task="Insert SFP module into NIC port" \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true
```

### 1.3 Data Collection Strategy

**Diversity requirements** (essential for generalization across randomized trials):
- Vary task board position and yaw (match the randomization range from the engine config)
- Vary which NIC rail the card is on (`nic_rail_0` through `nic_rail_4`)
- Vary NIC card translation offset along the rail
- Collect both SFP (trials 1/2) and SC plug (trial 3) demonstrations in separate datasets

**Quality bar per episode**:
- Full successful insertion
- Smooth motion (avoid jerky inputs — penalized up to -6 pts)
- Complete in <30s ideally (scored up to 12 pts for ≤5s)
- No collisions with board enclosure

**Target**: 100+ episodes SFP, 50+ episodes SC.

**Tip**: Use the `CheatCode` policy's motion as a reference for what a good trajectory looks like. Try to replicate: approach from current pose → align → gentle push.

---

## Phase 2: Policy Training

**Goal**: Train ACT on collected demonstrations.

### 2.1 Train ACT Policy

```bash
pixi run lerobot-train \
  --dataset.repo_id=local/aic_sfp_insertion \
  --policy.type=act \
  --output_dir=outputs/train/act_sfp \
  --job_name=act_sfp_insertion \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=local/act_sfp_policy
```

### 2.2 Key Hyperparameters to Consider

ACT defaults work well for short-horizon manipulation. Tune if needed:
- `chunk_size` (action chunk length): default 100 — reduce to 20–50 for faster-reacting policy
- `n_action_steps`: how many steps to execute per chunk before re-querying
- `dim_feedforward`, `n_heads`, `n_encoder_layers`: model capacity

### 2.3 Observation Space

The existing `RunACT.py` uses a **26-dim state vector**:
- TCP position (3) + orientation quaternion (4) + linear velocity (3) + angular velocity (3) + TCP error (6) + joint positions (7)

And **3 camera images** (left, center, right) resized to 25% of 1152×1024.

**Action space**: 7-dim Cartesian velocity twist (3 linear + 3 angular + 1 implicit).

This matches the existing checkpoint — keep the same structure if fine-tuning.

### 2.4 Training with Multiple Simulators (Domain Randomization)

The competition explicitly encourages training across MuJoCo / Isaac Sim for sim-to-sim transfer. If Gazebo-only training shows poor generalization:
- Use `aic_mujoco` or `aic_isaac` to collect additional diverse data
- Mix datasets from different simulators during training

---

## Phase 3: Evaluation & Iteration

**Goal**: Close the loop between training and scoring.

### 3.1 Local Evaluation

Run the full engine evaluation to get a numerical score:
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=false \
  start_aic_engine:=true
```

Run your trained policy:
```bash
ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.RunACT
```
(or your custom policy class)

### 3.2 Interpreting Scores

| Score breakdown | What to fix |
|---|---|
| Tier 3 = 0, proximity score low | Policy not reaching the port — improve demonstration diversity or coverage |
| Tier 3 partial (38–50) | Policy reaching but not fully inserting — tune final push motion, reduce stiffness |
| Tier 2 smoothness low | Jerky policy output — increase action chunk size, add temporal smoothing |
| Tier 2 duration low | Too slow — reduce inference latency, increase control rate |
| Force penalty (-12) | Too much force during insertion — tune `target_stiffness` and `wrench_feedback_gains_at_tip` in `RunACT.py` |
| Off-limit contact (-24) | Colliding with enclosure — add safety margin to approach trajectory |

### 3.3 Iteration Loop

```
Evaluate → Identify weakest tier → Add targeted demos or adjust policy → Retrain → Repeat
```

Specific iteration strategies:
- **Low proximity score**: Add demos with more varied starting positions
- **Fails SC trial**: Train a separate SC model or a multi-task model with task conditioning
- **Inconsistent insertion**: Collect more demos near the insertion point with slower, more precise motion

---

## Phase 4: Policy Integration

**Goal**: Package your trained policy as a clean submission.

### 4.1 Update RunACT or Create Your Own Policy Class

Modify `RunACT.py` (or create a new class) to point to your trained checkpoint:
```python
repo_id = "your-hf-username/your-act-policy"  # or a local path
```

Update `set_cartesian_twist_target()` stiffness/damping values based on what worked during evaluation.

### 4.2 Handle Multiple Plug Types

If training separate SFP and SC models, use the `task` parameter in `insert_cable()` to dispatch:
```python
def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
    plug_type = task.plug_type  # inspect Task.msg for the actual field name
    if plug_type == "SFP_MODULE":
        self._run_sfp_policy(...)
    else:
        self._run_sc_policy(...)
```

### 4.3 Build Submission Container

```bash
docker build -f docker/aic_model/Dockerfile -t my-aic-policy:latest .
```

Test the container locally before submitting.

---

## Decision Points

| Decision | Options | Recommendation |
|---|---|---|
| Policy architecture | ACT, Diffusion Policy, BC | Start with ACT (already integrated) |
| Control space | Cartesian velocity, Cartesian pose, joint | Cartesian velocity (as in `RunACT.py`) |
| One model vs. two | Single multi-task, separate SFP/SC models | Start with one; split if SC generalization fails |
| Data source | Gazebo only, multi-sim | Gazebo first; add MuJoCo if overfitting |
| Fine-tune vs. train from scratch | Fine-tune `grkw/aic_act_policy` checkpoint | Fine-tune first (faster convergence) |

---

## File Reference

| File | Purpose |
|---|---|
| `aic_example_policies/aic_example_policies/ros/RunACT.py` | ACT inference policy to extend or copy |
| `aic_example_policies/aic_example_policies/ros/CheatCode.py` | Ground truth oracle for understanding target motion |
| `aic_utils/lerobot_robot_aic/README.md` | Teleoperation and recording commands |
| `aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_robot.py` | Camera/joint config for LeRobot |
| `aic_engine/config/sample_config.yaml` | Grasp pose offsets (SFP: z=0.04245, SC: z=0.04045) |
| `docs/scoring.md` | Full scoring breakdown |
| `docs/qualification_phase.md` | Trial descriptions and randomization ranges |
| `docker/aic_model/Dockerfile` | Submission container definition |
