# Sim-to-Real Diffusion Policy for AIC Cable Insertion

**Pipeline:** Ground-truth Sim Data Collection → Diffusion Policy Training → AIC Scoring vs ACT Baseline

---

## Overview

This report documents a complete imitation learning pipeline for the AIC cable insertion task:

1. **Data Collection** — Record 20 ground-truth demonstrations from Gazebo simulation using `RecordCheatCode`, with diverse board/cable poses baked into per-episode engine configs
2. **Training** — Train a Diffusion Policy (ResNet-18 × 3 cameras + DDPM U-Net) on the real sim trajectories using LeRobot
3. **Evaluation** — Run the trained policy through the AIC scoring engine and compare with the `RunACT` baseline

**Key Result:** SimDiffusion achieves **6.5× lower validation loss** than synthetic-data diffusion and **successfully inserts the cable** in live AIC scoring while the RunACT baseline fails.

---

## 1. Dataset

### Collection Setup

| Parameter | Value |
|---|---|
| Simulator | Gazebo (ROS 2 Kilted Kaiju, headless) |
| Policy | `RecordCheatCode` — CheatCode + observation recording |
| Episodes | 20 |
| Steps/episode | 530 (100 approach + 430 insertion) |
| Control rate | 20 Hz |
| Total frames | 10,600 |

### Scene Diversity

Each episode uses a per-episode `aic_engine_config_file` with varied poses:

| Parameter | Default | Perturbation |
|---|---|---|
| `task_board_x` | 0.15 m | ± 2.0 cm |
| `task_board_y` | −0.20 m | ± 2.0 cm |
| `task_board_yaw` | 3.14 rad | ± 0.05 rad |
| `cable_roll` | 0.4432 rad | ± 0.03 rad |
| `cable_pitch` | −0.4838 rad | ± 0.03 rad |
| `cable_yaw` | 1.3303 rad | ± 0.03 rad |

> Episodes 1–10 use the default board pose. Episodes 11–20 use the varied pose configs.

### Data Format

Each episode is stored as a `.npz` file and then encoded into a LeRobot dataset (`local/aic_cable_insertion_sim`) with video-compressed camera streams:

```
observation.state                (530, 26)  float32
  tcp_pose       (7D) — position xyz + quaternion xyzw
  tcp_velocity   (6D) — linear + angular velocity
  tcp_error      (6D) — Cartesian impedance tracking error
  joint_positions (7D) — UR5e joint angles

observation.images.center_camera (530, 128, 144, 3)  uint8 RGB → mp4
observation.images.left_camera   (530, 128, 144, 3)  uint8 RGB → mp4
observation.images.right_camera  (530, 128, 144, 3)  uint8 RGB → mp4

action                           (530, 6)   float32
  linear  (3D) — TCP cartesian velocity (m/s)
  angular (3D) — TCP angular velocity (rad/s)
```

### Dataset Statistics

![Dataset statistics](report_assets/fig6_dataset_stats.png)

All 20 episodes complete 530 steps. The `vz` (insertion axis) dominates the action distribution — the policy primarily learns a downward insertion motion with small xy corrections.

---

## 2. Input / Output Example

### Camera Observations (3 views at 128×144)

The policy conditions on the **last 2 time steps** (`n_obs_steps=2`) of all three cameras plus the proprioceptive state.

![Camera observations across episode](report_assets/fig3_camera_observations.png)

*Rows: center / left / right camera. Columns: key timesteps from approach (t=0) to full insertion (t=26.5 s).*

### State Vector (26D) — Frame t=200 (insertion phase)

```
tcp_pose.position    : x=−0.3754  y= 0.1955  z= 0.2261  (m)
tcp_pose.quaternion  : x= 0.9848  y=−0.0076  z=−0.0043  w= 0.1737
tcp_velocity.linear  : vx=−0.0009  vy=−0.0001  vz=−0.0065  (m/s)
tcp_velocity.angular : wx=−0.0000  wy=−0.0003  wz= 0.0001  (rad/s)
tcp_error            : ex= 0.0021  ey= 0.0003  ez=−0.0006  (m)
                       erx= 0.0002 ery=−0.0000 erz= 0.0000  (rad)
joint_positions      : [−0.389, −1.505, −1.815, −1.519,  1.897,  1.176,  0.004]  (rad)
```

### Action (6D cartesian velocity) — Frame t=200

```
linear.x  : −0.000897 m/s   (lateral)
linear.y  : −0.000096 m/s   (depth)
linear.z  : −0.006519 m/s   ← dominant: downward insertion
angular.x : −0.000040 rad/s
angular.y : −0.000329 rad/s
angular.z :  0.000068 rad/s
```

### Full Episode Trajectories

![State trajectory](report_assets/fig4_state_trajectory.png)

*Blue shading = approach phase (0–5 s), green shading = insertion phase (5–26.5 s). TCP z descends steadily during insertion.*

![Action trajectory](report_assets/fig5_action_trajectory.png)

*`vz` (blue) is the dominant action component. Small `ang.x` corrections (pink dashed) compensate for cable orientation.*

---

## 3. Model Architecture

```
DiffusionPolicy (LeRobot)          80.4M parameters

Observation encoder:
  ┌─ ResNet-18 (center_camera)  →  32 spatial softmax keypoints (×2 obs_steps)
  ├─ ResNet-18 (left_camera)    →  32 spatial softmax keypoints (×2 obs_steps)
  ├─ ResNet-18 (right_camera)   →  32 spatial softmax keypoints (×2 obs_steps)
  └─ Linear(26 → 128) (state)   →  128-D embedding (×2 obs_steps)
  concat → FiLM conditioning vector (global)

Denoising backbone:
  1D U-Net  (down_dims: 256 → 512 → 1024)
  kernel_size=5, GroupNorm(8), FiLM scale modulation

Noise scheduler: DDPM, 100 timesteps, ε-prediction, squaredcos_cap_v2 schedule

Temporal:
  n_obs_steps    = 2    (condition on last 2 observations)
  horizon        = 16   (predict 16 future actions)
  n_action_steps = 8    (execute 8 before re-planning)
```

---

## 4. Training

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β₁=0.95, β₂=0.999, ε=1e-8) |
| Learning rate | 1e-4 with cosine annealing |
| Weight decay | 1e-6 |
| Batch size | 16 |
| Total steps | 5000 |
| Epochs | 8 |
| Train/val split | 85% / 15% (~9,010 / 1,590 frames) |
| Video backend | pyav |
| Normalization | MIN_MAX for state+action, MEAN_STD for images |

### Training Curves

![Training curves](report_assets/fig1_training_curves.png)

*Log-scale. Sim-Diffusion (blue) converges far below Syn-Diffusion (red) at every epoch.*

### Full Training Log

| Epoch | LR | Train Loss | Val Loss | Best |
|---|---|---|---|---|
| 1 | 9.69e-5 | 0.15280 | 0.05351 | |
| 2 | 8.81e-5 | 0.04179 | 0.03320 | |
| 3 | 7.46e-5 | 0.03391 | 0.02983 | |
| 4 | 5.82e-5 | 0.02941 | 0.03301 | |
| 5 | 4.08e-5 | 0.02831 | 0.02632 | |
| 6 | 2.46e-5 | 0.02500 | 0.02440 | |
| **7** | **1.16e-5** | **0.02369** | **0.02325** | **✓ Best** |
| 8 | 3.38e-6 | 0.02168 | 0.02332 | |

Best checkpoint saved at epoch 7: `checkpoints_diffusion_aic_cable_insertion_sim/best_model/`

---

## 5. Results

### Validation Loss Comparison

![Validation loss comparison](report_assets/fig2_val_loss_comparison.png)

| Model | Data Source | Best Val Loss | Notes |
|---|---|---|---|
| **SimDiffusion (this work)** | 20 real sim demos | **0.02325** | Ground-truth trajectories |
| Syn-Diffusion | Synthetic procedural | 0.15079 | 6.5× worse |
| ACT Baseline | — | 1.069 | CVAE ELBO (different scale) |

### Live AIC Scoring

Both policies were evaluated through the AIC engine (single trial, default scene, `ground_truth:=false`):

![Scoring comparison](report_assets/fig7_scoring_comparison.png)

| Metric | **SimDiffusion** | RunACT |
|---|---|---|
| **Cable insertion success** | **YES ✅** | NO ❌ |

**SimDiffusion successfully inserts the cable.** RunACT fails under the same conditions.

---

## 6. Inference Policy

The trained policy runs as a standard AIC policy node:

```bash
# Deploy:
ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.RunSimDiffusion.RunSimDiffusion

# Override checkpoint path:
export AIC_DIFFUSION_CKPT=/path/to/checkpoints_diffusion_aic_cable_insertion_sim/best_model
```

**Inference loop** (`RunSimDiffusion.py`):
```
For each control step (20 Hz):
  1. Observe: buffer last N_OBS=2 observations
  2. Re-plan (every N_ACTION=8 steps):
       preprocess → normalize → model.select_action()
       → 16 future action vectors (DDPM 100-step denoising)
  3. Execute: action[i] → integrate velocity → TCP pose target
       target_pos = current_pos + velocity × dt (dt=0.05 s)
```

---

## 7. Reproduction

```bash
# 1. Collect 20 sim demonstrations
python3 lerobot_aic/collect_sim_demos.py \
    --n_episodes 20 \
    --dataset_name local/aic_cable_insertion_sim

# 2. Train diffusion policy
python3 lerobot_aic/train_diffusion.py \
    --repo_id local/aic_cable_insertion_sim \
    --run_name aic_cable_insertion_sim \
    --steps 5000 --batch_size 16

# 3. Compare with ACT baseline (metrics only)
python3 lerobot_aic/compare_with_act_baseline.py

# 4. Full live scoring eval (requires simulation)
python3 lerobot_aic/compare_with_act_baseline.py --run_eval
```

Or use the single pipeline script:
```bash
bash lerobot_aic/run_pipeline.sh
```

---

## 8. Files

```
lerobot_aic/
  collect_sim_demos.py                        Data collection orchestrator
  train_diffusion.py                          Diffusion policy trainer
  compare_with_act_baseline.py                Metrics + scoring comparison
  run_pipeline.sh                             End-to-end pipeline script
  single_trial_config.yaml                    Per-episode engine config template
  checkpoints_diffusion_aic_cable_insertion_sim/best_model/   Trained checkpoint
  outputs_diffusion_aic_cable_insertion_sim/
    metrics.json                              Full training history
    run_config.json                           Training hyperparameters
    vs_act_comparison.json                    Final comparison results
  report_assets/
    fig1_training_curves.png
    fig2_val_loss_comparison.png
    fig3_camera_observations.png
    fig4_state_trajectory.png
    fig5_action_trajectory.png
    fig6_dataset_stats.png
    fig7_scoring_comparison.png

aic_example_policies/aic_example_policies/ros/
  RecordCheatCode.py                          Demonstration recorder
  RunSimDiffusion.py                          Inference policy
```
