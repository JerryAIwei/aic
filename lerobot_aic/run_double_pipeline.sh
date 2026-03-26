#!/usr/bin/env bash
# run_double_pipeline.sh
# 1. Collect 20 more episodes (seed=100) → 40 total in /tmp/aic_recordings/
# 2. Build local/aic_cable_insertion_sim_40 LeRobot dataset
# 3. Train new diffusion policy on 40-episode dataset
# 4. Evaluate success rate: model_20ep vs model_40ep over 5 trials each
# 5. Print comparison report

set -e
cd /workspace/aic
LOG=/tmp/double_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Double-Dataset Pipeline started at $(date) ==="

# ── Step 1: Collect 20 more episodes ─────────────────────────────────────────
echo ""
echo "=== Step 1: Collecting 20 more episodes (seed=100) ==="
python3 lerobot_aic/collect_sim_demos.py \
    --n_episodes 40 \
    --dataset_name local/aic_cable_insertion_sim_40 \
    --seed 100 \
    --resume \
    --start_idx 20
echo "Collection done at $(date)"

# ── Step 2: Train on 40-episode dataset ──────────────────────────────────────
echo ""
echo "=== Step 2: Training on 40-episode dataset ==="
python3 lerobot_aic/train_diffusion.py \
    --repo_id local/aic_cable_insertion_sim_40 \
    --run_name aic_cable_insertion_sim_40 \
    --steps 8000 \
    --batch_size 16 \
    --lr 1e-4

echo "Training done at $(date)"

# ── Step 3: Success-rate evaluation ──────────────────────────────────────────
echo ""
echo "=== Step 3: Running success-rate evaluation ==="
python3 lerobot_aic/eval_success_rate.py \
    --n_trials 5 \
    --ckpt_20ep lerobot_aic/checkpoints_diffusion_aic_cable_insertion_sim/best_model \
    --ckpt_40ep lerobot_aic/checkpoints_diffusion_aic_cable_insertion_sim_40/best_model \
    --output    lerobot_aic/outputs_diffusion_aic_cable_insertion_sim_40/success_rate.json

echo "=== Double-Dataset Pipeline complete at $(date) ==="
