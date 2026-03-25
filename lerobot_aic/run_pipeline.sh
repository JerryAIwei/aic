#!/usr/bin/env bash
# run_pipeline.sh
# Wait for collection to finish, then train diffusion policy and compare with ACT.
# Usage: bash lerobot_aic/run_pipeline.sh

set -e
cd /workspace/aic

LOG=/tmp/pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Pipeline started at $(date) ==="

# ── Wait for collection ───────────────────────────────────────────────────────
echo "Waiting for collect_sim_demos.py to finish..."
COLL_PID=$(pgrep -f "collect_sim_demos.py" || true)
if [ -n "$COLL_PID" ]; then
    echo "  collection PID: $COLL_PID"
    tail --pid="$COLL_PID" -f /dev/null 2>/dev/null || true
else
    echo "  no collection process found — assuming done"
fi
echo "Collection done at $(date)"

# Verify dataset exists
python3 -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
root = HF_LEROBOT_HOME / 'local/aic_cable_insertion_sim'
ds = LeRobotDataset('local/aic_cable_insertion_sim', root=root)
print(f'Dataset: {len(ds):,} frames, {ds.num_episodes} episodes')
"

# ── Train diffusion policy ────────────────────────────────────────────────────
echo ""
echo "=== Training Diffusion Policy at $(date) ==="
python3 lerobot_aic/train_diffusion.py \
    --repo_id local/aic_cable_insertion_sim \
    --run_name aic_cable_insertion_sim \
    --steps 5000 \
    --batch_size 16 \
    --lr 1e-4

echo "Training done at $(date)"

# ── Compare training metrics (no sim eval — just metrics) ────────────────────
echo ""
echo "=== Comparing Training Metrics ==="
python3 lerobot_aic/compare_with_act_baseline.py

echo "=== Pipeline complete at $(date) ==="
