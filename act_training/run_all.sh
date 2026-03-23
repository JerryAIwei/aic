#!/bin/bash
# run_all.sh – End-to-end ACT training pipeline
# Usage: bash run_all.sh [--epochs N] [--batch_size N] [--lr F] [--beta F]

set -e
cd "$(dirname "$0")"

echo "============================================"
echo " ACT Policy Training Pipeline"
echo "============================================"

# Parse optional args to pass through to train.py
TRAIN_ARGS="$@"

echo ""
echo "[1/4] Generating synthetic dataset …"
python3 generate_dataset.py

echo ""
echo "[2/4] Training ACT policy …"
python3 train.py $TRAIN_ARGS

echo ""
echo "[3/4] Generating visualisation plots …"
python3 monitor.py

echo ""
echo "[4/4] Building HTML report …"
python3 report.py

echo ""
echo "============================================"
echo " Done!  Open reports/training_report.html"
echo "============================================"
