#!/usr/bin/env bash
# setup_local_training.sh
# Sets up a local Python venv with GPU support for AIC training + visualization.
# Usage:  bash lerobot_aic/setup_local_training.sh

set -euo pipefail
cd "$(dirname "$0")/.."   # → aic/

VENV_DIR="$PWD/lerobot_aic/.venv"

echo "══════════════════════════════════════════════════════════"
echo " AIC Local GPU Training Setup"
echo "══════════════════════════════════════════════════════════"

# ── 1. Create venv ────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[1/5] Creating Python virtual environment …"
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "[1/5] Virtual environment already exists."
fi
source "$VENV_DIR/bin/activate"

# ── 2. Upgrade pip ────────────────────────────────────────────
echo "[2/5] Upgrading pip …"
pip install --upgrade pip setuptools wheel

# ── 3. Install PyTorch with CUDA ──────────────────────────────
echo "[3/5] Installing PyTorch with CUDA support …"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ── 4. Install lerobot + training dependencies ────────────────
echo "[4/5] Installing lerobot and dependencies …"
pip install "lerobot==0.4.3" "huggingface-hub[hf-transfer]==0.35.3" \
    matplotlib numpy Pillow opencv-python-headless

# ── 5. Verify ─────────────────────────────────────────────────
echo "[5/5] Verifying setup …"
python3 -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM     : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import lerobot
print(f'  LeRobot  : {lerobot.__version__}')
import matplotlib
print(f'  Matplotlib: {matplotlib.__version__}')
print()
print('  ✓ All dependencies installed successfully!')
"

echo ""
echo "══════════════════════════════════════════════════════════"
echo " Setup complete!"
echo ""
echo " To activate:   source lerobot_aic/.venv/bin/activate"
echo ""
echo " Quick start:"
echo "   # 1. Generate synthetic dataset (no simulator needed):"
echo "   python lerobot_aic/create_larger_dataset.py"
echo ""
echo "   # 2. Train Diffusion Policy on GPU:"
echo "   python lerobot_aic/train_diffusion.py --steps 5000 --batch_size 8"
echo ""
echo "   # 3. Train ACT Policy on GPU:"
echo "   python lerobot_aic/train_act.py --steps 5000 --batch_size 8"
echo ""
echo "   # 4. Visualise results:"
echo "   python lerobot_aic/visualize_training.py"
echo "══════════════════════════════════════════════════════════"
