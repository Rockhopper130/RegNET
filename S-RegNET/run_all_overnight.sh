#!/usr/bin/env bash
# =============================================================================
# Automated Overnight Pipeline: Convert HCP Data & Launch Training
# =============================================================================
set -e

echo "=========================================================="
echo "Step 1: Pulling latest code..."
echo "=========================================================="
git pull origin hcp-pial

echo "=========================================================="
echo "Step 2: Generating Train / Val Splits..."
echo "=========================================================="
python3 create_hcp_splits.py \
    --hcp_root /shared/scratch/0/home/v_mahith_reddy/hcp_data \
    --val_count 100 \
    --seed 42

echo "=========================================================="
echo "Step 3: Fast-converting Validation Segmentations..."
echo "=========================================================="
python3 convert_hcp_one_hot.py \
    --hcp_root /shared/scratch/0/home/v_mahith_reddy/hcp_data \
    --subjects_file /shared/scratch/0/home/v_mahith_reddy/hcp_data/val.txt \
    --workers 16

echo "=========================================================="
echo "Step 4: Fast-converting Training Segmentations..."
echo "=========================================================="
python3 convert_hcp_one_hot.py \
    --hcp_root /shared/scratch/0/home/v_mahith_reddy/hcp_data \
    --subjects_file /shared/scratch/0/home/v_mahith_reddy/hcp_data/train.txt \
    --workers 16

echo "=========================================================="
echo "Step 5: Launching S-RegNET Training on GPU..."
echo "=========================================================="
python3 train.py --config config_hcp_pial.yaml
