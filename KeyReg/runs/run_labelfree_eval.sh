#!/bin/bash
# Phase 1: label-free evaluation of svf_E_full on both GT and recon-all-clinical.
# Survives a closed laptop -- launch under tmux.
#
#   GPU=2 bash runs/run_labelfree_eval.sh            # full
#   GPU=2 bash runs/run_labelfree_eval.sh --limit 10 # probe
set -euo pipefail

# /tmp on this host is 100% full (another user's checkpoints); anything that spills
# there dies hours in.
export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# Required from the environment on purpose, so the launcher has to check nvidia-smi
# rather than pinning a busy GPU out of habit.
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU= after checking nvidia-smi}

FSPY=/shared/scratch/0/home/v_vijay_bala_mahalingam/freesurfer/python/bin/python3
DATA=/shared/scratch/0/home/v_vijay_bala_mahalingam/neurite_oasis
cd /shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg

EXTRA="$*"
RUN=${RUN:-svf_E_full}

echo "[$(date +%F_%H:%M:%S)] === GT arm (paired cohort) ==="
"$FSPY" -u eval_labelfree.py \
    --val "$DATA/full_val.txt" --fixed_name seg4_onehot.npy \
    --run "$RUN" --paired_only \
    --int_sweep 4,6,8,10,12,14,16 \
    --csv results/labelfree_gt.csv \
    --device cuda:0 $EXTRA

echo "[$(date +%F_%H:%M:%S)] === recon-all-clinical arm (same cohort) ==="
"$FSPY" -u eval_labelfree.py \
    --val "$DATA/full_val.txt" --fixed_name seg4_onehot_clinical.npy \
    --run "$RUN" --paired_only \
    --csv results/labelfree_clinical.csv \
    --device cuda:0 $EXTRA

echo "[$(date +%F_%H:%M:%S)] === DONE ==="
