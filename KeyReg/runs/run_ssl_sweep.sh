#!/bin/bash
# Phase 2a: smoothness sweep for the intensity-driven SSL baseline.
# Short runs whose only job is to find the smooth_w whose REALIZED deformation
# regularity matches SVF-E (std log det J ~0.336, mean |disp| ~2.33 mm).
# Copying SVF-E's smooth_w=3000 would not be fair: same units, different data term.
set -uo pipefail
export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU= after checking nvidia-smi}
FSPY=/shared/scratch/0/home/v_vijay_bala_mahalingam/freesurfer/python/bin/python3
cd /shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg
SM=${SMOOTH:?set SMOOTH=}
EP=${EPOCHS:-40}
OUT=/shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/ssl_sweep_${SM}
echo "[$(date +%F_%H:%M:%S)] SSL sweep smooth_w=$SM epochs=$EP -> $OUT"
"$FSPY" -u ssl_intensity.py --smooth_w "$SM" --fold_w 40 --epochs "$EP" \
    --out "$OUT" --device cuda:0
echo "[$(date +%F_%H:%M:%S)] EXIT=$?"
