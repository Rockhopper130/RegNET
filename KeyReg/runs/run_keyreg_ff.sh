#!/bin/bash
export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp; mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg
FSPY=/shared/scratch/0/home/v_vijay_bala_mahalingam/freesurfer/python/bin/python3
echo "[$(date +%F_%H:%M:%S)] START Hybrid FOLD-FREE (affine-TPS + SVF residual) GPU4"
"$FSPY" keyreg.py --hybrid --diffeo --tps_affine --int_steps 6 --epochs 300 --steps 80 \
  --K 256 --field_n 64 --delta 0.6 --fold_w 1.0 --lr 4e-4 --wm_w 3.0 \
  --out /shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/hybrid_foldfree
echo "[$(date +%F_%H:%M:%S)] FF_EXIT=$?"
