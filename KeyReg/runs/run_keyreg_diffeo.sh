#!/bin/bash
export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp; mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg
FSPY=/shared/scratch/0/home/v_vijay_bala_mahalingam/freesurfer/python/bin/python3
echo "[$(date +%F_%H:%M:%S)] START Hybrid DIFFEO v2 (int_steps=5) GPU6"
"$FSPY" keyreg.py --hybrid --diffeo --int_steps 5 --epochs 300 --steps 80 --K 256 --field_n 64 \
  --lam 0.8 --delta 0.5 --fold_w 2.0 --lr 4e-4 --wm_w 3.0 \
  --out /shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/hybrid_diffeo2
echo "[$(date +%F_%H:%M:%S)] DIFFEO_EXIT=$?"
