#!/bin/bash
export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp; mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg
FSPY=/shared/scratch/0/home/v_vijay_bala_mahalingam/freesurfer/python/bin/python3
echo "[$(date +%F_%H:%M:%S)] START Hybrid FINAL (tighter fold control) GPU7"
"$FSPY" keyreg.py --hybrid --epochs 300 --steps 80 --K 256 --field_n 64 \
  --lam 0.15 --delta 0.35 --fold_w 5.0 --lr 4e-4 --wm_w 3.0 \
  --out /shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/hybrid_final
echo "[$(date +%F_%H:%M:%S)] KEYREG_EXIT=$?"
