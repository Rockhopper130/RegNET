#!/bin/bash
export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp; mkdir -p "$TMPDIR"; export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg
FSPY=/shared/scratch/0/home/v_vijay_bala_mahalingam/freesurfer/python/bin/python3
echo "[$(date +%F_%H:%M:%S)] START SVF-B min-fold (smooth_w=400,int=10) GPU6"
"$FSPY" keyreg.py --svf --smooth_w 400 --int_steps 10 --delta 0.4 --fold_w 2.0 \
  --epochs 300 --steps 80 --K 256 --lr 4e-4 --wm_w 3.0 \
  --out /shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/svf_B
echo "[$(date +%F_%H:%M:%S)] SVFB_EXIT=$?"
