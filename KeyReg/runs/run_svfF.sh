#!/bin/bash
export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp; mkdir -p "$TMPDIR"; export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg
FSPY=/shared/scratch/0/home/v_vijay_bala_mahalingam/freesurfer/python/bin/python3
echo "[$(date +%F_%H:%M:%S)] START SVF-F ultra-smooth (smooth=8000 int=14 delta=0.18) GPU7"
"$FSPY" keyreg.py --svf --int_steps 14 --smooth_w 8000 --delta 0.18 --fold_w 60 \
  --epochs 250 --steps 80 --K 256 --lr 4e-4 --wm_w 3.0 \
  --out /shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/svf_F
echo "[$(date +%F_%H:%M:%S)] SVFF_EXIT=$?"
