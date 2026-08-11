#!/bin/bash
export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp; mkdir -p "$TMPDIR"; export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg
FSPY=/shared/scratch/0/home/v_vijay_bala_mahalingam/freesurfer/python/bin/python3
echo "[$(date +%F_%H:%M:%S)] START SVF-C diffeo-max (int=12 smooth=600 delta=0.30) GPU4"
"$FSPY" keyreg.py --svf --int_steps 12 --smooth_w 600 --delta 0.30 --fold_w 10   --epochs 300 --steps 80 --K 256 --lr 4e-4 --wm_w 3.0   --out /shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/svf_C
echo "[$(date +%F_%H:%M:%S)] SVFC_EXIT=$?"
