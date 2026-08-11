#!/bin/bash
export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp; mkdir -p "$TMPDIR"; export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg
FSPY=/shared/scratch/0/home/v_vijay_bala_mahalingam/freesurfer/python/bin/python3
echo "[$(date +%F_%H:%M:%S)] START SVF-E beat-team (int=12 smooth=3000 delta=0.20 fold_w=40) GPU6"
"$FSPY" keyreg.py --svf --int_steps 12 --smooth_w 3000 --delta 0.20 --fold_w 40   --epochs 250 --steps 80 --K 256 --lr 4e-4 --wm_w 3.0   --out /shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/svf_E
echo "[$(date +%F_%H:%M:%S)] SVFE_EXIT=$?"
