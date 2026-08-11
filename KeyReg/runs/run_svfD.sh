#!/bin/bash
export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp; mkdir -p "$TMPDIR"; export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd /shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg
FSPY=/shared/scratch/0/home/v_vijay_bala_mahalingam/freesurfer/python/bin/python3
echo "[$(date +%F_%H:%M:%S)] START SVF-D beat-team (int=14 smooth=1200 delta=0.28 fold_w=20) GPU4"
"$FSPY" keyreg.py --svf --int_steps 14 --smooth_w 1200 --delta 0.28 --fold_w 20   --epochs 250 --steps 80 --K 256 --lr 4e-4 --wm_w 3.0   --out /shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/svf_D
echo "[$(date +%F_%H:%M:%S)] SVFD_EXIT=$?"
