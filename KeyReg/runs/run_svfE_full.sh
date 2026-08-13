#!/bin/bash
# SVF-E ("balanced", the 0.91 config) on the full neurite-OASIS set: 330 train / 83 val.
# Chained so it survives a closed laptop: build split -> one-hot the missing
# subjects -> train. Each stage aborts the run if it fails.
set -euo pipefail

export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp; mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

FSPY=/shared/scratch/0/home/v_vijay_bala_mahalingam/freesurfer/python/bin/python3
DATA=/shared/scratch/0/home/v_vijay_bala_mahalingam/neurite_oasis
OUT=/shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/svf_E_full
TEMPLATE=$DATA/OASIS_OAS1_0001_MR1/seg4_onehot.npy
mkdir -p "$OUT"

echo "[$(date +%F_%H:%M:%S)] === STAGE 1/3: build 330/83 split ==="
"$FSPY" - <<PY
import os, random
d = "$DATA"
subs = sorted(s for s in os.listdir(d) if s.startswith("OASIS_"))
subs = [s for s in subs if s != "OASIS_OAS1_0001_MR1"]     # template held out
assert len(subs) == 413, f"expected 413 non-template subjects, got {len(subs)}"
random.Random(0).shuffle(subs)
tr, va = subs[:330], subs[330:]
assert len(tr) == 330 and len(va) == 83
for name, group in (("full_train.txt", tr), ("full_val.txt", va)):
    with open(os.path.join(d, name), "w") as f:
        for s in group:
            f.write(os.path.join(d, s, "seg4_onehot.npy") + "\n")
print(f"wrote full_train.txt ({len(tr)}) and full_val.txt ({len(va)})")
PY

echo "[$(date +%F_%H:%M:%S)] === STAGE 2/3: one-hot the missing subjects ==="
"$FSPY" - <<PY
import nibabel as nib, numpy as np, os, time
d = "$DATA"
subs = sorted(s for s in os.listdir(d) if s.startswith("OASIS_"))
todo = [s for s in subs if not os.path.exists(os.path.join(d, s, "seg4_onehot.npy"))]
print(f"{len(subs)} subjects, {len(subs)-len(todo)} already done, {len(todo)} to convert", flush=True)
bad = []
t0 = time.time()
for i, s in enumerate(todo, 1):
    src = os.path.join(d, s, "aligned_seg4.nii.gz")
    seg = nib.load(src).get_fdata().astype(np.int16)
    oh = np.zeros((5, *seg.shape), dtype=np.uint8)
    for c in range(5):
        oh[c] = (seg == c)
    chk = oh.sum(0)
    if not (chk.min() == 1 and chk.max() == 1):
        bad.append(s); print("BAD onehot (labels outside 0-4):", s, flush=True)
    np.save(os.path.join(d, s, "seg4_onehot.npy"), oh)
    if i % 25 == 0:
        print(f"  {i}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"converted {len(todo)}, malformed {len(bad)}", flush=True)
if bad:
    raise SystemExit("refusing to train: malformed one-hot for " + ", ".join(bad))
PY

echo "[$(date +%F_%H:%M:%S)] === STAGE 3/3: train SVF-E on 330/83 ==="
cd /shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg
"$FSPY" keyreg.py --svf --int_steps 12 --smooth_w 3000 --delta 0.20 --fold_w 40 \
  --epochs 250 --steps 80 --K 256 --lr 4e-4 --wm_w 3.0 \
  --train_txt "$DATA/full_train.txt" \
  --val_txt   "$DATA/full_val.txt" \
  --template  "$TEMPLATE" \
  --out "$OUT"
echo "[$(date +%F_%H:%M:%S)] SVFE_FULL_EXIT=$?"
