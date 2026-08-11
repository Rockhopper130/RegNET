"""
WM-surface comparison over N random OASIS subjects (model loaded ONCE).

For each sampled subject this registers the template → SynthSeg, extracts the
generated / synthseg / actual WM surfaces with the SAME marching-cubes
extractor, writes the per-subject figure (heatmap contours + histograms) to
IMAGES_DIR/wm_compare_<idx>.png, and records the three pairwise mean/hd95
surface distances.

A single image can't answer "is the warp closer to GT than SynthSeg" — the eye
reads contour smoothness, not distance. The aggregate over N subjects can: at
the end this prints the mean/median of each pairwise distance and writes
IMAGES_DIR/summary.png, a scatter of synthseg↔actual vs generated↔actual with
the y=x line. Points above the line are subjects where the warp measures
farther from GT than SynthSeg does — the typical (not guaranteed) outcome,
since the model is trained against SynthSeg and never sees GT. Points below the
line are real and informative: there the clean template-warp landed closer to
GT than SynthSeg's noisier boundary.

Run from the S-RegNET directory:
    python eval/compare_wm_surface_all.py
"""

from pathlib import Path
import json
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import setup_inference
from compare_wm_surface import warp_to_synthseg, analyze_wm, render_compare

# =============================================================================
# CONFIGURE HERE
# =============================================================================

# training_seg_acm/20260528_115247
CHECKPOINT = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/training_seg_acm/20260528_115247/checkpoints/best_model.pth"

OASIS_SCANS_DIR = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans"
SYNTHSEG_DIR    = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/oasis_dataset/oasis_synthseg_output/output"

IMAGES_DIR = Path("images")
N_SAMPLES  = 20
SEED       = 0            # reproducible draw from 1..99; set None for fresh each run
SMOOTH     = 1.2
DEVICE     = "cuda:0"
USE_AFFINE = False        # must match the checkpoint

# =============================================================================

ctx = setup_inference(CHECKPOINT, config_path=None, device=DEVICE,
                      use_affine=USE_AFFINE, verbose=True)
vox_mm = 256.0 / ctx['target_size'][0]
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(SEED)
idxs = sorted(rng.sample(range(1, 100), N_SAMPLES))

results = []   # {idx, gen_gt, ss_gt, gen_ss} each a (mean_mm, hd95_mm) tuple
skipped = []

for i in idxs:
    idx = f"{i:04d}"
    synthseg = f"{SYNTHSEG_DIR}/OASIS_OAS1_{idx}_MR1/orig_synthseg.nii.gz"
    gt       = f"{OASIS_SCANS_DIR}/OASIS_OAS1_{idx}_MR1/seg4_onehot.npy"

    if not (Path(synthseg).exists() and Path(gt).exists()):
        print(f"[{idx}] SKIP — missing input")
        skipped.append(idx)
        continue

    print(f"[{idx}] Running...", flush=True)
    try:
        gen_lbl, ss_lbl, gt_lbl = warp_to_synthseg(ctx, synthseg, gt)
        analysis = analyze_wm(gen_lbl, ss_lbl, gt_lbl, SMOOTH)
    except Exception as e:
        print(f"[{idx}] FAILED: {e}")
        skipped.append(idx)
        continue

    render_compare(analysis, gt_lbl, IMAGES_DIR / f"wm_compare_{idx}.png",
                   f"OASIS_OAS1_{idx}_MR1 — WM surface vs GT: generated / synthseg",
                   vox_mm)

    p = analysis['pairs']
    r = {'idx': idx,
         'gen_gt': [p['gen_gt'][0] * vox_mm, p['gen_gt'][1] * vox_mm],
         'ss_gt':  [p['ss_gt'][0]  * vox_mm, p['ss_gt'][1]  * vox_mm],
         'gen_ss': [p['gen_ss'][0] * vox_mm, p['gen_ss'][1] * vox_mm]}
    results.append(r)
    print(f"    gen↔gt {r['gen_gt'][0]:.3f}  ss↔gt {r['ss_gt'][0]:.3f}  "
          f"gen↔ss {r['gen_ss'][0]:.3f}  (mean mm)")

# =============================================================================
# Aggregate
# =============================================================================
print("\n" + "=" * 64)
print(f"Processed : {len(results)} / {N_SAMPLES}")
if skipped:
    print(f"Skipped   : {skipped}")
if not results:
    raise SystemExit(0)

gen_gt = np.array([r['gen_gt'] for r in results])   # (N,2): mean, hd95
ss_gt  = np.array([r['ss_gt']  for r in results])
gen_ss = np.array([r['gen_ss'] for r in results])

print(f"\n{'pair':22s} {'mean mm':>9s} {'median mm':>11s} {'mean hd95':>11s}")
for label, arr in [('generated ↔ synthseg', gen_ss),
                   ('synthseg  ↔ actual',  ss_gt),
                   ('generated ↔ actual',  gen_gt)]:
    print(f"{label:22s} {arr[:, 0].mean():9.3f} {np.median(arr[:, 0]):11.3f} "
          f"{arr[:, 1].mean():11.3f}")

n_warp_worse = int((gen_gt[:, 0] > ss_gt[:, 0]).sum())
print(f"\nwarp farther from GT than synthseg : {n_warp_worse}/{len(results)} subjects")

# Scatter: synthseg↔actual (x) vs generated↔actual (y), y=x reference.
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(ss_gt[:, 0], gen_gt[:, 0], c='steelblue', s=40, zorder=3)
lim = float(max(ss_gt[:, 0].max(), gen_gt[:, 0].max())) * 1.1
ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y = x (equal to GT)')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('synthseg ↔ actual  mean dist (mm)')
ax.set_ylabel('generated ↔ actual  mean dist (mm)')
ax.set_title(f'WM surface distance to GT  (n={len(results)})\n'
             f'above the line = warp farther from GT than SynthSeg')
ax.legend()
fig.tight_layout()
fig.savefig(IMAGES_DIR / "summary.png", dpi=150)
plt.close(fig)
print(f"\nSummary scatter → {IMAGES_DIR / 'summary.png'}")

(IMAGES_DIR / "summary.json").write_text(json.dumps({
    'checkpoint': CHECKPOINT, 'seed': SEED, 'n_samples': N_SAMPLES,
    'vox_mm': vox_mm, 'per_sample': results, 'skipped': skipped,
    'aggregate_mean_mm': {
        'generated_synthseg': float(gen_ss[:, 0].mean()),
        'synthseg_actual':    float(ss_gt[:, 0].mean()),
        'generated_actual':   float(gen_gt[:, 0].mean()),
    },
}, indent=2))
print(f"Per-sample metrics → {IMAGES_DIR / 'summary.json'}")
print("=" * 64)
