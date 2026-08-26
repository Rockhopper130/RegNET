"""
Run S-RegNET inference for OASIS_OAS1_{idx}_MR1, idx = 0001..0100.

Mirrors M-RegNET/run_inference_all.py: loads the model + template ONCE,
then loops in-process over every sample, scoring the warped template
against the SynthSeg target (NOT the GT). Prints per-sample Dice / AVD /
folding % and an overall summary at the end.

Run from the S-RegNET directory (imports the local `inference` module):
    python temp_run_inference_all.py
"""

from pathlib import Path
import random

from inference import setup_inference, run_inference_on_sample

# =============================================================================
# CONFIGURE HERE
# =============================================================================

# training_seg_acm/20260528_115247
CHECKPOINT = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/training_seg_acm/20260528_115247/checkpoints/best_model.pth"

# OASIS SynthSeg segs — registration target AND scoring reference.
SYNTHSEG_DIR = "/shared/home/v_nishchay_nilabh/shared_scratch/oasis_data/anna_data/oasis_dataset/oasis_synthseg_output/output"
INPUT_SEG_TEMPLATE  = SYNTHSEG_DIR + "/OASIS_OAS1_{idx}_MR1/orig_synthseg.nii.gz"

# Outputs land next to the checkpoint, under .../<TIMESTAMP>/run_inference_all_results/
OUTPUT_DIR_TEMPLATE = str(Path(CHECKPOINT).parent.parent /
                          "run_inference_all_results" / "OASIS_OAS1_{idx}_MR1")

DEVICE = "cuda:7"
USE_AFFINE = False        # must match the checkpoint (config.yaml affine.enabled: false)
LOSSES_ONLY = False       # True skips per-sample viz + NIfTI (faster over 100 samples)

# =============================================================================

# Load template + model ONCE; reuse across all samples.
ctx = setup_inference(CHECKPOINT, config_path=None, device=DEVICE,
                      use_affine=USE_AFFINE, verbose=True)

dice_scores = {}
avd_scores = {}
folding_scores = {}   # per-sample folding % (voxels with det(J) < 0)
min_dets = {}         # per-sample worst (most-negative) normalized det
skipped = []

k = 10
idxs = random.sample(range(1, 101), k)

for i in idxs:
    idx = f"{i:04d}"
    input_seg = INPUT_SEG_TEMPLATE.format(idx=idx)
    output_dir = OUTPUT_DIR_TEMPLATE.format(idx=idx)

    if not Path(input_seg).exists():
        print(f"[{idx}] SKIP — input file not found")
        skipped.append(idx)
        continue

    print(f"[{idx}] Running inference...", flush=True)

    try:
        losses = run_inference_on_sample(
            ctx, input_seg, output_dir,
            losses_only=LOSSES_ONLY, verbose=False,
        )
    except Exception as e:
        print(f"[{idx}] FAILED: {e}")
        skipped.append(idx)
        continue

    dice = losses.get("dice_score")
    dice_scores[idx] = dice
    print(f"[{idx}] Mean Dice: {dice:.4f}")

    avd = losses.get("avg_hausdorff")
    avd_scores[idx] = avd
    print(f"[{idx}] Mean AVD: {avd:.4f}")

    fold = losses.get("folding_pct")
    folding_scores[idx] = fold
    min_dets[idx] = losses.get("min_det")
    print(f"[{idx}] Folding %: {fold:.4f}%  (min det = {min_dets[idx]:.4f})")


# Summary
print("\n" + "=" * 50)
print(f"Processed : {len(dice_scores)} / 100")
print(f"Skipped   : {len(skipped)} {skipped if skipped else ''}")

if not dice_scores:
    print("No samples processed.")
    raise SystemExit(0)

avg_dice = sum(dice_scores.values()) / len(dice_scores)
best_idx  = max(dice_scores, key=dice_scores.get)   # highest Dice = best
worst_idx = min(dice_scores, key=dice_scores.get)
print(f"Avg Dice  : {avg_dice:.4f}")
print(f"Best      : {best_idx}  ({dice_scores[best_idx]:.4f})")
print(f"Worst     : {worst_idx}  ({dice_scores[worst_idx]:.4f})")

avg_avd = sum(avd_scores.values()) / len(avd_scores)
best_idx  = min(avd_scores, key=avd_scores.get)     # lowest AVD = best
worst_idx = max(avd_scores, key=avd_scores.get)
print(f"Avg AVD   : {avg_avd:.4f}")
print(f"Best      : {best_idx}  ({avd_scores[best_idx]:.4f})")
print(f"Worst     : {worst_idx}  ({avd_scores[worst_idx]:.4f})")

avg_folding = sum(folding_scores.values()) / len(folding_scores)
best_idx  = min(folding_scores, key=folding_scores.get)   # least folding = best
worst_idx = max(folding_scores, key=folding_scores.get)   # most folding  = worst
print(f"Avg Folding : {avg_folding:.4f}%")
print(f"Best        : {best_idx}  ({folding_scores[best_idx]:.4f}%)")
print(f"Worst       : {worst_idx}  ({folding_scores[worst_idx]:.4f}%, "
      f"min det = {min_dets[worst_idx]:.4f})")

print("=" * 50)
