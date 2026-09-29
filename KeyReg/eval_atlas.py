"""Phase 1 metric 4: atlas sharpness / population consistency.

Carries every subject's intensity into TEMPLATE space and measures how well the
cohort agrees there. A registration that aligns anatomy produces a sharp cohort mean
and low voxelwise variance; a registration that does nothing produces a blurry mean.

Two guards make the number honest:

  * A FIXED template-space mask. If the ROI followed the warp, a deformation that
    shrinks the brain could win by pushing tissue out of the region being scored.
  * atlas_sharpness reported beside atlas_std. Under the degenerate failure (a warp
    that collapses volume) std falls AND sharpness falls, so the pair separates
    "better aligned" from "squashed". Either number alone cannot.

The unwarped row is mandatory -- without it neither number means anything.

Direction: the forward grid is defined on subject voxels, so resampling a subject
INTO template space uses exp(-v). Getting this backwards produces a perfectly
plausible blurry atlas, which is why the reversed control is asserted.

    python eval_atlas.py --fixed_name seg4_onehot.npy --run svf_E_full
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import svfcommon as S
from keyreg import load_seg, read_list, identity_grid_vol

ap = argparse.ArgumentParser()
ap.add_argument("--val", default=f"{S.NEURITE}/full_val.txt")
ap.add_argument("--fixed_name", default="seg4_onehot.npy")
ap.add_argument("--run", default="svf_E_full")
ap.add_argument("--ckpt", default="best.pth")
ap.add_argument("--template_subject", default=S.TEMPLATE_SUBJ)
ap.add_argument("--paired_only", action="store_true")
ap.add_argument("--erode", type=int, default=6)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--csv", default="")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

S.require_shared_tmpdir()
S.assert_scale_mm(args.template_subject)
dev = args.device

subs = S.check_lists(read_list(args.val))
if args.paired_only:
    keep, _ = S.paired_subjects(args.val)
    subs = [s for s in subs if s in set(keep)]
if args.limit:
    subs = subs[:args.limit]
print(f"[atlas] {len(subs)} subjects, input {args.fixed_name}", flush=True)

ck = torch.load(os.path.join(S.RUNS, args.run, args.ckpt), map_location=dev,
                weights_only=False)
m = S.build_seg_model(ck, dev)
idg = identity_grid_vol(ck["args"]["target"], dev)
tpl_seg = load_seg(os.path.join(S.NEURITE, args.template_subject, "seg4_onehot.npy"),
                   S.TS).unsqueeze(0).to(dev)
tpl_int = S.load_intensity(args.template_subject, device=dev)
mask = S.brain_mask(tpl_int, erode=args.erode)          # FIXED template-space ROI
print(f"[atlas] fixed template mask = {mask.float().mean().item()*100:.1f}% of volume",
      flush=True)


def grad_mag(vol):
    """Mean |grad| inside the mask -- the sharpness of the cohort mean image."""
    gz = vol[:, :, 1:, :, :] - vol[:, :, :-1, :, :]
    gy = vol[:, :, :, 1:, :] - vol[:, :, :, :-1, :]
    gx = vol[:, :, :, :, 1:] - vol[:, :, :, :, :-1]
    g = (gz[:, :, :, :-1, :-1] ** 2 + gy[:, :, :-1, :, :-1] ** 2
         + gx[:, :, :-1, :-1, :] ** 2).sqrt()
    mm = mask[:, :, :-1, :-1, :-1]
    return g[mm].mean().item()


# Streaming accumulators -- never stack the cohort (83 x 128^3 x 4B would be fine,
# but 413 would not, and this keeps the code identical for both).
acc = {k: {"s": torch.zeros_like(tpl_int), "s2": torch.zeros_like(tpl_int), "n": 0}
       for k in ("unwarped", "model", "reversed")}


@torch.no_grad()
def run():
    for i, s in enumerate(subs, 1):
        fix = load_seg(os.path.join(S.NEURITE, s, args.fixed_name),
                       S.TS).unsqueeze(0).to(dev)
        _, g_f, g_i = S.fwd_inv_grids(m, tpl_seg, fix, idg)
        I = S.load_intensity(s, device=dev)
        for key, vol in (("unwarped", I),
                         ("model", S.warp(I, g_i)),     # subject -> template
                         ("reversed", S.warp(I, g_f))): # control: wrong direction
            acc[key]["s"] += vol
            acc[key]["s2"] += vol * vol
            acc[key]["n"] += 1
        del fix, g_f, g_i, I
        torch.cuda.empty_cache()
        if i % 10 == 0 or i == len(subs):
            print(f"  [{i}/{len(subs)}] {s}", flush=True)


run()

out = {}
for k, a in acc.items():
    n = a["n"]
    mean = a["s"] / n
    var = (a["s2"] / n - mean * mean).clamp_min(0)
    out[k] = {"atlas_std": var.sqrt()[mask].mean().item(),
              "atlas_sharpness": grad_mag(mean)}

print("\n" + "=" * 70)
print(f"ATLAS CONSISTENCY  {args.run}  {args.fixed_name}  n={len(subs)}")
print("=" * 70)
print(f"{'arm':<12}{'atlas_std':>14}{'atlas_sharpness':>18}")
print("-" * 70)
for k in ("unwarped", "model", "reversed"):
    print(f"{k:<12}{out[k]['atlas_std']:>14.5f}{out[k]['atlas_sharpness']:>18.5f}")
print("-" * 70)
print("lower std = better alignment; higher sharpness = better.")
print("They move OPPOSITELY under a volume-collapsing warp, so read them together.")

assert out["model"]["atlas_std"] < out["unwarped"]["atlas_std"], (
    "model atlas is not tighter than the unwarped cohort -- registration is not helping")
assert out["reversed"]["atlas_std"] > out["model"]["atlas_std"], (
    "reversed-direction control beats the model -- the warp direction is flipped")
print("\n[control] direction + improvement asserts pass")

if args.csv:
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "atlas_std", "atlas_sharpness", "n", "input"])
        for k in ("unwarped", "model", "reversed"):
            w.writerow([k, out[k]["atlas_std"], out[k]["atlas_sharpness"],
                        len(subs), args.fixed_name])
    print(f"CSV -> {args.csv}")
