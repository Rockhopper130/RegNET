"""
Convert per-subject SynthSeg label volumes (.nii.gz FreeSurfer labels) to
5-class one-hot .npy next to each subject's GT seg4_onehot.npy. Like
convert_one_hot.py but applies the remap (label_mapping.remap_to_5class) first.

Stored at native resolution; SegDataset resizes at load time. Subjects come
from the train/val list files (each line a subject's seg4_onehot.npy path); the
SynthSeg volume is read from <synthseg_dir>/<subject>/<synthseg_name>.

Usage:
    python convert_synthseg_one_hot.py \
        --lists /path/train.txt /path/val.txt \
        --synthseg_dir /path/oasis_synthseg_output/output \
        --synthseg_name orig_synthseg.nii.gz \
        --out_name synthseg_onehot.npy
"""

import argparse
import os

import numpy as np
import nibabel as nib
from tqdm import tqdm

from label_mapping import remap_to_5class

NUM_CLASSES = 5


def to_one_hot(seg, num_classes=NUM_CLASSES):
    """(D,H,W) int labels → (num_classes, D,H,W) uint8 one-hot."""
    one_hot = np.zeros((num_classes, *seg.shape), dtype=np.uint8)
    for c in range(num_classes):
        one_hot[c] = (seg == c).astype(np.uint8)
    return one_hot


def remap_to_onehot(int_volume, num_classes=NUM_CLASSES):
    """FreeSurfer/SynthSeg integer labels → (num_classes, *vol) uint8 one-hot."""
    return to_one_hot(remap_to_5class(int_volume), num_classes)


def _fg_dice(a_oh, b_oh, num_classes=NUM_CLASSES):
    """Mean foreground Dice between two (C,*vol) one-hot volumes (sanity only)."""
    dices = []
    for c in range(1, num_classes):
        inter = 2.0 * np.sum(a_oh[c] * b_oh[c])
        union = a_oh[c].sum() + b_oh[c].sum()
        dices.append((inter + 1e-5) / (union + 1e-5))
    return float(np.mean(dices))


def main():
    ap = argparse.ArgumentParser(description="SynthSeg .nii.gz → 5-class one-hot .npy")
    ap.add_argument('--lists', nargs='+', required=True,
                    help='train/val txt files; each line = a subject seg4_onehot.npy path')
    ap.add_argument('--synthseg_dir', required=True,
                    help='root holding <subject>/<synthseg_name>')
    ap.add_argument('--synthseg_name', default='orig_synthseg.nii.gz')
    ap.add_argument('--out_name', default='synthseg_onehot.npy')
    args = ap.parse_args()

    gt_paths = []
    for lst in args.lists:
        with open(lst) as f:
            gt_paths += [ln.strip() for ln in f if ln.strip()]

    print(f"{len(gt_paths)} subjects from {len(args.lists)} list(s)")
    sanity_done = False
    written, missing = 0, []

    for gt_path in tqdm(gt_paths, desc="Converting SynthSeg", ncols=80):
        subject = os.path.basename(os.path.dirname(gt_path))   # OASIS_OAS1_0146_MR1
        ss_path = os.path.join(args.synthseg_dir, subject, args.synthseg_name)
        if not os.path.exists(ss_path):
            missing.append(subject)
            continue

        data = nib.load(ss_path).get_fdata().astype(np.int64)
        onehot = remap_to_onehot(data)
        out_path = os.path.join(os.path.dirname(gt_path), args.out_name)
        np.save(out_path, onehot)
        written += 1

        # One-subject sanity check: SynthSeg and GT must share the grid.
        if not sanity_done and os.path.exists(gt_path):
            gt_oh = np.load(gt_path)
            if gt_oh.shape == onehot.shape:
                print(f"\n[sanity] {subject}: SynthSeg/GT foreground Dice = "
                      f"{_fg_dice(onehot, gt_oh):.4f} (shapes match {onehot.shape})")
            else:
                print(f"\n[sanity][WARN] {subject}: shape mismatch "
                      f"synthseg {onehot.shape} vs gt {gt_oh.shape} (not same grid)")
            sanity_done = True

    print(f"\nWrote {written} files. Missing SynthSeg for {len(missing)} subjects.")
    if missing:
        print("Missing:", missing[:10], "..." if len(missing) > 10 else "")


if __name__ == "__main__":
    main()
