"""
Build ground-truth 5-class segmentation from FreeSurfer aseg.

This is the missing preprocessing step upstream of convert_one_hot.py: it takes a
FreeSurfer aseg (aseg.mgz / aseg.nii.gz), remaps the integer FreeSurfer labels to
the 5-class scheme (0 bg, 1 cortex, 2 subcortical GM, 3 WM, 4 CSF) via
label_mapping.remap_to_5class, and writes, next to each subject's scan dir:

    seg4.nii.gz        int16 5-class label volume (same affine/space as aseg;
                       used as the coordinate ref by wm_template.py)
    seg4_onehot.npy    (5, D, H, W) uint8 one-hot (the training target)

Input is a mapping file, one line per subject:  <scan_subject> <aseg_clinical_id>
e.g.  OASIS_OAS1_0025_MR1 OAS1_0025
aseg is read from   <aseg_root>/<aseg_clinical_id>_clinical/mri/aseg.mgz
outputs are written to <scans_root>/<scan_subject>/

Usage (run with FreeSurfer fspython, which has nibabel):
    fspython convert_aseg_gt.py --map map.txt \
        --aseg_root .../oasis_mri_outputs \
        --scans_root .../oasis_data/scans
"""

import argparse
import os

import numpy as np
import nibabel as nib
from tqdm import tqdm

from label_mapping import remap_to_5class

NUM_CLASSES = 5


def to_one_hot(seg, num_classes=NUM_CLASSES):
    """(D,H,W) int labels -> (num_classes, D,H,W) uint8 one-hot."""
    one_hot = np.zeros((num_classes, *seg.shape), dtype=np.uint8)
    for c in range(num_classes):
        one_hot[c] = (seg == c).astype(np.uint8)
    return one_hot


def main():
    ap = argparse.ArgumentParser(description="FreeSurfer aseg -> 5-class seg4.nii.gz + seg4_onehot.npy")
    ap.add_argument('--map', required=True,
                    help='mapping file; each line "<scan_subject> <aseg_clinical_id>"')
    ap.add_argument('--aseg_root', required=True,
                    help='root holding <aseg_clinical_id>_clinical/mri/aseg.mgz')
    ap.add_argument('--scans_root', required=True,
                    help='root holding <scan_subject>/ (outputs written here)')
    ap.add_argument('--aseg_name', default='aseg.mgz')
    args = ap.parse_args()

    pairs = []
    with open(args.map) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            scan_subject, clinical_id = ln.split()[:2]
            pairs.append((scan_subject, clinical_id))

    print(f"{len(pairs)} subjects to convert")
    written, missing = 0, []

    for scan_subject, clinical_id in tqdm(pairs, desc="aseg -> seg4", ncols=80):
        aseg_path = os.path.join(args.aseg_root, f"{clinical_id}_clinical", "mri", args.aseg_name)
        out_dir = os.path.join(args.scans_root, scan_subject)
        if not (os.path.exists(aseg_path) and os.path.isdir(out_dir)):
            missing.append(scan_subject)
            continue

        img = nib.load(aseg_path)
        aseg = np.asarray(img.dataobj).astype(np.int32)
        seg4 = remap_to_5class(aseg).astype(np.int16)

        # seg4.nii.gz in the aseg's native space (ref frame for wm_template.py).
        seg4_img = nib.Nifti1Image(seg4, img.affine, img.header)
        seg4_img.header.set_data_dtype(np.int16)
        nib.save(seg4_img, os.path.join(out_dir, "seg4.nii.gz"))

        # seg4_onehot.npy — the training target.
        np.save(os.path.join(out_dir, "seg4_onehot.npy"), to_one_hot(seg4))
        written += 1

    print(f"\nWrote {written} subjects. Missing aseg/scan-dir for {len(missing)}.")
    if missing:
        print("Missing:", missing[:10], "..." if len(missing) > 10 else "")


if __name__ == "__main__":
    main()
