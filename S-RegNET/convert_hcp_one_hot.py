"""
Convert HCP aparc+aseg.nii.gz segmentations to the 5-class one-hot format
used by S-RegNET:
    0: Background
    1: Cortex (Cortical Gray Matter)
    2: Other Brain (Cerebellum, Brainstem, Hippocampus, Amygdala)
    3: White-Surface Interior (Cerebral WM + deep subcortical nuclei)
    4: CSF (Ventricles)

Reads from:
    <subject_dir>/T1w/aparc+aseg.nii.gz
Writes to:
    <subject_dir>/T1w/seg4_white.nii.gz
    <subject_dir>/T1w/seg4_white_onehot.npy (5, D, H, W uint8)

Usage:
    python convert_hcp_one_hot.py \
        --hcp_root /shared/scratch/0/home/v_mahith_reddy/hcp_data \
        --subjects_file /shared/scratch/0/home/v_mahith_reddy/subjects.txt \
        --workers 8
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

# Add parent directory for utils
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from label_mapping import remap_to_5class

LABELS = [0, 1, 2, 3, 4]


def to_one_hot(seg, labels=LABELS):
    """(D, H, W) int labels -> (5, D, H, W) uint8 one-hot."""
    one_hot = np.zeros((len(labels), *seg.shape), dtype=np.uint8)
    for i, label in enumerate(labels):
        one_hot[i] = (seg == label).astype(np.uint8)
    return one_hot


def convert_single_subject(job):
    subj_dir, seg_name, out_seg_name, out_npy_name = job
    in_path = os.path.join(subj_dir, "T1w", seg_name)
    out_nii_path = os.path.join(subj_dir, "T1w", out_seg_name)
    out_npy_path = os.path.join(subj_dir, "T1w", out_npy_name)

    if not os.path.exists(in_path):
        return False, f"Missing {in_path}"

    # If already converted and not corrupt, skip
    if os.path.exists(out_nii_path) and os.path.exists(out_npy_path):
        return True, "Already converted"

    try:
        img = nib.load(in_path)
        raw_seg = np.asanyarray(img.dataobj).astype(np.int64)
        seg5 = remap_to_5class(raw_seg)

        # Save remapped NIfTI
        nib.save(
            nib.Nifti1Image(seg5.astype(np.uint8), img.affine, img.header),
            out_nii_path
        )
        # Save one-hot npy
        np.save(out_npy_path, to_one_hot(seg5))
        return True, "Success"
    except Exception as e:
        return False, f"Error processing {subj_dir}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Convert HCP aparc+aseg to 5-class one-hot")
    parser.add_argument("--hcp_root", required=True, help="Root directory holding HCP subject folders")
    parser.add_argument("--subjects_file", required=True, help="Text file with subject IDs (one per line)")
    parser.add_argument("--seg_name", default="aparc+aseg.nii.gz")
    parser.add_argument("--out_seg_name", default="seg4_white.nii.gz")
    parser.add_argument("--out_npy_name", default="seg4_white_onehot.npy")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    hcp_root = Path(args.hcp_root)
    with open(args.subjects_file, "r") as f:
        subjects = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    jobs = []
    for subj in subjects:
        subj_dir = str(hcp_root / subj)
        jobs.append((subj_dir, args.seg_name, args.out_seg_name, args.out_npy_name))

    print(f"Converting {len(jobs)} subjects to 5-class one-hot format...")
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(convert_single_subject, jobs), total=len(jobs), desc="HCP 5-class conversion"))

    success_count = sum(1 for ok, _ in results if ok)
    print(f"Done! Successfully converted {success_count}/{len(jobs)} subjects.")


if __name__ == "__main__":
    main()

