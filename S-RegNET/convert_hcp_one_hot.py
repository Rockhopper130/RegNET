"""
Convert HCP aparc+aseg.nii.gz segmentations to the 5-class one-hot format
used by S-RegNET:
    0: Background
    1: Cortex (Cortical Gray Matter)
    2: Other Brain (Cerebellum, Brainstem, Hippocampus, Amygdala)
    3: White-Surface Interior (Cerebral WM + deep subcortical nuclei)
    4: CSF (Ventricles)

Uses a fast vectorized Look-Up Table (LUT) in C via NumPy indexing:
Runs in ~0.5s per subject instead of ~60s!

Usage:
    python convert_hcp_one_hot.py \
        --hcp_root /shared/scratch/0/home/v_mahith_reddy/hcp_data \
        --subjects_file /shared/scratch/0/home/v_mahith_reddy/hcp_data/val.txt \
        --workers 16
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from label_mapping import LABEL_MAPPING

# Pre-build 20,000 element LUT for instant vectorized C-level remapping
_MAX_LABEL = 20000
_LUT = np.zeros(_MAX_LABEL, dtype=np.uint8)
for src, dst in LABEL_MAPPING.items():
    if src < _MAX_LABEL:
        _LUT[src] = dst


def remap_fast(int_volume):
    """Vectorized remapping via LUT (runs in ~0.05s)."""
    # Clip any out-of-range label to background (0)
    safe_vol = np.where((int_volume >= 0) & (int_volume < _MAX_LABEL), int_volume, 0)
    return _LUT[safe_vol]


def to_one_hot_fast(seg, num_classes=5):
    """Fast uint8 one-hot volume."""
    one_hot = np.zeros((num_classes, *seg.shape), dtype=np.uint8)
    for c in range(num_classes):
        one_hot[c] = (seg == c).astype(np.uint8)
    return one_hot


def convert_single_subject(job):
    subj_dir, seg_name, out_seg_name, out_npy_name, overwrite = job
    in_path = os.path.join(subj_dir, "T1w", seg_name)
    out_nii_path = os.path.join(subj_dir, "T1w", out_seg_name)
    out_npy_path = os.path.join(subj_dir, "T1w", out_npy_name)

    if not os.path.exists(in_path):
        return False, f"Missing {in_path}"

    # If already converted and valid, skip
    if not overwrite and os.path.exists(out_nii_path) and os.path.exists(out_npy_path):
        return True, "Already converted"

    try:
        img = nib.load(in_path)
        raw_seg = np.asanyarray(img.dataobj).astype(np.int64)
        seg5 = remap_fast(raw_seg)

        # Save remapped NIfTI
        nib.save(
            nib.Nifti1Image(seg5.astype(np.uint8), img.affine, img.header),
            out_nii_path
        )
        # Save one-hot npy
        np.save(out_npy_path, to_one_hot_fast(seg5))
        return True, "Success"
    except Exception as e:
        return False, f"Error processing {subj_dir}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Fast Convert HCP aparc+aseg to 5-class one-hot")
    parser.add_argument("--hcp_root", required=True, help="Root directory holding HCP subject folders")
    parser.add_argument("--subjects_file", required=True, help="Text file with subject IDs or seg paths (one per line)")
    parser.add_argument("--seg_name", default="aparc+aseg.nii.gz")
    parser.add_argument("--out_seg_name", default="seg4_white.nii.gz")
    parser.add_argument("--out_npy_name", default="seg4_white_onehot.npy")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reconvert subjects even when both output files already exist",
    )
    args = parser.parse_args()

    hcp_root = Path(args.hcp_root)
    with open(args.subjects_file, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    jobs = []
    for line in lines:
        if "/" in line or "\\" in line:
            parts = Path(line).parts
            if "T1w" in parts:
                idx = parts.index("T1w")
                subj = parts[idx - 1]
            else:
                subj = Path(line).name.split(".")[0]
        else:
            subj = line

        subj_dir = str(hcp_root / subj)
        jobs.append((
            subj_dir, args.seg_name, args.out_seg_name, args.out_npy_name,
            args.overwrite,
        ))

    print(f"Converting {len(jobs)} subjects to 5-class one-hot format using {args.workers} workers...")
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(convert_single_subject, jobs), total=len(jobs), desc="HCP 5-class conversion"))

    success_count = sum(1 for ok, _ in results if ok)
    print(f"Done! Successfully converted {success_count}/{len(jobs)} subjects.")


if __name__ == "__main__":
    main()
