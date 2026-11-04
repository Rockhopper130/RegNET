"""
Create train.txt and val.txt data splits for HCP S-RegNET.

Performs:
1. Scans HCP directory for all valid downloaded subjects.
2. Explicitly EXCLUDES the template subject (100206).
3. Verifies that essential meshes and segmentations exist for each candidate.
4. Splits into Train and Validation sets (e.g. 100 validation subjects, remainder training).
5. Writes train.txt and val.txt formatted as:
       <hcp_root>/<subject_id>/T1w/seg4_white_onehot.npy

Usage:
    python create_hcp_splits.py \
        --hcp_root /shared/scratch/0/home/v_mahith_reddy/hcp_data \
        --val_count 100 \
        --seed 42
"""

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create reproducible train/val splits for HCP")
    parser.add_argument("--hcp_root", default="/shared/scratch/0/home/v_mahith_reddy/hcp_data", help="HCP data root")
    parser.add_argument("--template_id", default="100206", help="Subject ID reserved as canonical template")
    parser.add_argument("--val_count", type=int, default=100, help="Number of validation subjects")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--out_dir", default=None, help="Directory to save train.txt and val.txt (defaults to hcp_root)")
    args = parser.parse_args()

    hcp_root = Path(args.hcp_root)
    out_dir = Path(args.out_dir) if args.out_dir else hcp_root

    subjs = []
    print(f"Scanning {hcp_root} for downloaded HCP subjects...")
    for p in sorted(hcp_root.iterdir()):
        if not p.is_dir() or p.name == args.template_id or p.name.startswith("template"):
            continue

        # Check required files exist
        has_seg = (p / "T1w" / "aparc+aseg.nii.gz").exists()
        has_pial = any((p / "Native_surf").glob("*pial.native.surf.gii"))

        if has_seg and has_pial:
            subjs.append(p.name)

    print(f"Found {len(subjs)} valid candidate subjects (template {args.template_id} excluded).")

    random.seed(args.seed)
    random.shuffle(subjs)

    val_subjs = sorted(subjs[:args.val_count])
    train_subjs = sorted(subjs[args.val_count:])

    print(f"Split: {len(train_subjs)} Train, {len(val_subjs)} Validation.")

    # Write files
    train_file = out_dir / "train.txt"
    val_file = out_dir / "val.txt"

    with open(train_file, "w") as f:
        for s in train_subjs:
            f.write(f"{hcp_root}/{s}/T1w/seg4_white_onehot.npy\n")

    with open(val_file, "w") as f:
        for s in val_subjs:
            f.write(f"{hcp_root}/{s}/T1w/seg4_white_onehot.npy\n")

    print(f"Saved: {train_file}")
    print(f"Saved: {val_file}")


if __name__ == "__main__":
    main()

