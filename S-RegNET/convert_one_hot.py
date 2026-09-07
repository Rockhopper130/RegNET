"""
Convert the GT 5-class segmentation of every listed subject (seg4.nii.gz,
labels 0 bg / 1 cortex / 2 subcortical GM / 3 white matter / 4 CSF) into a
5-channel uint8 one-hot .npy next to it. Native resolution; SegDataset
resizes at load time. convert_synthseg_one_hot.py is the SynthSeg twin.

Usage:
    python convert_one_hot.py --lists /path/train.txt /path/val.txt

Each list line is a subject's seg4_onehot.npy path (the train.txt/val.txt
format); the .nii.gz is read from, and the .npy written to, that directory.
"""

import argparse
import os

import numpy as np
import nibabel as nib
from tqdm import tqdm

LABELS = [0, 1, 2, 3, 4]


def to_one_hot(seg, labels=LABELS):
    one_hot = np.zeros((len(labels), *seg.shape), dtype=np.uint8)
    for i, label in enumerate(labels):
        one_hot[i] = seg == label
    return one_hot


def main():
    ap = argparse.ArgumentParser(description="GT seg .nii.gz -> 5-class one-hot .npy")
    ap.add_argument('--lists', nargs='+', required=True,
                    help='train/val txt files; each line = a subject seg4_onehot.npy path')
    ap.add_argument('--seg_name', default='seg4.nii.gz')
    ap.add_argument('--out_name', default='seg4_onehot.npy')
    args = ap.parse_args()

    paths = []
    for lst in args.lists:
        with open(lst) as f:
            paths += [ln.strip() for ln in f if ln.strip()]

    for p in tqdm(paths, desc="Converting GT segs", ncols=80):
        subject_dir = os.path.dirname(p)
        seg = nib.load(os.path.join(subject_dir, args.seg_name)).get_fdata().astype(np.int16)
        np.save(os.path.join(subject_dir, args.out_name), to_one_hot(seg))


if __name__ == "__main__":
    main()
