"""
Segmentation dataset for registration.

Serves, per subject: the shared template seg, the GT one-hot, and the SynthSeg
one-hot. All are resized to target_size with mode='nearest' so they stay
one-hot. No spatial augmentation — the template/sample misalignment is the
registration signal.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import yaml


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


class SegDataset(Dataset):
    """Returns a dict per subject:
        template_seg : (5, D, H, W) one-hot, shared across samples
        synthseg_seg : (5, D, H, W) one-hot SynthSeg prediction
        gt_seg       : (5, D, H, W) one-hot ground truth
    """

    def __init__(
        self,
        data_list_file,
        template_seg_path,
        target_size=(128, 128, 128),
        seg_filename="seg4_onehot.npy",
        synthseg_filename="synthseg_onehot.npy",
        preload=True,
    ):
        with open(data_list_file, 'r') as f:
            seg_paths = f.read().splitlines()

        self.subject_dirs = [os.path.dirname(p) for p in seg_paths]
        self.seg_filename = seg_filename
        self.synthseg_filename = synthseg_filename
        self.target_size = target_size

        self.template_seg = self._load_seg(template_seg_path, target_size)

        self._gt_cache = None
        self._synthseg_cache = None
        if preload:
            self._preload_all()

    def _preload_all(self):
        from tqdm import tqdm
        n = len(self.subject_dirs)
        print(f"Preloading {n} GT + SynthSeg seg volumes into RAM...")
        self._gt_cache, self._synthseg_cache = [], []
        for subject_dir in tqdm(self.subject_dirs, desc="Preloading", ncols=80):
            self._gt_cache.append(
                self._load_seg(os.path.join(subject_dir, self.seg_filename), self.target_size))
            self._synthseg_cache.append(
                self._load_seg(os.path.join(subject_dir, self.synthseg_filename), self.target_size))
        print(f"Preloading complete. RAM cached {n} GT + {n} SynthSeg volumes.")

    def _load_seg(self, path, target_size):
        seg = np.load(path)
        seg = torch.tensor(seg, dtype=torch.float32)
        seg = F.interpolate(seg.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)
        # Valid only because every load/warp here uses nearest interpolation;
        # bilinear/trilinear anywhere upstream would make voxels sum to <1.
        assert torch.all(torch.sum(seg, dim=0) == 1), f"Invalid one-hot at {path}"
        return seg

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        if self._gt_cache is not None:
            gt_seg = self._gt_cache[idx]
            synthseg_seg = self._synthseg_cache[idx]
        else:
            subject_dir = self.subject_dirs[idx]
            gt_seg = self._load_seg(
                os.path.join(subject_dir, self.seg_filename), self.target_size)
            synthseg_seg = self._load_seg(
                os.path.join(subject_dir, self.synthseg_filename), self.target_size)

        return {
            'template_seg': self.template_seg,
            'synthseg_seg': synthseg_seg,
            'gt_seg': gt_seg,
        }


if __name__ == "__main__":
    config = load_config()
    if config is None:
        raise Exception("Config file not found. Check config.yaml.")

    data_list = config['data']['train_txt']
    template = config['data']['template_seg_path']
    target_size = tuple(config['model']['target_size'])

    if os.path.exists(data_list) and os.path.exists(template):
        print(f"Loading dataset from config.yaml...")
        print(f"  Data list: {data_list}")
        print(f"  Template:  {template}")
        print(f"  Target size: {target_size}")
        dataset = SegDataset(data_list, template, target_size=target_size, preload=False)
        print(f"Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"Template seg shape: {sample['template_seg'].shape}")
        print(f"SynthSeg seg shape: {sample['synthseg_seg'].shape}")
        print(f"GT seg shape:       {sample['gt_seg'].shape}")
        for k in ('synthseg_seg', 'gt_seg'):
            s = sample[k].sum(0)
            print(f"{k} sum-per-voxel range: [{s.min().item()}, {s.max().item()}] (should be 1.0)")
    else:
        print("Test data not found. Check paths in config.yaml.")
