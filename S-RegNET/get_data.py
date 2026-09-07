"""
Segmentation Dataset for Registration

Loads:
- Template segmentation (5 channels, one-hot)
- Sample segmentation  (5 channels, one-hot) -- the supervision target
- Optionally a second sample segmentation used as the network input
  instead of the target (the deployment condition: SynthSeg in, GT out)

Resizes to a fixed target_size using mode='nearest' so the seg stays
strictly one-hot. No spatial augmentation here — misalignment between
template and sample is the registration signal.

File layout expected (per subject directory):
    seg4_onehot.npy         — shape (5, D, H, W) one-hot uint8/float
    synthseg_onehot_v1.npy  — same shape; only read when
                              input_seg_filename is set
    distill_vel_v1.npy      — shape (3, D, H, W) float32 velocity; only read
                              when target_vel_filename is set, and optional
                              per subject

Configuration:
    Reads config.yaml for default paths and target_size when run as a
    script. The Dataset class itself takes explicit arguments.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
import yaml
from pathlib import Path


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


class SegDataset(Dataset):
    """
    Dataset for segmentation-only registration.

    Returns dict with:
        - template_seg: (5, D, H, W) one-hot (shared across all samples)
        - sample_seg:   (5, D, H, W) one-hot — the SUPERVISION TARGET
        - input_seg:    (5, D, H, W) one-hot — the network's 2nd INPUT
        - target_vel, has_target_vel — only when target_vel_filename is set

    `input_seg` is the very same tensor as `sample_seg` unless
    `input_seg_filename` is given. That one knob is the whole deployment
    condition: the SynthSeg seg reaches the network, while every loss term
    (forward Dice/CE, the reverse branch, the λ prior) still scores against
    the GT seg — so the net is trained to denoise its input, not reproduce it.
    """

    def __init__(
        self,
        data_list_file: str,
        template_seg_path: str,
        target_size=(128, 128, 128),
        seg_filename="seg4_onehot.npy",
        input_seg_filename=None,
        target_vel_filename=None,
        preload=True,
    ):
        """
        Args:
            data_list_file: text file with one path per line; each path
                points at a subject's seg .npy. Subject dir is inferred
                via os.path.dirname.
            template_seg_path: absolute path to the template seg .npy.
            target_size: (D, H, W) resize target.
            seg_filename: supervision-target seg inside each subject dir.
            input_seg_filename: optional second seg in the same dir, used
                as the network input in place of `seg_filename` (e.g.
                "synthseg_onehot_v1.npy"). None reuses the target.
            target_vel_filename: optional per-subject distillation target
                (e.g. "distill_vel_v1.npy", written by
                bandlimit_opt/generate_distill_targets.py). Unlike the segs
                this one is optional PER SUBJECT — a subject without it is
                kept, and its `has_target_vel` flag is False.
            preload: if True, load every subject's seg into RAM at init
                to eliminate per-epoch I/O on slow filesystems.
        """
        with open(data_list_file, 'r') as f:
            seg_paths = f.read().splitlines()

        self.subject_dirs = [os.path.dirname(p) for p in seg_paths]
        self.seg_filename = seg_filename
        self.input_seg_filename = input_seg_filename
        self.target_vel_filename = target_vel_filename
        self.target_size = target_size

        self.template_seg = self._load_seg(template_seg_path, target_size)

        if input_seg_filename is not None:
            self._require(input_seg_filename)

        if target_vel_filename is not None:
            # Not _require: target generation legitimately drops subjects
            # (diverged fits), and the regression term simply skips them.
            n = sum(os.path.exists(os.path.join(d, target_vel_filename))
                    for d in self.subject_dirs)
            print(f"Distillation targets: {n}/{len(self.subject_dirs)} subjects "
                  f"have {target_vel_filename}")

        self._seg_cache = None
        self._input_cache = None
        self._vel_cache = None
        if preload:
            self._preload_all()

    def _require(self, filename):
        """Fail at construction, not 40 minutes into a preload."""
        missing = [d for d in self.subject_dirs
                   if not os.path.exists(os.path.join(d, filename))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(self.subject_dirs)} subjects have no "
                f"{filename} (first: {missing[:3]})")

    def _preload_all(self):
        """Load every sample seg into RAM once.

        Cached as uint8: the segs are strictly one-hot (asserted below), so
        the cast is lossless, and float32 would cost 42 MB per volume — a
        second stream at that size is ~14 GB of RAM for OASIS.
        """
        from tqdm import tqdm
        n = len(self.subject_dirs)
        streams = 2 if self.input_seg_filename is not None else 1
        print(f"Preloading {n} x {streams} seg volumes into RAM...")
        self._seg_cache = []
        self._input_cache = [] if streams == 2 else None
        self._vel_cache = [] if self.target_vel_filename is not None else None
        for subject_dir in tqdm(self.subject_dirs, desc="Preloading", ncols=80):
            self._seg_cache.append(self._cached(subject_dir, self.seg_filename))
            if self._input_cache is not None:
                self._input_cache.append(
                    self._cached(subject_dir, self.input_seg_filename))
            if self._vel_cache is not None:
                self._vel_cache.append(self._load_vel(subject_dir))
        print(f"Preloading complete. RAM cached {n * streams} seg volumes.")

    def _cached(self, subject_dir, filename):
        seg = self._load_seg(os.path.join(subject_dir, filename), self.target_size)
        return seg.to(torch.uint8)

    def _load_vel(self, subject_dir):
        """Distillation target as float16, or None when the subject has none.

        float16 halves the cache (3 x 128³ fp32 is 25 MB per subject, ~8 GB
        over the training set) and its ~1e-3 relative error is far below one
        voxel after the velocity is integrated.
        """
        path = os.path.join(subject_dir, self.target_vel_filename)
        if not os.path.exists(path):
            return None
        return torch.tensor(np.load(path), dtype=torch.float16)

    def _load_seg(self, path, target_size):
        """Load and preprocess a one-hot seg volume."""
        seg = np.load(path)
        seg = torch.tensor(seg, dtype=torch.float32)

        seg = F.interpolate(
            seg.unsqueeze(0),
            size=target_size,
            mode='nearest',
        ).squeeze(0)

        # One-hot integrity check. Valid only because every load/warp
        # path here uses mode='nearest'; switching to bilinear/trilinear
        # anywhere upstream would make voxels sum to <1 and trip this.
        assert torch.all(torch.sum(seg, dim=0) == 1), f"Invalid one-hot at {path}"
        return seg

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        if self._seg_cache is not None:
            sample_seg = self._seg_cache[idx].float()
            input_seg = (self._input_cache[idx].float()
                         if self._input_cache is not None else sample_seg)
        else:
            subject_dir = self.subject_dirs[idx]
            sample_seg = self._load_seg(os.path.join(subject_dir, self.seg_filename),
                                        self.target_size)
            input_seg = sample_seg if self.input_seg_filename is None else \
                self._load_seg(os.path.join(subject_dir, self.input_seg_filename),
                               self.target_size)

        item = {
            'template_seg': self.template_seg,
            'sample_seg': sample_seg,
            'input_seg': input_seg,
        }
        if self.target_vel_filename is not None:
            item['target_vel'], item['has_target_vel'] = self._target_vel(idx)
        return item

    def _target_vel(self, idx):
        """(velocity, flag) for the distillation target, cast back to float32.

        A subject without a target gets a zero field so the batch still
        collates at any batch size; the flag is what the train step reads, so
        those zeros never reach the regression term.
        """
        vel = (self._vel_cache[idx] if self._vel_cache is not None
               else self._load_vel(self.subject_dirs[idx]))
        if vel is None:
            return torch.zeros(3, *self.target_size), False
        return vel.float(), True


# =============================================================================
# Test
# =============================================================================

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
        print()

        dataset = SegDataset(data_list, template, target_size=target_size,
                             seg_filename=config['data'].get('seg_filename',
                                                             'seg4_onehot.npy'),
                             input_seg_filename=config['data'].get('input_seg_filename'),
                             target_vel_filename=config['data'].get('target_vel_filename'),
                             preload=False)
        print(f"Dataset size: {len(dataset)}")

        sample = dataset[0]
        print(f"Template seg shape: {sample['template_seg'].shape}")
        print(f"Sample seg shape:   {sample['sample_seg'].shape}  (target)")
        print(f"Input seg shape:    {sample['input_seg'].shape}  (net input; "
              f"same tensor as target: "
              f"{sample['input_seg'] is sample['sample_seg']})")
        for k in ('sample_seg', 'input_seg'):
            s = sample[k].sum(0)
            print(f"{k} sum-per-voxel range: "
                  f"[{s.min().item()}, {s.max().item()}] (should be 1.0)")
        if 'target_vel' in sample:
            v = sample['target_vel']
            print(f"Target vel shape:   {tuple(v.shape)} {v.dtype}  "
                  f"(present: {sample['has_target_vel']}, "
                  f"|v| max {v.abs().max().item():.4f})")
    else:
        print("Test data not found. Check paths in config.yaml.")
