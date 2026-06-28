"""
Training loop: warp the template seg onto each sample with voxel multi-class
Dice, driving a bounded B-spline FFD (affine optional, then the FFD cascade).

The FFD clamp keeps the warp a diffeomorphism, so topology is not enforced by
the loss. Folding %, mesh triangle-flip %, and the mesh-inversion residual are
logged each epoch as diagnostics (all expected near zero). Checkpoints are
selected on validation WM Dice.

Usage:
    python train.py                    # use default config.yaml
    python train.py --config my.yaml   # use a custom config
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import logging
import time
import signal
from contextlib import contextmanager
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from model import SegRegistrationNet, SpatialTransformer
from losses import SegRegistrationLoss, compute_dice_score, jacobian_det
from get_data import SegDataset
from wm_template import load_template_wm_mesh, triangle_flip_fraction
from experiment_routing import build_step_inputs
from git_provenance import write_git_sha


# Normalized distance -> mm: 1 unit = half of the 256 mm conformed FOV.
MM_PER_NORM = 256.0 / 2.0
WM_CLASS = 3   # 5-class one-hot: 0 bg, 1 cortex, 2 subGM, 3 WM, 4 CSF


# =============================================================================
# Configuration
# =============================================================================

def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Config:
    """Training configuration. Loads from config.yaml by default."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        if Path(config_path).exists():
            self._load_from_yaml(load_yaml_config(config_path))
        else:
            raise FileNotFoundError(f"config.yaml not found at {config_path}")

    def _load_from_yaml(self, cfg):
        # Paths
        self.train_txt = cfg['data']['train_txt']
        self.val_txt = cfg['data']['val_txt']
        self.template_seg_path = cfg['data']['template_seg_path']
        self.seg_filename = cfg['data'].get('seg_filename', 'seg4_onehot.npy')
        self.synthseg_filename = cfg['data'].get('synthseg_filename', 'synthseg_onehot.npy')
        self.template_wm_mesh_path = cfg['data']['template_wm_mesh_path']

        # Output
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = cfg['output']['base_dir']
        self.output_dir = f"{base_dir}/{timestamp}"
        self.checkpoint_dir = os.path.join(self.output_dir, cfg['output']['checkpoint_subdir'])
        self.log_dir = os.path.join(self.output_dir, cfg['output']['log_subdir'])

        # Model
        self.target_size = tuple(cfg['model']['target_size'])
        self.num_classes = cfg['model']['num_classes']
        self.seg_channels = cfg['model'].get('seg_channels', self.num_classes)

        # Transform (bounded B-spline FFD)
        tcfg = cfg.get('transform', {})
        self.cp_spacing = tcfg.get('cp_spacing', 8)
        self.n_stages = tcfg.get('n_stages', 1)
        self.injectivity_k = tcfg.get('injectivity_k', 0.40)

        # Affine
        affine_cfg = cfg.get('affine', {})
        self.use_affine = affine_cfg.get('enabled', False)

        # Experiment routing. input_source = model's 2nd input (synthseg default);
        # supervision_target = the Dice/CE target seg.
        exp = cfg.get('experiment', {})
        self.input_source = exp.get('input_source', 'synthseg')
        self.supervision_target = exp.get('supervision_target', 'gt')
        assert self.input_source in ('synthseg', 'gt'), \
            f"experiment.input_source must be 'synthseg' or 'gt', got {self.input_source!r}"
        assert self.supervision_target in ('gt', 'synthseg'), \
            f"experiment.supervision_target must be 'gt' or 'synthseg', got {self.supervision_target!r}"

        # Training
        self.batch_size = cfg['training']['batch_size']
        self.num_epochs = cfg['training']['num_epochs']
        self.num_workers = cfg['training']['num_workers']
        self.pin_memory = cfg['training']['pin_memory']

        self.lr = cfg['training']['learning_rate']
        self.min_lr = cfg['training']['min_learning_rate']
        self.weight_decay = cfg['training']['weight_decay']
        self.warmup_epochs = cfg['training']['warmup_epochs']

        self.use_amp = cfg['training']['use_amp']

        self.save_every = cfg['training']['save_every']
        self.patience = cfg['training']['patience']
        self.max_grad_norm = cfg['training'].get('max_grad_norm', 1.0)

        # Device
        self.device = cfg['device']['gpu'] if torch.cuda.is_available() else "cpu"
        self.cudnn_benchmark = cfg['device'].get('cudnn_benchmark', True)

        # Resume
        self.resume_from = cfg['resume']['checkpoint_path']

        # Reproducibility
        self.seed = cfg.get('seed', 42)

        # Pull the per-class weight vector out of the term-weight dict so the
        # latter stays scalar-only (the loss-component logger floats every value).
        loss_cfg = dict(cfg['loss'])
        self.class_weights = loss_cfg.pop('class_weights', None)
        self.loss_weights = loss_cfg
        if self.use_affine:
            self.loss_weights.setdefault('affine_reg', affine_cfg.get('regularization_weight', 0.01))

    def save(self, path):
        config_dict = {k: str(v) if isinstance(v, Path) else v
                       for k, v in self.__dict__.items()}
        config_dict['target_size'] = list(self.target_size)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logger(log_dir, name="wm_mesh_registration"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_file


# =============================================================================
# Learning Rate Scheduler with Warmup
# =============================================================================

class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# =============================================================================
# NFS-safe Checkpoint Save
# =============================================================================

@contextmanager
def _timeout(seconds):
    def _handler(signum, frame):
        raise TimeoutError(f"I/O timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def log_loss_components_artifact(epoch, train_components, val_components,
                                  weights, output_dir, logger=None):
    """Append the per-epoch raw + weighted loss-component breakdown to
    loss_components.jsonl and log the top weighted contributors."""
    def weighted_view(components):
        return {k: float(weights.get(k, 0.0)) * float(v) for k, v in components.items()}

    train_w = weighted_view(train_components)
    val_w = weighted_view(val_components)

    record = {
        'epoch': epoch,
        'weights': {k: float(v) for k, v in weights.items()},
        'train': {'raw': {k: float(v) for k, v in train_components.items()}, 'weighted': train_w},
        'val':   {'raw': {k: float(v) for k, v in val_components.items()},   'weighted': val_w},
    }

    path = Path(output_dir) / "loss_components.jsonl"
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')

    if logger is not None:
        total = sum(abs(v) for v in val_w.values()) or 1.0
        ranked = sorted(val_w.items(), key=lambda kv: -abs(kv[1]))[:5]
        parts = [f"{k}={v:.4g} ({100 * abs(v) / total:.1f}%)" for k, v in ranked]
        logger.info(f"  Val   - Top contributors: {' | '.join(parts)}")


def safe_save(obj, path, logger=None, timeout=300):
    """torch.save with a hard timeout so NFS hangs don't kill the run."""
    try:
        with _timeout(timeout):
            torch.save(obj, path)
    except TimeoutError as e:
        if logger:
            logger.warning(f"Checkpoint save timed out ({path}): {e} — skipping")
    except Exception as e:
        if logger:
            logger.warning(f"Checkpoint save failed ({path}): {e} — skipping")


# =============================================================================
# Warp path + out-of-loss diagnostics
# =============================================================================

def _warp_template(template_seg, flow, affine_matrix, stn):
    """Warp the template: affine first, then the dense FFD field through the
    SpatialTransformer (used as a pull field). Bilinear so Dice has a gradient.
    inference.py and the eval scripts must replay this same affine-then-flow order.
    """
    if affine_matrix is not None:
        affine_grid = F.affine_grid(affine_matrix, template_seg.size(), align_corners=False)
        aligned_seg = F.grid_sample(template_seg, affine_grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=False)
        return stn(aligned_seg, flow)
    return stn(template_seg, flow)


def _folding_pct(flow, det_ref):
    """Voxel-folding % of the dense field (diagnostic; ~0 by construction)."""
    det = jacobian_det(flow) / det_ref
    return (det < 0).float().mean().item() * 100.0


def _mesh_push_diagnostics(model, template_verts, cps_list, affine_matrix, faces_np):
    """Push the genus-0 template mesh into subject space via the field's numerical
    inverse and measure triangle-flip % and the per-vertex inversion residual (mm).
    Returns (flip_pct, res_mean_mm, res_max_mm)."""
    pushed, res = model.push_points_to_sample(template_verts, cps_list, affine_matrix)
    flip = triangle_flip_fraction(
        template_verts.detach().cpu().numpy(),
        pushed.detach().cpu().numpy(), faces_np) * 100.0
    return flip, res.mean().item() * MM_PER_NORM, res.max().item() * MM_PER_NORM


# =============================================================================
# Training / Validation
# =============================================================================

def train_epoch(model, stn, dataloader, loss_fn, optimizer, device, epoch, config):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    loss_components_sum = {}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs} [Train]",
                leave=False, ncols=100)

    for batch in pbar:
        template_seg = batch['template_seg'].to(device)
        synthseg_seg = batch['synthseg_seg'].to(device)
        gt_seg = batch['gt_seg'].to(device)

        input_seg, target_seg = build_step_inputs(
            synthseg_seg, gt_seg,
            input_source=config.input_source, supervision_target=config.supervision_target)

        optimizer.zero_grad()

        cps_list, affine_matrix = model(template_seg, input_seg)
        flow = model.dense_flow_from_cps(cps_list)
        warped_seg = _warp_template(template_seg, flow, affine_matrix, stn)
        loss, loss_dict = loss_fn(warped_seg, target_seg, cps_list,
                                  affine_matrix=affine_matrix, return_components=True)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_dice += 1 - loss_dict['dice'].item()
        for k, v in loss_dict.items():
            loss_components_sum[k] = loss_components_sum.get(k, 0) + (
                v.item() if isinstance(v, torch.Tensor) else v)

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'dice': f'{1 - loss_dict["dice"].item():.4f}'})

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'dice': total_dice / n,
        'components': {k: v / n for k, v in loss_components_sum.items()},
    }


@torch.no_grad()
def validate_epoch(model, stn, dataloader, loss_fn, device, epoch, config, template_verts, faces_np):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0          # foreground-mean Dice — the checkpoint metric
    total_folding = 0.0
    total_flip = 0.0
    worst_flip = 0.0
    total_res_mean = 0.0
    worst_res_max = 0.0
    dice_per_class_sum = [0.0] * config.num_classes
    loss_components_sum = {}

    D, H, W = config.target_size
    det_ref = (2.0 / D) * (2.0 / H) * (2.0 / W)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs} [Val]",
                leave=False, ncols=100)

    for batch in pbar:
        template_seg = batch['template_seg'].to(device)
        synthseg_seg = batch['synthseg_seg'].to(device)
        gt_seg = batch['gt_seg'].to(device)

        input_seg, target_seg = build_step_inputs(
            synthseg_seg, gt_seg,
            input_source=config.input_source, supervision_target=config.supervision_target)

        cps_list, affine_matrix = model(template_seg, input_seg)
        flow = model.dense_flow_from_cps(cps_list)
        warped_seg = _warp_template(template_seg, flow, affine_matrix, stn)

        loss, loss_dict = loss_fn(warped_seg, target_seg, cps_list,
                                  affine_matrix=affine_matrix, return_components=True)
        total_loss += loss.item()

        # Reported Dice is the unweighted hard-label fg mean (comparable across
        # runs); the loss 'dice' term is WM-weighted, so it isn't reused here.
        dice_per_class, fg_mean = compute_dice_score(warped_seg, target_seg, config.num_classes)
        total_dice += fg_mean
        for c in range(config.num_classes):
            dice_per_class_sum[c] += dice_per_class[c]

        # Topology diagnostics (not optimised; expected near zero).
        total_folding += _folding_pct(flow, det_ref)
        flip, res_mean_mm, res_max_mm = _mesh_push_diagnostics(
            model, template_verts, cps_list, affine_matrix, faces_np)
        total_flip += flip
        worst_flip = max(worst_flip, flip)
        total_res_mean += res_mean_mm
        worst_res_max = max(worst_res_max, res_max_mm)

        for k, v in loss_dict.items():
            loss_components_sum[k] = loss_components_sum.get(k, 0) + (
                v.item() if isinstance(v, torch.Tensor) else v)

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'dice': f'{1 - loss_dict["dice"].item():.4f}'})

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'dice': total_dice / n,
        'dice_per_class': [s / n for s in dice_per_class_sum],
        'folding_pct': total_folding / n,
        'flip_pct': total_flip / n,
        'worst_flip_pct': worst_flip,
        'inv_res_mean_mm': total_res_mean / n,
        'worst_inv_res_max_mm': worst_res_max,
        'components': {k: v / n for k, v in loss_components_sum.items()},
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='WM-mesh Registration Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (default: config.yaml in script directory)')
    args = parser.parse_args()

    config = Config(args.config)

    assert config.batch_size == 1, (
        "Training assumes batch_size 1: the mesh-push diagnostics replay the "
        "affine-on-points inverse, which is written for B=1 (_apply_affine_inverse).")
    assert not config.use_amp, (
        "use_amp must be false: this loop uses plain fp32 backward (no GradScaler). "
        "To enable AMP, add a GradScaler and unscale before clip_grad_norm_.")

    # Reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = not config.cudnn_benchmark

    for dir_path in [config.output_dir, config.checkpoint_dir, config.log_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logger(config.log_dir)

    logger.info("=" * 70)
    logger.info("Seg Registration (voxel Dice on bounded B-spline FFD)")
    logger.info("=" * 70)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Device: {config.device}")

    write_git_sha(config.output_dir, logger)
    config.save(os.path.join(config.output_dir, "config.json"))

    device = torch.device(config.device)
    if 'cuda' in config.device:
        logger.info(f"GPU: {torch.cuda.get_device_name()}")

    # ==========================================================================
    # Data
    # ==========================================================================
    logger.info("Loading datasets...")
    train_dataset = SegDataset(
        config.train_txt, config.template_seg_path,
        target_size=config.target_size,
        seg_filename=config.seg_filename, synthseg_filename=config.synthseg_filename,
    )
    val_dataset = SegDataset(
        config.val_txt, config.template_seg_path,
        target_size=config.target_size,
        seg_filename=config.seg_filename, synthseg_filename=config.synthseg_filename,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Template mesh (genus-0; used only by the mesh-push diagnostics, not the loss)
    logger.info(f"Loading template WM mesh: {config.template_wm_mesh_path}")
    verts_np, faces_np, n_lh = load_template_wm_mesh(config.template_wm_mesh_path)
    template_verts = torch.tensor(verts_np, dtype=torch.float32, device=device)   # (N, 3)
    logger.info(f"Mesh: {len(template_verts):,} verts ({n_lh:,} lh) | {len(faces_np):,} faces")

    # ==========================================================================
    # Model + loss + warp transformer
    # ==========================================================================
    logger.info("Initializing model...")
    model = SegRegistrationNet(
        seg_channels=config.seg_channels, use_affine=config.use_affine,
        cp_spacing=config.cp_spacing, n_stages=config.n_stages,
        injectivity_k=config.injectivity_k, target_size=config.target_size,
    ).to(device)
    stn = SpatialTransformer(config.target_size, device=device).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Affine pre-alignment: {'ENABLED' if config.use_affine else 'DISABLED'}")
    logger.info(f"Transform: cp_spacing={config.cp_spacing} n_stages={config.n_stages} "
                f"injectivity_k={config.injectivity_k}")
    logger.info(f"Experiment: input={config.input_source} | "
                f"supervision_target={config.supervision_target}")

    loss_fn = SegRegistrationLoss(config.loss_weights, class_weights=config.class_weights).to(device)
    if config.class_weights is not None:
        logger.info(f"Class weights [bg,cortex,subGM,WM,CSF]: {config.class_weights} "
                    f"(WM-focused supervision; checkpoint selects on WM Dice)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, config.warmup_epochs,
                                      config.num_epochs, config.min_lr)
    logger.info(f"Optimizer: AdamW (lr={config.lr}, weight_decay={config.weight_decay})")

    # Resume — best metric is MAX validation Dice.
    start_epoch = 1
    best_dice = 0.0
    if config.resume_from and os.path.exists(config.resume_from):
        logger.info(f"Resuming from: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        logger.info(f"Resumed from epoch {start_epoch - 1}, best Dice: {best_dice:.4f}")

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    epochs_without_improvement = 0
    training_start = time.time()

    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start = time.time()
        current_lr = scheduler.step(epoch - 1)

        train_metrics = train_epoch(
            model, stn, train_loader, loss_fn, optimizer, device, epoch, config)
        val_metrics = validate_epoch(
            model, stn, val_loader, loss_fn, device, epoch, config, template_verts, faces_np)

        epoch_time = time.time() - epoch_start

        wm_dice = val_metrics['dice_per_class'][WM_CLASS]
        logger.info("-" * 70)
        logger.info(f"Epoch {epoch}/{config.num_epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.5f} | Dice: {train_metrics['dice']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.5f} | Dice: {val_metrics['dice']:.4f} "
                    f"(WM {wm_dice:.4f})")
        logger.info(f"  Val   - Folding: {val_metrics['folding_pct']:.4f}% | "
                    f"Mesh flips: {val_metrics['flip_pct']:.4f}% (worst {val_metrics['worst_flip_pct']:.4f}%) | "
                    f"Inv-residual: {val_metrics['inv_res_mean_mm']:.3f} mm "
                    f"(worst {val_metrics['worst_inv_res_max_mm']:.3f})  [~0 expected]")

        log_loss_components_artifact(
            epoch, train_metrics['components'], val_metrics['components'],
            loss_fn.weights, config.output_dir, logger)

        # Select on WM Dice (the target structure), not the 4-class mean.
        is_best = wm_dice > best_dice
        if is_best:
            best_dice = wm_dice
            epochs_without_improvement = 0
            logger.info(f"  New best model! Val WM Dice: {best_dice:.4f}")
        else:
            epochs_without_improvement += 1

        if is_best or epoch % config.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_dice': val_metrics['dice'],
                'val_dice_per_class': val_metrics['dice_per_class'],
                'val_folding_pct': val_metrics['folding_pct'],
                'val_flip_pct': val_metrics['flip_pct'],
                'best_dice': best_dice,
            }
            if is_best:
                safe_save(checkpoint, os.path.join(config.checkpoint_dir, "best_model.pth"), logger)
            if epoch % config.save_every == 0:
                safe_save(checkpoint,
                          os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth"),
                          logger)

        if epochs_without_improvement >= config.patience:
            logger.warning(f"Early stopping! No improvement for {config.patience} epochs.")
            break

        if 'cuda' in config.device:
            torch.cuda.empty_cache()

    total_time = time.time() - training_start
    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time / 3600:.2f} hours")
    logger.info(f"Best validation WM Dice: {best_dice:.4f}")
    logger.info(f"Checkpoints: {config.checkpoint_dir}")
    logger.info(f"Logs: {log_file}")


if __name__ == "__main__":
    main()
