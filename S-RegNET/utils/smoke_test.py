"""
Smoke test (needs torch) for the bounded-FFD pipeline, on tiny synthetic data
so it runs without the dataset/mesh files. Checks:

  1. Zero control field leaves vertices unmoved and dense flow ~0 (coordinate parity).
  2. forward() returns the right control-lattice shape and respects the clamp.
  3. A full forward -> warp -> Dice loss -> backward is finite with non-zero grads.
  4. The n_stages>1 cascade-composition path runs.

Run:  python utils/smoke_test.py --device cuda:0   (or no flag for CPU)
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SegRegistrationNet, SpatialTransformer
from losses import SegRegistrationLoss

CP_SPACING = 8
WEIGHTS = {'dice': 1.0, 'cross_entropy': 0.2, 'bending': 0.01, 'affine_reg': 0.01}


def _rand_onehot(C, N, device):
    lbl = torch.randint(0, C, (1, N, N, N), device=device)
    return F.one_hot(lbl, C).permute(0, 4, 1, 2, 3).float()


def _cube(device):
    """Small closed cube mesh inside the FOV (verts in [-0.4, 0.4])."""
    verts = torch.tensor([
        [-.4, -.4, -.4], [.4, -.4, -.4], [.4, .4, -.4], [-.4, .4, -.4],
        [-.4, -.4, .4], [.4, -.4, .4], [.4, .4, .4], [-.4, .4, .4],
    ], dtype=torch.float32, device=device)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
    ], dtype=np.int64)
    return verts, faces


def _build_model(n_stages, N, device):
    return SegRegistrationNet(
        seg_channels=5, use_affine=False, cp_spacing=CP_SPACING,
        n_stages=n_stages, injectivity_k=0.40, target_size=(N, N, N),
    ).to(device)


def main():
    ap = argparse.ArgumentParser(description="WM-mesh FFD smoke test")
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--size', type=int, default=32, help='cube volume size (mult. of cp_spacing)')
    args = ap.parse_args()
    device = torch.device(args.device)
    N, nl = args.size, args.size // CP_SPACING
    assert args.size % CP_SPACING == 0
    print(f"smoke: device={device} size={N}^3 cp_lattice={nl}^3")

    model = _build_model(n_stages=1, N=N, device=device).eval()

    # (1) Zero control field -> identity (coordinate parity).
    zero = [torch.zeros(1, 3, nl, nl, nl, device=device)]
    pts = torch.rand(256, 3, device=device) * 1.6 - 0.8
    moved = model.deform_points(pts, zero, None)
    max_move = (moved - pts).abs().max().item()
    assert max_move < 1e-6, f"zero control field MOVED vertices by {max_move} (coordinate-parity bug)"
    dense = model.dense_flow_from_cps(zero)
    assert dense.shape == (1, 3, N, N, N), f"dense flow shape {tuple(dense.shape)}"
    assert dense.abs().max().item() < 1e-6, "zero cps → non-zero dense flow"
    print(f"  ok  (1) zero control field → vertices unmoved (max {max_move:.2e}), dense flow ~0")

    # (2) forward shapes + clamp bound.
    tseg, sseg = _rand_onehot(5, N, device), _rand_onehot(5, N, device)
    with torch.no_grad():
        cps_list, affine = model(tseg, sseg)
    assert affine is None
    assert len(cps_list) == 1 and tuple(cps_list[0].shape) == (1, 3, nl, nl, nl), \
        f"cps shape {tuple(cps_list[0].shape)} != (1,3,{nl},{nl},{nl})"
    cp_max, dm = cps_list[0].abs().max().item(), model.ffd.delta_max
    assert cp_max <= dm + 1e-6, f"clamp violated: |cp|={cp_max} > delta_max={dm}"
    print(f"  ok  (2) forward cps shape ok | |cp| {cp_max:.4f} <= delta_max {dm:.4f}")

    # (3) full forward → warp → Dice loss → backward, finite + non-zero grads.
    verts, faces = _cube(device)                   # mesh fixture reused by (4)
    stn = SpatialTransformer((N, N, N), device=device).to(device)
    loss_fn = SegRegistrationLoss(WEIGHTS).to(device)

    model.train()
    cps_list, affine = model(tseg, sseg)
    flow = model.dense_flow_from_cps(cps_list)
    warped = stn(tseg, flow)                        # affine is None here (use_affine=False)
    loss, comps = loss_fn(warped, sseg, cps_list, affine_matrix=affine, return_components=True)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    gnorm = sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None) ** 0.5
    assert np.isfinite(gnorm) and gnorm > 0, f"bad grad norm {gnorm}"
    comp_str = " ".join(f"{k}={v.item():.4f}" for k, v in comps.items())
    print(f"  ok  (3) loss {loss.item():.4f} finite | grad-norm {gnorm:.4f} | {comp_str}")

    # (4) cascade composition path (n_stages=2).
    m2 = _build_model(n_stages=2, N=N, device=device).eval()
    with torch.no_grad():
        cps2, _ = m2(tseg, sseg)
        dense2 = m2.dense_flow_from_cps(cps2)
        dv = m2.deform_points(verts, cps2, None)
    assert len(cps2) == 2 and dense2.shape == (1, 3, N, N, N)
    assert torch.isfinite(dense2).all() and torch.isfinite(dv).all()
    print(f"  ok  (4) n_stages=2 composition runs | dense flow finite, max |u| {dense2.abs().max().item():.4f}")

    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
