"""Recompute TRUE folding (interior voxels, strict det<0, + brain-masked) and Dice
for saved KeyReg checkpoints. No retraining."""
import sys, glob, os
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg")
import time
from keyreg import HybridReg, KeyReg, SVFReg, load_seg, read_list, dice_per_class

dev = "cuda:0"
VAL = "/shared/scratch/0/home/v_vijay_bala_mahalingam/neurite_oasis/val.txt"
TPL = "/shared/scratch/0/home/v_vijay_bala_mahalingam/neurite_oasis/OASIS_OAS1_0001_MR1/seg4_onehot.npy"
ts = (128, 128, 128)


def jdet_interior(grid):
    """Jacobian det of sampling grid on INTERIOR voxels (no boundary padding).
    grid (1,D,H,W,3) -> det (1, D-1,H-1,W-1)."""
    g = grid.permute(0, 4, 1, 2, 3)                       # (1,3,D,H,W)
    dz = (g[:, :, 1:, :, :] - g[:, :, :-1, :, :])[:, :, :, :-1, :-1]
    dy = (g[:, :, :, 1:, :] - g[:, :, :, :-1, :])[:, :, :-1, :, :-1]
    dx = (g[:, :, :, :, 1:] - g[:, :, :, :, :-1])[:, :, :-1, :-1, :]
    det = (dx[:, 0] * (dy[:, 1] * dz[:, 2] - dy[:, 2] * dz[:, 1])
           - dx[:, 1] * (dy[:, 0] * dz[:, 2] - dy[:, 2] * dz[:, 0])
           + dx[:, 2] * (dy[:, 0] * dz[:, 1] - dy[:, 1] * dz[:, 0]))
    return det                                            # (1,D-1,H-1,W-1)


def build_model(ck):
    a = ck["args"]
    if a.get("svf"):
        m = SVFReg(target=a["target"], delta=a["delta"], int_steps=a["int_steps"], K=a["K"]).to(dev)
    else:
        m = HybridReg(K=a["K"], lam=a["lam"], field_n=a["field_n"], target=a["target"],
                      delta=a["delta"], diffeo=a.get("diffeo", False),
                      int_steps=a.get("int_steps", 6), tps_affine=a.get("tps_affine", False)).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m


def evaluate(path, name):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = build_model(ck)
    va = read_list(VAL)
    tpl = load_seg(TPL, ts).unsqueeze(0).to(dev)
    old_fold = []      # old metric (<=0, padded boundary) for reference
    new_fold = []      # interior, strict <0
    brain_fold = []    # interior AND inside brain (template or warped foreground)
    pcs = np.zeros(5)
    with torch.no_grad():
        for p in va:
            fix = load_seg(p, ts).unsqueeze(0).to(dev)
            warped, grid, _, _ = m(tpl, fix)
            pcs += np.array(dice_per_class(warped, fix))
            det = jdet_interior(grid)                     # (1,d,h,w)
            neg = (det < 0)
            new_fold.append(neg.float().mean().item() * 100)
            # brain mask: foreground of warped (any class 1..4), cropped to interior
            fg = (warped[:, 1:].sum(1) > 0.5)[:, :-1, :-1, :-1]
            denom = fg.sum().item()
            brain_fold.append((neg & fg).sum().item() / denom * 100 if denom else 0.0)
    # inference speed (real-time claim): time one registration forward pass
    fix = load_seg(va[0], ts).unsqueeze(0).to(dev)
    with torch.no_grad():
        for _ in range(3):
            _ = m(tpl, fix)                       # warmup
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            _ = m(tpl, fix)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.time() - t0) / 20 * 1000
    n = len(va)
    print(f"=== {name} (epoch {ck.get('epoch')}) ===")
    print(f"  WM Dice (C3): {pcs[3]/n:.4f} | fg-mean: {pcs[1:].mean()/n:.4f}")
    print(f"  TRUE folding (interior, det<0):        {np.mean(new_fold):.4f}%")
    print(f"  brain-region folding:                  {np.mean(brain_fold):.4f}%")
    print(f"  inference speed:                       {ms:.1f} ms/registration")


if __name__ == "__main__":
    runs = [
        ("svf_A", "SVF-A balanced (smooth=150)"),
        ("svf_B", "SVF-B min-fold (smooth=400, int=10)"),
    ]
    base = "/shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs"
    for d, name in runs:
        p = os.path.join(base, d, "best.pth")
        if os.path.exists(p):
            try:
                evaluate(p, name)
            except Exception as e:
                print(f"=== {name}: ERROR {e}")
