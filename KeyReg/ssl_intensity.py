"""Phase 2: intensity-driven self-supervised registration baseline.

Architecture-matched twin of SVFReg: identical FlowUNet body, identical
scaling-and-squaring, identical grid convention. The ONLY difference is the input --
2 channels of intensity instead of 10-12 channels of one-hot segmentation plus
keypoint saliency. That is what makes the Phase 3 comparison a test of supervision
rather than a test of architecture.

Trains on aligned_norm.nii.gz with NO labels anywhere in the loss OR in checkpoint
selection. Selection is on masked validation NCC; if it were on Dice the model would
be secretly label-supervised and every Dice number in Phase 3 would be void. Dice is
logged as MONITOR ONLY.

Capacity note: dropping the KeypointNet branch (it consumes labels) removes ~1.1M
parameters. Both counts are logged so the comparison can be read honestly.

    python ssl_intensity.py --smooth_w 3000 --fold_w 40 --epochs 250 \
        --out <...>/keyreg_runs/ssl_ncc
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "archive", "M-RegNET"))

import svfcommon as S
from keyreg import (FlowUNet, svf_integrate, identity_grid_vol, grad_smooth,
                    jac_penalty, folding_pct, dice_per_class, read_list, load_seg)
from losses_mri import ncc_loss          # the file the team already reviewed


class IntensitySVF(nn.Module):
    """Stationary velocity field driven by image intensity.

    kp_guided = False and .flow / .int_steps / .target are exposed deliberately, so
    svfcommon.velocity() and every Phase-1 metric run on this model with zero
    branching -- the Phase-3 scorer must not need to know which model it holds."""

    def __init__(self, target=128, delta=0.20, int_steps=12, b=20):
        super().__init__()
        self.kp_guided = False
        self.flow = FlowUNet(in_ch=2, b=b, delta=delta)
        self.target, self.int_steps = target, int_steps

    def forward(self, moving, fixed):
        vel = self.flow(torch.cat([moving, fixed], 1))
        idg = identity_grid_vol(self.target, moving.device).expand(
            moving.shape[0], -1, -1, -1, -1)
        disp = svf_integrate(vel, self.int_steps, idg)
        grid = idg + disp.permute(0, 2, 3, 4, 1)
        warped = S.warp(moving, grid)
        return warped, grid, None, None


def preload(subjects, name="aligned_norm.nii.gz", workers=16):
    """Intensities at 128^3, trilinear. 330 x 128^3 x 4B = 2.8 GB, fits in RAM."""
    def one(s):
        v = S.load_intensity(s, name=name)
        p999 = torch.quantile(v.flatten().float(), 0.999).item()
        assert 0.3 <= p999 <= 1.5, f"{s}: intensity p99.9 = {p999:.3f}, unexpected scale"
        return v
    with ThreadPoolExecutor(workers) as ex:
        return torch.cat(list(ex.map(one, subjects)), 0)     # (N,1,D,H,W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", default=f"{S.NEURITE}/full_train.txt")
    ap.add_argument("--val_txt", default=f"{S.NEURITE}/full_val.txt")
    ap.add_argument("--template", default=S.TEMPLATE_SUBJ)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--smooth_w", type=float, default=3000.0)
    ap.add_argument("--fold_w", type=float, default=40.0)
    ap.add_argument("--delta", type=float, default=0.20)
    ap.add_argument("--int_steps", type=int, default=12)
    ap.add_argument("--target", type=int, default=128)
    ap.add_argument("--ncc_win", type=int, default=9)
    ap.add_argument("--erode", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    S.require_shared_tmpdir()
    S.assert_scale_mm()
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "train.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device

    tr_subs = S.check_lists(read_list(args.train_txt), must_exclude_template=True,
                            is_train=True)
    va_subs = S.check_lists(read_list(args.val_txt), must_exclude_template=True)
    assert not (set(tr_subs) & set(va_subs)), "train/val overlap"
    log(f"SELECT_ON=val_ncc   (Dice is MONITOR ONLY and must never drive selection)")
    log(f"train {len(tr_subs)}  val {len(va_subs)}  template {args.template}")

    t0 = time.time()
    tr = preload(tr_subs).to("cpu")
    va = preload(va_subs).to("cpu")
    tpl = S.load_intensity(args.template, device=dev)
    log(f"preloaded intensities in {time.time()-t0:.0f}s  "
        f"train {tuple(tr.shape)}  val {tuple(va.shape)}")

    # Segmentations for the MONITOR-ONLY Dice column.
    va_seg = [load_seg(os.path.join(S.NEURITE, s, "seg4_onehot.npy"), S.TS)
              for s in va_subs]
    tpl_seg = load_seg(os.path.join(S.NEURITE, args.template, "seg4_onehot.npy"),
                       S.TS).unsqueeze(0).to(dev)

    model = IntensitySVF(target=args.target, delta=args.delta,
                         int_steps=args.int_steps).to(dev)
    log(f"IntensitySVF params: {sum(p.numel() for p in model.parameters()):,} "
        "(SVF-E carries an extra ~1.1M in the label-driven KeypointNet branch)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    idg = identity_grid_vol(args.target, dev)
    best = -1e9

    for ep in range(1, args.epochs + 1):
        model.train()
        te = time.time()
        tot = 0.0
        for _ in range(args.steps):
            i, j = random.randrange(len(tr)), random.randrange(len(tr))
            moving = tr[i:i + 1].to(dev)
            fixed = tr[j:j + 1].to(dev)
            warped, grid, _, _ = model(moving, fixed)
            loss = (ncc_loss(warped, fixed, win=args.ncc_win)
                    + args.smooth_w * grad_smooth(grid, idg)
                    + args.fold_w * jac_penalty(grid))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()

        model.eval()
        with torch.no_grad():
            nccs, folds, disps, dices = [], [], [], []
            for k, s in enumerate(va_subs):
                fixed = va[k:k + 1].to(dev)
                warped, grid, _, _ = model(tpl, fixed)
                mask = S.brain_mask(fixed, erode=args.erode)
                v, _ = S.masked_local_ncc(warped, fixed, mask, args.ncc_win)
                nccs.append(v)
                folds.append(folding_pct(grid))
                disps.append(S.mean_disp_mm(grid, idg, mask))
                # MONITOR ONLY -- never used for selection
                dices.append(dice_per_class(S.warp(tpl_seg, grid),
                                            va_seg[k].unsqueeze(0).to(dev))[3])
            vn, vf = float(np.mean(nccs)), float(np.mean(folds))
            vd, vdice = float(np.mean(disps)), float(np.mean(dices))
            lj = S.logjac_stats(grid)

        log(f"Ep {ep}/{args.epochs} | {time.time()-te:.0f}s | train {tot/args.steps:.4f} "
            f"| VAL ncc {vn:.4f} | fold {vf:.4f}% | disp {vd:.3f}mm "
            f"| stdlogdet {lj['std_log_det']:.4f} | WM_Dice {vdice:.4f} MONITOR")

        state = {"model": model.state_dict(), "epoch": ep, "val_ncc": vn,
                 "fold": vf, "disp_mm": vd, "std_log_det": lj["std_log_det"],
                 "wm_dice_monitor": vdice, "select_on": "val_ncc",
                 "args": {**vars(args), "K": 0}}
        if vn > best:
            best = vn
            torch.save(state, os.path.join(args.out, "best.pth"))
            log(f"   * new best val_ncc {vn:.4f}")
        torch.save(state, os.path.join(args.out, "last.pth"))
        if ep % 25 == 0:
            torch.save(state, os.path.join(args.out, f"snap_{ep}.pth"))

    log(f"done. best val_ncc {best:.4f}")


if __name__ == "__main__":
    main()
