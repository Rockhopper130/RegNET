"""
KeyReg — correspondence-based diffeomorphic registration of brain segmentations.

A different paradigm from the team's dense-field + regularizer models: a 3D CNN
detects K anatomical keypoints on any segmentation (learned end-to-end, ordered
so channel i = the same anatomical point across subjects). Registration solves a
closed-form Thin-Plate-Spline (TPS) from the K correspondences — smoothness /
invertibility come analytically from the TPS, not from a folding penalty.
Test-time = two forward passes + one linear solve (real-time).

Trains on random subject pairs; evaluates fixed-template -> val (comparable to the
team's WM-Dice table). fspython only (torch + numpy).
"""
import os, sys, time, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Data: preload seg4_onehot volumes, resized to target_size (nearest).
# --------------------------------------------------------------------------- #
def load_seg(path, ts):
    a = torch.tensor(np.load(path), dtype=torch.float32)          # (5,*)
    a = F.interpolate(a.unsqueeze(0), size=ts, mode="nearest")[0]  # (5,D,H,W)
    return a

def read_list(p):
    return [l.strip() for l in open(p) if l.strip()]


# --------------------------------------------------------------------------- #
# Keypoint detector: strided 3D CNN -> K heatmaps -> soft-argmax coords in [-1,1]
# --------------------------------------------------------------------------- #
def cbr(ci, co, s=1):
    return nn.Sequential(nn.Conv3d(ci, co, 3, stride=s, padding=1),
                         nn.InstanceNorm3d(co), nn.LeakyReLU(0.2, inplace=True))

class KeypointNet(nn.Module):
    def __init__(self, in_ch=5, K=128, width=32):
        super().__init__()
        w = width
        self.enc = nn.Sequential(
            cbr(in_ch, w), cbr(w, w, 2),          # 128 -> 64
            cbr(w, 2*w), cbr(2*w, 2*w),           # 64  (keep resolution)
            cbr(2*w, 2*w), cbr(2*w, 2*w),         # 64
            cbr(2*w, 2*w),
        )
        self.head = nn.Conv3d(2*w, K, 1)          # K heatmaps at 64^3 (finer keypoints)
        self.K = K

    def forward(self, x):
        h = self.head(self.enc(x))                # (B,K,d,h,w)
        B, K, D, H, W = h.shape
        p = h.flatten(2).softmax(-1).view(B, K, D, H, W)   # spatial softmax
        dev = x.device
        lz = torch.linspace(-1, 1, D, device=dev)
        ly = torch.linspace(-1, 1, H, device=dev)
        lx = torch.linspace(-1, 1, W, device=dev)
        z = (p.sum((3, 4)) * lz).sum(-1)          # (B,K)  along D
        y = (p.sum((2, 4)) * ly).sum(-1)          # along H
        x_ = (p.sum((2, 3)) * lx).sum(-1)         # along W
        return torch.stack([x_, y, z], -1)        # (B,K,3) order (x,y,z)


# --------------------------------------------------------------------------- #
# Thin-Plate-Spline (3D), batched & differentiable. Kernel U(r)=r (biharmonic).
# --------------------------------------------------------------------------- #
def tps_solve(src, dst, lam):
    """src,dst: (B,K,3). Returns params (B,K+4,3) for f with f(src)=dst."""
    B, K, _ = src.shape
    dev = src.device
    Kmat = torch.cdist(src, src) + lam * torch.eye(K, device=dev).unsqueeze(0)  # (B,K,K)
    P = torch.cat([torch.ones(B, K, 1, device=dev), src], -1)                   # (B,K,4)
    top = torch.cat([Kmat, P], -1)                                              # (B,K,K+4)
    bot = torch.cat([P.transpose(1, 2), torch.zeros(B, 4, 4, device=dev)], -1)  # (B,4,K+4)
    A = torch.cat([top, bot], 1)                                                # (B,K+4,K+4)
    Y = torch.cat([dst, torch.zeros(B, 4, 3, device=dev)], 1)                   # (B,K+4,3)
    return torch.linalg.solve(A, Y)

def tps_apply(params, src, G):
    """params (B,K+4,3), src (B,K,3), G (B,M,3) -> f(G) (B,M,3)."""
    K = src.shape[1]
    w, a = params[:, :K], params[:, K:]
    U = torch.cdist(G, src)                                   # (B,M,K)
    Pg = torch.cat([torch.ones(G.shape[0], G.shape[1], 1, device=G.device), G], -1)
    return U @ w + Pg @ a


def identity_grid(n, dev):
    """(1, n^3, 3) identity grid in [-1,1], order (x,y,z)."""
    l = torch.linspace(-1, 1, n, device=dev)
    z, y, x = torch.meshgrid(l, l, l, indexing="ij")
    return torch.stack([x, y, z], -1).reshape(1, -1, 3)


def identity_grid_vol(n, dev):
    """(1, n, n, n, 3) identity sampling grid, order (x,y,z)."""
    return identity_grid(n, dev).view(1, n, n, n, 3)


def compose(gridA, gridB):
    """Grid for A∘B: x -> A(B(x)). Both (B,D,H,W,3)."""
    a = gridA.permute(0, 4, 1, 2, 3)
    s = F.grid_sample(a, gridB, mode="bilinear", padding_mode="border", align_corners=True)
    return s.permute(0, 2, 3, 4, 1)


def svf_integrate(v, nsteps, id_grid):
    """Scaling-and-squaring: turn a stationary velocity field v (B,3,D,H,W) into
    the displacement of a diffeomorphism phi=exp(v). Returns (B,3,D,H,W)."""
    disp = v / (2 ** nsteps)
    for _ in range(nsteps):
        grid = id_grid + disp.permute(0, 2, 3, 4, 1)                 # (B,D,H,W,3)
        sampled = F.grid_sample(disp, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        disp = disp + sampled
    return disp


class KeyReg(nn.Module):
    def __init__(self, K=128, lam=0.1, field_n=64, target=128, width=32, affine_only=False):
        super().__init__()
        self.net = KeypointNet(K=K, width=width)
        self.K, self.lam, self.field_n, self.target = K, lam, field_n, target
        self.affine_only = affine_only

    def warp_field(self, moving_kp, fixed_kp):
        """Return sampling grid (B, T,T,T, 3) mapping fixed-space -> moving coords."""
        B = fixed_kp.shape[0]; dev = fixed_kp.device
        params = tps_solve(fixed_kp, moving_kp, self.lam)          # phi(fixed_kp)=moving_kp
        if self.affine_only:
            params = params.clone()
            params[:, :self.K] = 0                                 # drop TPS nonlinearity -> diffeo affine
        G = identity_grid(self.field_n, dev).expand(B, -1, -1)     # coarse field
        phi = tps_apply(params, fixed_kp, G)                       # (B, n^3, 3)
        n = self.field_n
        phi = phi.view(B, n, n, n, 3).permute(0, 4, 1, 2, 3)      # (B,3,n,n,n)
        phi = F.interpolate(phi, size=(self.target,) * 3, mode="trilinear", align_corners=True)
        return phi.permute(0, 2, 3, 4, 1)                          # (B,T,T,T,3)

    def forward(self, moving, fixed):
        mkp = self.net(moving)
        fkp = self.net(fixed)
        grid = self.warp_field(mkp, fkp)
        warped = F.grid_sample(moving, grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return warped, grid, mkp, fkp


# --------------------------------------------------------------------------- #
# Hybrid: keypoint-TPS (global) + bounded residual flow U-Net (cortical detail)
# --------------------------------------------------------------------------- #
class FlowUNet(nn.Module):
    """Small 3D U-Net: (moving_coarse, fixed) -> bounded residual displacement."""
    def __init__(self, in_ch=10, b=16, delta=0.4):
        super().__init__()
        self.delta = delta
        self.e1 = cbr(in_ch, b)
        self.d1 = cbr(b, 2*b, 2)          # 128->64
        self.e2 = cbr(2*b, 2*b)
        self.d2 = cbr(2*b, 4*b, 2)        # 64->32
        self.e3 = cbr(4*b, 4*b)
        self.d3 = cbr(4*b, 6*b, 2)        # 32->16
        self.bott = cbr(6*b, 6*b)
        self.u3 = cbr(6*b + 4*b, 4*b)
        self.u2 = cbr(4*b + 2*b, 2*b)
        self.u1 = cbr(2*b + b, b)
        self.out = nn.Conv3d(b, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # start at identity

    def up(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
        return torch.cat([x, skip], 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        b = self.bott(self.d3(e3))
        u = self.u3(self.up(b, e3))
        u = self.u2(self.up(u, e2))
        u = self.u1(self.up(u, e1))
        return torch.tanh(self.out(u)) * self.delta        # (B,3,D,H,W)


class HybridReg(nn.Module):
    def __init__(self, K=256, lam=0.3, field_n=64, target=128, width=32, delta=0.4,
                 diffeo=False, int_steps=6, tps_affine=False):
        super().__init__()
        self.kr = KeyReg(K=K, lam=lam, field_n=field_n, target=target, width=width,
                         affine_only=tps_affine)
        self.flow = FlowUNet(in_ch=10, b=16, delta=delta)
        self.target = target
        self.diffeo = diffeo
        self.int_steps = int_steps

    def forward(self, moving, fixed):
        mkp = self.kr.net(moving); fkp = self.kr.net(fixed)
        tps_grid = self.kr.warp_field(mkp, fkp)                       # (B,T,T,T,3)
        moving_c = F.grid_sample(moving, tps_grid, mode="bilinear",
                                 padding_mode="border", align_corners=True)
        vel = self.flow(torch.cat([moving_c, fixed], 1))            # (B,3,T,T,T) velocity/disp
        idg = identity_grid_vol(self.target, moving.device).expand(moving.shape[0], -1, -1, -1, -1)
        # diffeomorphic: integrate velocity (scaling-squaring) -> invertible by construction
        disp = svf_integrate(vel, self.int_steps, idg) if self.diffeo else vel
        res_grid = idg + disp.permute(0, 2, 3, 4, 1)
        total = compose(tps_grid, res_grid)                          # true composed warp
        warped = F.grid_sample(moving, total, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return warped, total, mkp, fkp


class SVFReg(nn.Module):
    """Keypoint-guided DIFFEOMORPHIC registration. A single full-resolution
    stationary velocity field (from a U-Net conditioned on moving/fixed + their
    keypoint saliency) is integrated by scaling-and-squaring -> the warp is a
    diffeomorphism by construction, so folding ~ 0. No TPS, no grid composition
    (the two folding sources in the hybrid)."""
    def __init__(self, target=128, delta=0.5, int_steps=7, K=256, width=32, kp_guided=True):
        super().__init__()
        self.kp_guided = kp_guided
        self.kpnet = KeypointNet(K=K, width=width) if kp_guided else None
        self.flow = FlowUNet(in_ch=10 + (2 if kp_guided else 0), b=20, delta=delta)
        self.target, self.int_steps = target, int_steps

    def _sal(self, x):
        h = self.kpnet.head(self.kpnet.enc(x))                 # (B,K,d,h,w)
        s = h.flatten(2).softmax(-1).view_as(h).max(1, keepdim=True)[0]
        return F.interpolate(s, size=(self.target,) * 3, mode="trilinear", align_corners=True)

    def forward(self, moving, fixed):
        x = torch.cat([moving, fixed], 1)
        if self.kp_guided:
            x = torch.cat([x, self._sal(moving), self._sal(fixed)], 1)
        vel = self.flow(x)                                     # (B,3,T,T,T) velocity
        idg = identity_grid_vol(self.target, moving.device).expand(moving.shape[0], -1, -1, -1, -1)
        disp = svf_integrate(vel, self.int_steps, idg)         # diffeomorphic
        grid = idg + disp.permute(0, 2, 3, 4, 1)
        warped = F.grid_sample(moving, grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return warped, grid, None, None


# --------------------------------------------------------------------------- #
# Losses & metrics
# --------------------------------------------------------------------------- #
def dice_loss(pred, tgt, cw):
    # pred,tgt: (B,5,*). foreground classes 1..4, weighted.
    p, t = pred[:, 1:], tgt[:, 1:]
    ax = (2, 3, 4)
    inter = (p * t).sum(ax); union = p.sum(ax) + t.sum(ax)
    d = (2 * inter + 1e-5) / (union + 1e-5)          # (B,4)
    w = cw.to(d.dtype)
    return 1 - (d * w).sum(1).mean() / w.sum()

def ce_loss(pred, tgt):
    return F.cross_entropy(pred.clamp_min(1e-7).log(), tgt.argmax(1))

@torch.no_grad()
def dice_per_class(pred, tgt):
    ph = F.one_hot(pred.argmax(1), 5).permute(0, 4, 1, 2, 3).float()
    out = []
    for c in range(5):
        p, t = ph[:, c], tgt[:, c]
        out.append(((2 * (p * t).sum() + 1e-5) / (p.sum() + t.sum() + 1e-5)).item())
    return out

def jac_penalty(grid):
    """Differentiable anti-folding penalty: ReLU(-det) mean + squared, on the
    sampling-grid Jacobian. Drives folding toward 0 without a hard SVF."""
    g = grid.permute(0, 4, 1, 2, 3)
    dz = F.pad(g[:, :, 1:] - g[:, :, :-1], (0, 0, 0, 0, 0, 1))
    dy = F.pad(g[:, :, :, 1:] - g[:, :, :, :-1], (0, 0, 0, 1))
    dx = F.pad(g[:, :, :, :, 1:] - g[:, :, :, :, :-1], (0, 1))
    det = (dx[:, 0] * (dy[:, 1] * dz[:, 2] - dy[:, 2] * dz[:, 1])
           - dx[:, 1] * (dy[:, 0] * dz[:, 2] - dy[:, 2] * dz[:, 0])
           + dx[:, 2] * (dy[:, 0] * dz[:, 1] - dy[:, 1] * dz[:, 0]))
    ref = (2.0 / grid.shape[1]) ** 3
    dn = det / ref
    neg = F.relu(-dn)
    return neg.mean() + (neg ** 2).mean()


def grad_smooth(grid, idg):
    """Diffusion regularizer: mean squared gradient of the displacement field.
    Keeps the flow smooth -> the integrated warp stays diffeomorphic (folding~0)
    even at large deformations. This is the term the raw SVF was missing."""
    d = (grid - idg).permute(0, 4, 1, 2, 3)           # (B,3,D,H,W) displacement
    dz = d[:, :, 1:] - d[:, :, :-1]
    dy = d[:, :, :, 1:] - d[:, :, :, :-1]
    dx = d[:, :, :, :, 1:] - d[:, :, :, :, :-1]
    return dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()


@torch.no_grad()
def folding_pct(grid):
    # grid (B,D,H,W,3); TRUE folding on INTERIOR voxels only, strict det<0
    # (no boundary padding, which previously counted the image shell as folding).
    g = grid.permute(0, 4, 1, 2, 3)                   # (B,3,D,H,W)
    dz = (g[:, :, 1:, :, :] - g[:, :, :-1, :, :])[:, :, :, :-1, :-1]
    dy = (g[:, :, :, 1:, :] - g[:, :, :, :-1, :])[:, :, :-1, :, :-1]
    dx = (g[:, :, :, :, 1:] - g[:, :, :, :, :-1])[:, :, :-1, :-1, :]
    det = (dx[:, 0] * (dy[:, 1] * dz[:, 2] - dy[:, 2] * dz[:, 1])
           - dx[:, 1] * (dy[:, 0] * dz[:, 2] - dy[:, 2] * dz[:, 0])
           + dx[:, 2] * (dy[:, 0] * dz[:, 1] - dy[:, 1] * dz[:, 0]))
    return (det < 0).float().mean().item() * 100.0


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", default="/shared/scratch/0/home/v_vijay_bala_mahalingam/neurite_oasis/train.txt")
    ap.add_argument("--val_txt",   default="/shared/scratch/0/home/v_vijay_bala_mahalingam/neurite_oasis/val.txt")
    ap.add_argument("--template",  default="/shared/scratch/0/home/v_vijay_bala_mahalingam/neurite_oasis/OASIS_OAS1_0001_MR1/seg4_onehot.npy")
    ap.add_argument("--out", default="/shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/run")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--steps", type=int, default=80)      # random pairs per epoch
    ap.add_argument("--K", type=int, default=192)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wm_w", type=float, default=3.0)
    ap.add_argument("--fold_w", type=float, default=1.0)
    ap.add_argument("--hybrid", action="store_true", help="keypoint-TPS + bounded residual flow")
    ap.add_argument("--delta", type=float, default=0.4, help="residual flow bound (normalized)")
    ap.add_argument("--diffeo", action="store_true", help="integrate residual velocity (scaling-squaring)")
    ap.add_argument("--int_steps", type=int, default=6)
    ap.add_argument("--tps_affine", action="store_true", help="use only affine part of keypoint transform (diffeo)")
    ap.add_argument("--svf", action="store_true", help="clean keypoint-guided diffeomorphic SVF (folding~0)")
    ap.add_argument("--smooth_w", type=float, default=0.0, help="velocity/flow diffusion smoothness weight")
    ap.add_argument("--target", type=int, default=128)
    ap.add_argument("--field_n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = "cuda:0"
    ts = (args.target,) * 3
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "train.log"), "a")
    def log(m):
        print(m); logf.write(m + "\n"); logf.flush()

    log(f"[{time.strftime('%H:%M:%S')}] KeyReg start | K={args.K} lam={args.lam} "
        f"lr={args.lr} epochs={args.epochs} field_n={args.field_n}")

    tr_paths = read_list(args.train_txt); va_paths = read_list(args.val_txt)
    log(f"Preloading {len(tr_paths)} train + {len(va_paths)} val ...")
    tr = torch.stack([load_seg(p, ts) for p in tr_paths]).to(dev)   # (Ntr,5,*)
    va = torch.stack([load_seg(p, ts) for p in va_paths]).to(dev)
    template = load_seg(args.template, ts).unsqueeze(0).to(dev)
    cw = torch.tensor([1., 1., 1., args.wm_w], device=dev)          # fg class weights c1..c4 (WM=idx2)

    if args.svf:
        model = SVFReg(target=args.target, delta=args.delta, int_steps=args.int_steps,
                       K=args.K).to(dev)
        log(f"MODEL: keypoint-guided diffeomorphic SVF (int_steps={args.int_steps}, delta={args.delta})")
    elif args.hybrid:
        model = HybridReg(K=args.K, lam=args.lam, field_n=args.field_n,
                          target=args.target, delta=args.delta,
                          diffeo=args.diffeo, int_steps=args.int_steps,
                          tps_affine=args.tps_affine).to(dev)
        log(f"MODEL: Hybrid (keypoint-TPS + residual flow{' + diffeo integration' if args.diffeo else ''})")
    else:
        model = KeyReg(K=args.K, lam=args.lam, field_n=args.field_n, target=args.target).to(dev)
        log("MODEL: pure keypoint-TPS")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    log(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    best = 0.0
    idg_full = identity_grid_vol(args.target, dev)
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; td = 0.0
        for _ in range(args.steps):
            i, j = random.randrange(len(tr)), random.randrange(len(tr))
            moving, fixed = tr[i:i+1], tr[j:j+1]
            warped, grid, mkp, fkp = model(moving, fixed)
            dl = dice_loss(warped, fixed, cw)
            loss = dl + 0.2 * ce_loss(warped, fixed) + args.fold_w * jac_penalty(grid) \
                   + args.smooth_w * grad_smooth(grid, idg_full)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); td += 1 - dl.item()
        sched.step()

        # ---- validation: fixed template -> each val subject ----
        model.eval()
        with torch.no_grad():
            pcs = np.zeros(5); fold = 0.0
            for k in range(len(va)):
                warped, grid, _, _ = model(template, va[k:k+1])
                pcs += np.array(dice_per_class(warped, va[k:k+1]))
                fold += folding_pct(grid)
            pcs /= len(va); fold /= len(va)
        wm = pcs[3]; fg = pcs[1:].mean()
        log(f"[{time.strftime('%H:%M:%S')}] Ep {ep}/{args.epochs} | {time.time()-t0:.0f}s "
            f"| train_dice {td/args.steps:.4f} | VAL fg {fg:.4f} WM {wm:.4f} "
            f"| fold {fold:.4f}% | perclass "
            f"C0 {pcs[0]:.3f} C1 {pcs[1]:.3f} C2 {pcs[2]:.3f} C3 {pcs[3]:.3f} C4 {pcs[4]:.3f}")

        if wm > best:
            best = wm
            torch.save({"epoch": ep, "model": model.state_dict(), "wm": wm,
                        "fg": fg, "fold": fold, "per_class": pcs.tolist(), "args": vars(args)},
                       os.path.join(args.out, "best.pth"))
            log(f"   * new best WM {best:.4f} (fold {fold:.4f}%)")

    log(f"[{time.strftime('%H:%M:%S')}] DONE. best WM {best:.4f}")
    logf.close()


if __name__ == "__main__":
    main()
