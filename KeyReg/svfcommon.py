"""Shared helpers for SVF evaluation.

No module-level side effects, no argparse -- unlike eval_selfint_mesh.py, this is
safe to import. The geometry helpers are copied verbatim from eval_selfint_mesh.py /
check_push_control.py / export_deformed_surfs.py (which currently carry three
drifting copies each); the guards below are new.

Grid convention, used everywhere:

    vel  = velocity(m, M, F)                    # (B,3,T,T,T)
    grid = idg + svf_integrate(vel, n, idg)     # (B,T,T,T,3), order (x,y,z)
    warped = grid_sample(M, grid)               # lives in F-space

so the grid encodes the coordinate map T(M,F): F-space -> M-space, and because the
warp is a stationary velocity field, T(M,F)^-1 = exp(-v) = svf_integrate(-vel).
Template evaluation is M=template, F=subject, giving subject -> template.
"""
import os
import re
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

_D = "/shared/scratch/0/home/v_vijay_bala_mahalingam"
NEURITE = f"{_D}/neurite_oasis"
RUNS = f"{_D}/keyreg_runs"
SURF_ROOT = f"{_D}/oasis_mri_outputs"
SURF_SUFFIX = "_clinical"
TEMPLATE_SUBJ = "OASIS_OAS1_0001_MR1"
TS = (128, 128, 128)

# Normalized [-1,1] -> millimetres, per axis. The pre-resize grid is 160x192x224
# (L,I,A) at 1 mm, and normalized x/y/z map to volume axes 2/1/0, so the scale is
# anisotropic: a single scalar factor is WRONG. Verified by assert_scale_mm().
SCALE_MM = (111.5, 95.5, 79.5)          # (223/2, 191/2, 159/2)


# --------------------------------------------------------------------------- #
# Guards. Every one of these is a hard assert -- this project has twice shipped a
# wrong-but-plausible number, and a printed warning would not have stopped either.
# --------------------------------------------------------------------------- #
def require_shared_tmpdir():
    """/tmp on this host is 100% full (another user's checkpoints). Anything that
    spills there dies mid-run, hours in."""
    t = tempfile.gettempdir()
    assert t.startswith("/shared"), (
        f"TMPDIR is {t!r}; /tmp is full on this host. "
        "export TMPDIR=/shared/home/v_vijay_bala_mahalingam/tmp")


def assert_scale_mm(subj=TEMPLATE_SUBJ, n=2000, tol=1e-4, seed=0):
    """Confirm SCALE_MM reproduces the true affine-based distance. Guards the
    anisotropic-scaling bug: the 128^3 model grid is a resample of a 160x192x224
    volume, so normalized->mm is per-axis."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (n, 3))
    b = rng.uniform(-1, 1, (n, 3))
    d_true = np.linalg.norm(norm_to_world(a, subj) - norm_to_world(b, subj), axis=1)
    d_scale = np.linalg.norm((a - b) * np.asarray(SCALE_MM), axis=1)
    err = np.abs(d_true - d_scale).max()
    assert err < tol, f"SCALE_MM disagrees with the affine by {err:.3e} mm (tol {tol})"
    return err


def check_lists(paths, must_exclude_template=True, train_txt=f"{NEURITE}/full_train.txt",
                is_train=False):
    """Assert an evaluation list is actually held out. The old val.txt has 6 of its
    10 subjects inside full_train.txt -- scoring svf_E_full on it is meaningless.

    Pass is_train=True when validating the TRAIN list itself, which is of course
    allowed to overlap with train."""
    subs = [os.path.basename(os.path.dirname(p.strip())) for p in paths if p.strip()]
    assert len(subs) == len(set(subs)), "duplicate subjects in evaluation list"
    if os.path.exists(train_txt) and not is_train:
        tr = {os.path.basename(os.path.dirname(l.strip()))
              for l in open(train_txt) if l.strip()}
        leak = sorted(set(subs) & tr)
        assert not leak, f"{len(leak)} eval subjects are in the TRAIN split: {leak[:5]}"
    if must_exclude_template:
        assert TEMPLATE_SUBJ not in subs, f"{TEMPLATE_SUBJ} is the template; drop it"
    assert len(subs) in (82, 83, 330, 412, 413), \
        f"unexpected split size {len(subs)} -- expected one of 82/83/330/412/413"
    return subs


def paired_subjects(gt_txt=f"{NEURITE}/full_val.txt",
                    clinical_name="seg4_onehot_clinical.npy", verbose=True):
    """Subjects present in BOTH conditions, matched by subject ID and never by line
    position. selfint_clinical_unpaired_legacy.log documents what positional pairing
    cost last time."""
    gt = [l.strip() for l in open(gt_txt) if l.strip()]
    keep, dropped = [], []
    for p in gt:
        subj = os.path.basename(os.path.dirname(p))
        if os.path.exists(os.path.join(NEURITE, subj, clinical_name)):
            keep.append(subj)
        else:
            dropped.append(subj)
    if verbose and dropped:
        print(f"[paired] excluded {len(dropped)} without {clinical_name}: {dropped}",
              flush=True)
    return keep, dropped


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_intensity(subj, name="aligned_norm.nii.gz", ts=TS, device="cpu"):
    """Intensity volume at model resolution, TRILINEAR.

    Deliberately NOT load_seg: that uses nearest, which is right for one-hot labels
    and quantizes an intensity image into something that looks like a plausible
    model deficiency. The dtype assert makes the mix-up impossible to do silently."""
    import nibabel as nib
    p = os.path.join(NEURITE, subj, name)
    a = np.asarray(nib.load(p).get_fdata(), dtype=np.float32)
    assert a.ndim == 3, f"{p}: expected a 3-D intensity volume, got {a.shape}"
    t = torch.from_numpy(a)[None, None]
    t = F.interpolate(t, size=ts, mode="trilinear", align_corners=True)
    return t.to(device)                                  # (1,1,D,H,W)


def brain_mask(intensity, erode=6):
    """Boolean brain mask from a skull-stripped intensity volume, eroded so no NCC
    window and no composed-grid sample straddles the FOV boundary."""
    m = (intensity > 0).float()
    if erode > 0:
        k = 2 * erode + 1
        m = -F.max_pool3d(-m, kernel_size=k, stride=1, padding=erode)
    return m > 0.5


def build_seg_model(ck, device):
    """Rebuild SVFReg from a checkpoint's stored args (never hardcode int_steps)."""
    from keyreg import SVFReg
    a = ck["args"]
    m = SVFReg(target=a["target"], delta=a["delta"],
               int_steps=a["int_steps"], K=a["K"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def velocity(m, moving, fixed):
    """Replay forward up to the velocity field; forward() does not return it.

    Works unchanged for IntensitySVF because that class sets kp_guided = False and
    exposes .flow -- which is what lets Phase 3 run one metric path for both models."""
    x = torch.cat([moving, fixed], 1)
    if m.kp_guided:
        x = torch.cat([x, m._sal(moving), m._sal(fixed)], 1)
    return m.flow(x)                                     # (B,3,T,T,T)


def fwd_inv_grids(m, moving, fixed, idg, int_steps=None):
    """(vel, grid_fwd, grid_inv). exp(-v) is the exact inverse of exp(v)."""
    from keyreg import svf_integrate
    n = m.int_steps if int_steps is None else int_steps
    vel = velocity(m, moving, fixed)
    g_f = idg + svf_integrate(vel, n, idg).permute(0, 2, 3, 4, 1)
    g_i = idg + svf_integrate(-vel, n, idg).permute(0, 2, 3, 4, 1)
    return vel, g_f, g_i


def warp(vol, grid, mode="bilinear"):
    """The only grid_sample in new code. keyreg builds every grid with
    linspace(-1,1,n), i.e. align_corners=True; a single False anywhere shifts by half
    a voxel and looks like a small real error."""
    return F.grid_sample(vol, grid, mode=mode, padding_mode="border",
                         align_corners=True)


# --------------------------------------------------------------------------- #
# Geometry (copied verbatim from eval_selfint_mesh.py)
# --------------------------------------------------------------------------- #
def sample_field(field, pts):
    """Sample (1,3,D,H,W) field, channel order (x,y,z), at pts (N,3) normalized."""
    g = pts.view(1, 1, 1, -1, 3)
    s = F.grid_sample(field, g, mode="bilinear", padding_mode="border",
                      align_corners=True)
    return s[0, :, 0, 0, :].T


def norm_to_world(v, subj):
    """Model-normalized (x,y,z) -> scanner RAS millimetres via the subject's aligned
    volume affine. The pre-resize grid is 160x192x224, so a single isotropic
    normalized-to-mm scale factor is not valid."""
    import nibabel as nib
    im = nib.load(os.path.join(NEURITE, subj, "aligned_seg4.nii.gz"))
    nd = np.asarray(im.shape, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    vox = np.stack([(z + 1) / 2 * (nd[0] - 1),
                    (y + 1) / 2 * (nd[1] - 1),
                    (x + 1) / 2 * (nd[2] - 1)], 1)
    return (np.c_[vox, np.ones(len(vox))] @ im.affine.T)[:, :3]


def subject_wm_surface_norm(fix_wm, want_faces=False):
    """Marching-cubes a WM channel -> normalized (x,y,z)."""
    from skimage import measure
    sv, sf, _, _ = measure.marching_cubes(fix_wm, 0.5)
    D, H, W = fix_wm.shape
    v = np.stack([sv[:, 2] / (W - 1) * 2 - 1,
                  sv[:, 1] / (H - 1) * 2 - 1,
                  sv[:, 0] / (D - 1) * 2 - 1], 1)
    return (v, sf.astype(np.int64)) if want_faces else v


def load_sample_white(subj, surf_root=SURF_ROOT, surf_suffix=SURF_SUFFIX):
    """FreeSurfer lh+rh.white in the model-normalized frame, or (None, None)."""
    import nibabel as nib
    m = re.search(r"(OAS1_\d+)", subj)
    if not m:
        return None, None
    sd = os.path.join(surf_root, m.group(1) + surf_suffix, "surf")
    lh, rh = os.path.join(sd, "lh.white"), os.path.join(sd, "rh.white")
    ref = os.path.join(NEURITE, subj, "aligned_seg4.nii.gz")
    if not (os.path.exists(lh) and os.path.exists(rh) and os.path.exists(ref)):
        return None, None
    im = nib.load(ref)
    inv = np.linalg.inv(im.affine)
    nd = np.array(im.shape, dtype=np.float64)

    def hemi(pth):
        c, f, meta = nib.freesurfer.read_geometry(pth, read_metadata=True)
        w = c + meta.get("cras", np.zeros(3))
        vox = (np.c_[w, np.ones(len(w))] @ inv.T)[:, :3]
        v = np.stack([vox[:, 2] / (nd[2] - 1) * 2 - 1,
                      vox[:, 1] / (nd[1] - 1) * 2 - 1,
                      vox[:, 0] / (nd[0] - 1) * 2 - 1], 1)
        return v.astype(np.float32), np.asarray(f, dtype=np.int64)

    lv, lf = hemi(lh)
    rv, rf = hemi(rh)
    return np.concatenate([lv, rv]), np.concatenate([lf, rf + len(lv)])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mm_err(gridA, gridB, mask=None):
    """Per-voxel ||gridA - gridB|| in millimetres, reduced over mask."""
    s = torch.tensor(SCALE_MM, device=gridA.device, dtype=gridA.dtype)
    d = ((gridA - gridB) * s).norm(dim=-1)               # (B,D,H,W)
    if mask is not None:
        d = d[mask[:, 0]] if mask.dim() == 5 else d[mask]
    d = d.flatten().float()
    q = torch.quantile(d, torch.tensor([0.5, 0.95, 0.99], device=d.device))
    return {"mean": d.mean().item(), "p50": q[0].item(), "p95": q[1].item(),
            "p99": q[2].item(), "max": d.max().item()}


def _local_sums(x, win):
    k = torch.ones(1, 1, win, win, win, device=x.device, dtype=x.dtype)
    return F.conv3d(x, k, padding=win // 2)


def masked_local_ncc(a, b, mask, win=9, var_floor=1e-4):
    """Local windowed NCC, averaged over mask only.

    ncc_loss averages over the whole volume, where constant background gives
    cc = 0/(0+eps) = 0 and drags the mean down by an amount that depends on how much
    air the warp moved -- i.e. the metric becomes a function of the deformation.
    Masking removes that. Windows with near-zero variance are excluded and the
    excluded fraction is returned."""
    n = win ** 3
    Ia, Ib = a * a, b * b
    Iab = a * b
    sa, sb = _local_sums(a, win), _local_sums(b, win)
    saa, sbb, sab = _local_sums(Ia, win), _local_sums(Ib, win), _local_sums(Iab, win)
    mu_a, mu_b = sa / n, sb / n
    cross = sab - mu_b * sa - mu_a * sb + mu_a * mu_b * n
    va = saa - 2 * mu_a * sa + mu_a * mu_a * n
    vb = sbb - 2 * mu_b * sb + mu_b * mu_b * n
    good = (va > var_floor * n) & (vb > var_floor * n) & mask
    cc = cross * cross / (va * vb + 1e-5)
    frac_excluded = 1.0 - (good.sum().float() / mask.sum().float()).item()
    return cc[good].mean().item(), frac_excluded


def masked_pearson(a, b, mask):
    x, y = a[mask].float(), b[mask].float()
    x = x - x.mean()
    y = y - y.mean()
    return (x @ y / (x.norm() * y.norm() + 1e-12)).item()


def masked_mse(a, b, mask):
    return ((a[mask] - b[mask]) ** 2).mean().item()


def masked_nmi(a, b, mask, bins=64):
    """Normalized mutual information (H(A)+H(B))/H(A,B) over the mask.

    The one intensity metric that is nobody's training loss, so it carries more
    weight than NCC when comparing a seg-supervised model against an NCC-trained one."""
    x, y = a[mask].float(), b[mask].float()
    def q(v):
        lo, hi = v.min(), v.max()
        return ((v - lo) / (hi - lo + 1e-12) * (bins - 1)).round().long().clamp(0, bins - 1)
    xi, yi = q(x), q(y)
    joint = torch.bincount(xi * bins + yi, minlength=bins * bins).float().view(bins, bins)
    joint /= joint.sum()
    px, py = joint.sum(1), joint.sum(0)
    def H(p):
        p = p[p > 0]
        return -(p * p.log()).sum()
    hj = H(joint.flatten())
    return ((H(px) + H(py)) / (hj + 1e-12)).item()


def logjac_stats(grid):
    """det(J) statistics on the same interior finite differences as folding_pct.
    std_log_det is the standard deformation-regularity number and belongs next to
    the folding percentage."""
    g = grid.permute(0, 4, 1, 2, 3)
    dz = (g[:, :, 1:, :, :] - g[:, :, :-1, :, :])[:, :, :, :-1, :-1]
    dy = (g[:, :, :, 1:, :] - g[:, :, :, :-1, :])[:, :, :-1, :, :-1]
    dx = (g[:, :, :, :, 1:] - g[:, :, :, :, :-1])[:, :, :-1, :-1, :]
    det = (dx[:, 0] * (dy[:, 1] * dz[:, 2] - dy[:, 2] * dz[:, 1])
           - dx[:, 1] * (dy[:, 0] * dz[:, 2] - dy[:, 2] * dz[:, 0])
           + dx[:, 2] * (dy[:, 0] * dz[:, 1] - dy[:, 1] * dz[:, 0]))
    dn = det / ((2.0 / grid.shape[1]) ** 3)
    pos = dn[dn > 0]
    return {"mean_det": dn.mean().item(),
            "std_log_det": pos.log().std().item(),
            "frac_neg": (dn < 0).float().mean().item() * 100.0}


def mean_disp_mm(grid, idg, mask=None):
    """Mean displacement magnitude in mm. Mandatory alongside every regularity
    metric: folding, flips, inverse consistency and atlas variance are ALL optimised
    by the identity warp, so a regularity number without a magnitude number next to
    it cannot distinguish a good model from a lazy one."""
    return mm_err(grid, idg, mask)["mean"]
