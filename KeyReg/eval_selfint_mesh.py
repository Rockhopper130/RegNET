"""Corrected surface self-intersection (triangle-flip %) for the KeyReg SVF models,
plus mesh visualisations.

Why this exists
---------------
`eval_selfint.py` pushes template-mesh vertices into subject space with

    v_new = v - disp(v)

but `disp` is the *pull* field: the model builds `grid = idg + disp` and does
`warped[p] = moving[p + disp(p)]`, so the map it encodes is

    Phi : subject -> template,   Phi(p) = p + disp(p)

Carrying a *template* vertex v into subject space therefore needs Phi^-1(v),
i.e. the o with `o + disp(o) = v`.  `v - disp(v)` is only the first-order
(one-step Picard) approximation of that inverse.  Under the large deformations
SVF-E produces it is not accurate, and the error itself bends triangles and
fabricates orientation flips -- so the reported self-intersection is an upper
bound polluted by inverse error, not a property of the model.

This script pushes the mesh three ways and reports all three so the difference
is visible:

    approx   v - disp(v)                       what eval_selfint.py did
    fixedpt  solve o + disp(o) = v             damped fixed point --
                                               `invert_to_sample` ported from
                                               S-RegNET/model.py on the
                                               invertible-deform-SRegNET branch
    svfexact v + disp_inv(v),
             disp_inv = svf_integrate(-vel)    exact for a stationary velocity
                                               field: Phi = exp(vel) so
                                               Phi^-1 = exp(-vel)

`svfexact` is the one to quote.  All three were checked against a metric none of
them optimises -- mean distance from the pushed mesh to the subject's OWN WM
isosurface -- which puts `approx` at 1.63 mm and `fixedpt`/`svfexact` at 1.266 /
1.253 mm: the two corrected inverses are equally accurate as maps, so the gap to
`approx` is a genuine artifact of the first-order push.

They still differ on flip count (0.34% vs 0.14%) because flipping is a *local*
property.  `fixedpt` drives the residual to ~0 for >95% of vertices but strands
~1% of them by up to 4.6 voxels, and each stray vertex flips every triangle
touching it.  `svfexact` has no strays: exp(-vel) is a diffeomorphism by
construction, so its error is smooth and sub-voxel (p50 0.15 vox) and cannot flip
a triangle.  Warm-starting the fixed point from `svfexact` confirms the
mechanism -- it removes some strays and the count drops 0.34% -> 0.24% while the
map's surface accuracy is unchanged.

Usage
-----
    python eval_selfint_mesh.py \
        --runs svf_E:"SVF-E" --val .../val.txt --figdir figs_mesh --n_fig 3
"""
import sys, os, argparse
import numpy as np, torch, torch.nn.functional as F
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keyreg import SVFReg, load_seg, read_list, dice_per_class, identity_grid_vol, svf_integrate

_D = "/shared/scratch/0/home/v_vijay_bala_mahalingam"

ap = argparse.ArgumentParser()
ap.add_argument("--val", default=f"{_D}/neurite_oasis/val.txt")
ap.add_argument("--template", default=f"{_D}/neurite_oasis/OASIS_OAS1_0001_MR1/seg4_onehot.npy")
ap.add_argument("--runs", nargs="+", default=["svf_E:SVF-E"],
                help="one or more DIRNAME:LABEL under keyreg_runs/")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--n_iter", type=int, default=50, help="fixed-point iterations")
ap.add_argument("--alpha", type=float, default=0.5, help="fixed-point damping")
ap.add_argument("--res_tol", type=float, default=1e-3,
                help="residual above which a vertex counts as un-invertible (normalized units)")
ap.add_argument("--sweep_iters", type=int, nargs="*", default=None,
                help="also report fixed-point flip%% at these iteration counts (convergence check)")
ap.add_argument("--surfcheck", action="store_true",
                help="also measure each pushed mesh against the subject's OWN WM isosurface "
                     "(a discriminator neither inverse method optimises)")
ap.add_argument("--sample_surf", action="store_true",
                help="draw the sample WM from the subject's real FreeSurfer lh/rh.white "
                     "instead of a marching-cubes isosurface (skips subjects without one)")
ap.add_argument("--surf_root", default=f"{_D}/oasis_mri_outputs")
ap.add_argument("--surf_suffix", default="_clinical")
ap.add_argument("--surf_reg", default=None,
                help="npz of per-subject 4x4 RAS->RAS rigid transforms from "
                     "reg_surf_to_neurite.py (the clinical FS scan is not registered "
                     "to the neurite volumes; without this the surface is offset)")
ap.add_argument("--final", action="store_true",
                help="publication figure: correct push only, no old-vs-new comparison")
ap.add_argument("--figdir", default=None, help="if set, write mesh figures here")
ap.add_argument("--n_fig", type=int, default=3, help="how many val subjects to render")
args = ap.parse_args()

dev = args.device
ts = (128, 128, 128)

# --------------------------------------------------------------------------- #
# Template WM surface (channel 3), marching cubes -> normalized (x,y,z)
# --------------------------------------------------------------------------- #
tpl = load_seg(args.template, ts)                       # (5,D,H,W)
wm = tpl[3].numpy()
verts, faces, _, _ = measure.marching_cubes(wm, 0.5)    # verts in (i,j,k) = (D,H,W)
D, H, W = wm.shape
vx = verts[:, 2] / (W - 1) * 2 - 1                      # W -> x
vy = verts[:, 1] / (H - 1) * 2 - 1                      # H -> y
vz = verts[:, 0] / (D - 1) * 2 - 1                      # D -> z
vn = torch.tensor(np.stack([vx, vy, vz], 1), dtype=torch.float32, device=dev)   # (N,3)
faces_t = torch.tensor(faces.astype(np.int64), device=dev)
print(f"template WM surface: {len(vn):,} verts, {len(faces):,} faces", flush=True)


def sample_field(field, pts):
    """Sample (1,3,D,H,W) field, channel order (x,y,z), at pts (N,3) normalized
    (x,y,z) -> (N,3). align_corners=True to match keyreg's grid convention."""
    g = pts.view(1, 1, 1, -1, 3)
    s = F.grid_sample(field, g, mode="bilinear", padding_mode="border",
                      align_corners=True)               # (1,3,1,1,N)
    return s[0, :, 0, 0, :].T                           # (N,3)


def invert_to_sample(flow, points, n_iter=50, alpha=0.5, snapshots=None, init=None):
    """Damped fixed-point inverse of the pull field: find o with o + flow(o) = v.
    Ported from S-RegNET/model.py on invertible-deform-SRegNET; the only change is
    align_corners=True (keyreg builds its grids with linspace(-1,1,n), which is the
    align_corners=True convention, whereas S-RegNET uses pixel-centre coords).

    Where the forward map folds, no o exists and that vertex's residual stays
    high -- gate the mesh on the residual."""
    v = points
    o = v.clone() if init is None else init.clone()
    snaps, want = {}, set(snapshots or ())
    for it in range(1, n_iter + 1):
        o = o + alpha * (v - sample_field(flow, o) - o)
        if it in want:
            snaps[it] = o.clone()
    res = (o + sample_field(flow, o) - v).norm(dim=1)
    return o, res, snaps


def flip_mask(v0, v1):
    """Per-face orientation flip between the template mesh v0 and pushed mesh v1."""
    a0, b0, c0 = v0[faces_t[:, 0]], v0[faces_t[:, 1]], v0[faces_t[:, 2]]
    a1, b1, c1 = v1[faces_t[:, 0]], v1[faces_t[:, 1]], v1[faces_t[:, 2]]
    n0 = torch.cross(b0 - a0, c0 - a0, dim=1)
    n1 = torch.cross(b1 - a1, c1 - a1, dim=1)
    return (n0 * n1).sum(1) < 0


def build(ck):
    a = ck["args"]
    m = SVFReg(target=a["target"], delta=a["delta"], int_steps=a["int_steps"], K=a["K"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval(); return m


def velocity(m, moving, fixed):
    """Replay SVFReg.forward up to the velocity field (forward() does not return it)."""
    x = torch.cat([moving, fixed], 1)
    if m.kp_guided:
        x = torch.cat([x, m._sal(moving), m._sal(fixed)], 1)
    return m.flow(x)                                    # (B,3,T,T,T)


if args.surfcheck:
    from scipy.spatial import cKDTree

MM_PER_NORM = 256.0 / 2.0     # normalized -> mm (256 mm FOV across [-1,1])


def subject_wm_surface_norm(fix_wm, want_faces=False):
    """Marching-cubes the subject's WM channel and return its vertices in the same
    normalized (x,y,z) frame the pushed mesh lives in."""
    sv, sf, _, _ = measure.marching_cubes(fix_wm, 0.5)
    v = np.stack([sv[:, 2] / (W - 1) * 2 - 1,
                  sv[:, 1] / (H - 1) * 2 - 1,
                  sv[:, 0] / (D - 1) * 2 - 1], 1)
    return (v, sf.astype(np.int64)) if want_faces else v


def load_sample_white(subj):
    """Subject's real FreeSurfer white surface in the same normalized (x,y,z) frame
    as the pushed mesh. subj is e.g. OASIS_OAS1_0415_MR1 -> OAS1_0415<suffix>.
    surface-RAS + cras -> world -> aligned_seg4 voxel -> normalized. Returns
    (verts (N,3), faces (M,3)) or (None, None) if the subject has no surface."""
    import re
    m = re.search(r"(OAS1_\d+)", subj)
    if not m:
        return None, None
    sd = os.path.join(args.surf_root, m.group(1) + args.surf_suffix, "surf")
    lh, rh = os.path.join(sd, "lh.white"), os.path.join(sd, "rh.white")
    ref = os.path.join(f"{_D}/neurite_oasis", subj, "aligned_seg4.nii.gz")
    if not (os.path.exists(lh) and os.path.exists(rh) and os.path.exists(ref)):
        return None, None
    import nibabel as nib
    im = nib.load(ref)
    T = None
    if args.surf_reg:
        z = np.load(args.surf_reg)
        if subj not in z.files:
            return None, None
        T = z[subj]
    inv = np.linalg.inv(im.affine)
    nd = np.array(im.shape, dtype=np.float64)          # (160,192,224) = (L,I,A)

    def hemi(pth):
        c, f, meta = nib.freesurfer.read_geometry(pth, read_metadata=True)
        w = c + meta.get("cras", np.zeros(3))          # surface-RAS -> scanner RAS
        if T is not None:                              # clinical scan -> neurite scan
            w = (np.c_[w, np.ones(len(w))] @ T.T)[:, :3]
        vox = (np.c_[w, np.ones(len(w))] @ inv.T)[:, :3]
        v = np.stack([vox[:, 2] / (nd[2] - 1) * 2 - 1,     # axis2 = A -> x
                      vox[:, 1] / (nd[1] - 1) * 2 - 1,     # axis1 = I -> y
                      vox[:, 0] / (nd[0] - 1) * 2 - 1], 1) # axis0 = L -> z
        return v.astype(np.float32), np.asarray(f, dtype=np.int64)

    lv, lf = hemi(lh)
    rv, rf = hemi(rh)
    return np.concatenate([lv, rv]), np.concatenate([lf, rf + len(lv)])


idg = identity_grid_vol(128, dev)
tpl_b = tpl.unsqueeze(0).to(dev)
va = read_list(args.val)

# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
if args.figdir:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    os.makedirs(args.figdir, exist_ok=True)


def norm_to_vox(v):
    """normalized (x,y,z) in [-1,1] -> voxel (i,j,k), align_corners=True."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack([(z + 1) / 2 * (D - 1), (y + 1) / 2 * (H - 1), (x + 1) / 2 * (W - 1)], 1)


# The neurite-OASIS volumes are LIA: array axis 0 runs L, axis 1 runs I, axis 2
# runs A.  So slicing axis 0 gives a SAGITTAL plane, axis 1 an AXIAL plane and
# axis 2 a CORONAL plane -- not the axial/coronal/sagittal order the naive
# ordering assumes.  For each view, pick which source axis is screen-x and which
# is screen-y, and which needs flipping so the result is anatomically upright
# (superior up, anterior right on sagittal; anterior up on axial).
VIEWS = {
    0: dict(name="sagittal", x=2, y=1, xflip=False, yflip=True),   # x=A, y=S
    1: dict(name="axial",    x=0, y=2, xflip=False, yflip=False),  # x=L, y=A
    2: dict(name="coronal",  x=0, y=1, xflip=False, yflip=True),   # x=L, y=S
}
DIMS = (D, H, W)


def orient_image(sl, axis):
    """sl = np.take(vol, s, axis=axis), indexed [p, q] with p < q the in-plane
    source axes.  Return it as img[row=screen_y, col=screen_x], upright."""
    v = VIEWS[axis]
    p, q = [a for a in range(3) if a != axis]
    img = sl.T if (v["x"], v["y"]) == (p, q) else sl      # row must be the y source axis
    if v["yflip"]:
        img = img[::-1, :]
    if v["xflip"]:
        img = img[:, ::-1]
    return img


def orient_seg(seg, axis):
    """Map contour segments from in-plane (p, q) source coords to screen (x, y),
    applying the same swap and flips as orient_image."""
    if not len(seg):
        return seg
    v = VIEWS[axis]
    p, q = [a for a in range(3) if a != axis]
    out = seg.copy().astype(np.float64)
    if (v["x"], v["y"]) == (q, p):                        # swap the two columns
        out = out[:, :, ::-1]
    if v["xflip"]:
        out[:, :, 0] = (DIMS[v["x"]] - 1) - out[:, :, 0]
    if v["yflip"]:
        out[:, :, 1] = (DIMS[v["y"]] - 1) - out[:, :, 1]
    return out


def slice_segments(vox, fcs, axis, s):
    """Contour of the mesh at {axis == s}. Returns (segments (K,2,2) in-plane voxel
    coords, face index (K,)) so segments can be coloured by per-face flags."""
    p, q = [ax for ax in range(3) if ax != axis]
    d = vox[:, axis] - s
    xy = vox[:, [p, q]]
    df, pf = d[fcs], xy[fcs]                            # (M,3) (M,3,2)
    pts, masks = [], []
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        di, dj = df[:, i], df[:, j]
        cross = (di > 0) != (dj > 0)
        denom = di - dj
        safe = np.abs(denom) > 1e-12
        t = np.where(safe, di / np.where(safe, denom, 1.0), 0.5)
        pts.append(pf[:, i, :] + t[:, None] * (pf[:, j, :] - pf[:, i, :]))
        masks.append(cross)
    P = np.stack(pts, axis=1)
    Mk = np.stack(masks, axis=1)
    sel = Mk.sum(1) == 2
    seg = P[sel][Mk[sel]].reshape(-1, 2, 2)
    return seg, np.flatnonzero(sel)


def contour(vox, axis, s, fcs=None):
    """Screen-space contour segments + face indices."""
    seg, fidx = slice_segments(vox, faces if fcs is None else fcs, axis, s)
    return orient_seg(seg, axis), fidx


def render_three(name, run_label, v_tpl_vox, v_smp, v_ok_vox, fl_ok, out, smp_src="marching cubes"):
    """Template WM, sample WM and deformed WM surfaces, as contours on three
    anatomically-oriented ortho planes. v_smp is (verts_vox, faces) for the
    subject's own WM isosurface."""
    smp_vox, smp_faces = v_smp
    center = np.clip(np.round(v_ok_vox.mean(0)).astype(int), 0, np.array([D, H, W]) - 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    for a in range(3):
        ax = axes[a]
        sl = int(center[a])
        ax.set_facecolor("black")
        seg_t, _ = contour(v_tpl_vox, a, sl)
        if len(seg_t):
            ax.add_collection(LineCollection(seg_t, colors="orange", linewidths=1.1, alpha=0.9))
        seg_s, _ = contour(smp_vox, a, sl, fcs=smp_faces)
        if len(seg_s):
            ax.add_collection(LineCollection(seg_s, colors="limegreen", linewidths=1.1, alpha=0.9))
        seg, fidx = contour(v_ok_vox, a, sl)
        if len(seg):
            bad = fl_ok[fidx]
            if (~bad).any():
                ax.add_collection(LineCollection(seg[~bad], colors="deepskyblue", linewidths=1.5))
            if bad.any():
                ax.add_collection(LineCollection(seg[bad], colors="red", linewidths=2.4))
        ax.set_xlim(0, DIMS[VIEWS[a]["x"]] - 1)
        ax.set_ylim(0, DIMS[VIEWS[a]["y"]] - 1)
        ax.set_aspect("equal")
        ax.set_title(f"{VIEWS[a]['name']} @ {sl}")
        ax.set_xticks([]); ax.set_yticks([])
        if a == 0:
            ax.legend(handles=[
                Line2D([0], [0], color="orange", lw=2, label="template WM"),
                Line2D([0], [0], color="limegreen", lw=2, label=f"sample WM ({smp_src})"),
                Line2D([0], [0], color="deepskyblue", lw=2, label="deformed WM"),
                Line2D([0], [0], color="red", lw=2, label="self-intersecting"),
            ], loc="lower right", fontsize=8, framealpha=0.75)
    fig.suptitle(f"{run_label} — {name}   template / sample / deformed WM surface   "
                 f"[self-intersection {100*fl_ok.mean():.4f}%]",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    figure -> {out}", flush=True)


def render(name, run_label, fix_wm, v_tpl_vox, v_ok_vox, v_old_vox, fl_ok, fl_old, res, out):
    """3 anatomically-oriented ortho slices + residual histogram.  Grey = the
    subject's own WM mask (where the pushed mesh should land).  Orange = template
    surface before pushing.  Cyan = correctly pushed mesh (red where a triangle's
    orientation flipped).  Dashed magenta = the old v - disp(v) push."""
    center = np.clip(np.round(v_ok_vox.mean(0)).astype(int), 0, np.array(fix_wm.shape) - 1)
    ncol = 3 if args.final else 4
    fig, axes = plt.subplots(1, ncol, figsize=(17, 6.4) if args.final else (23, 6.4))
    for a in range(3):
        ax = axes[a]
        s = int(center[a])
        ax.imshow(orient_image(np.take(fix_wm, s, axis=a), a),
                  cmap="gray", origin="lower", aspect="equal")
        seg_t, _ = contour(v_tpl_vox, a, s)
        if len(seg_t):
            ax.add_collection(LineCollection(seg_t, colors="orange", linewidths=0.9, alpha=0.85))
        seg_o, _ = contour(v_old_vox, a, s)
        if len(seg_o) and not args.final:
            ax.add_collection(LineCollection(seg_o, colors="magenta", linewidths=0.9,
                                             alpha=0.75, linestyles="dashed"))
        seg, fidx = contour(v_ok_vox, a, s)
        if len(seg):
            bad = fl_ok[fidx]
            if (~bad).any():
                ax.add_collection(LineCollection(seg[~bad], colors="deepskyblue", linewidths=1.5))
            if bad.any():
                ax.add_collection(LineCollection(seg[bad], colors="red", linewidths=2.4))
        ax.set_title(f"{VIEWS[a]['name']} @ {s}")
        ax.axis("off")
        if a == 0:
            handles = [Line2D([0], [0], color="orange", lw=2, label="template WM surface (unpushed)")]
            if not args.final:
                handles.append(Line2D([0], [0], color="magenta", lw=2, ls="--",
                                      label="old push  v - disp(v)"))
            handles += [Line2D([0], [0], color="deepskyblue", lw=2,
                               label="deformed mesh" if args.final else "correct push  (true inverse)"),
                        Line2D([0], [0], color="red", lw=2, label="flipped triangle")]
            ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.65)

    if args.final:
        fig.suptitle(f"{run_label} — {name}   template WM mesh deformed into subject space   "
                     f"[self-intersection {100*fl_ok.mean():.4f}%]",
                     fontweight="bold", fontsize=13)
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    figure -> {out}", flush=True)
        return

    axh = axes[3]
    axh.hist(res, bins=80, color="steelblue", log=True)
    axh.axvline(args.res_tol, color="red", ls="--", lw=1.2,
                label=f"tol {args.res_tol:g}  ({100*(res>args.res_tol).mean():.3f}% over)")
    axh.set_xlabel("per-vertex inverse residual  ||o + disp(o) - v||  (normalized)")
    axh.set_ylabel("vertices (log)")
    axh.set_title("fixed-point inverse-solve residual")
    axh.legend(fontsize=8)

    fig.suptitle(f"{run_label} — {name}   template WM mesh pushed into subject space   "
                 f"[triangle flips: correct {100*fl_ok.mean():.4f}%   vs   old approx {100*fl_old.mean():.4f}%]",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> {out}", flush=True)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
v_tpl_vox = norm_to_vox(vn.cpu().numpy())

for spec in args.runs:
    d, _, name = spec.partition(":")
    name = name or d
    p = f"{_D}/keyreg_runs/{d}/best.pth"
    ck = torch.load(p, map_location=dev, weights_only=False)
    m = build(ck)
    fl_approx, fl_fixed, fl_exact = [], [], []
    res_max, res_frac, inv_err = [], [], []
    res_exact_max, res_exact_frac = [], []
    rq = {k: [] for k in ("fixedpt", "svfexact")}
    fl_warm, res_warm_q = [], []
    n_done = 0
    surf = {k: [] for k in ("approx", "fixedpt", "svfexact", "warm")}
    sweep = {k: [] for k in (args.sweep_iters or [])}
    pcs = np.zeros(5)
    with torch.no_grad():
        for si, fp in enumerate(va):
            fix = load_seg(fp, ts).unsqueeze(0).to(dev)
            vel = velocity(m, tpl_b, fix)
            disp = svf_integrate(vel, m.int_steps, idg)              # pull disp (B,3,D,H,W)
            grid = idg + disp.permute(0, 2, 3, 4, 1)
            warped = F.grid_sample(tpl_b, grid, mode="bilinear",
                                   padding_mode="border", align_corners=True)
            pcs += np.array(dice_per_class(warped, fix))

            # --- three pushes -------------------------------------------------
            v_approx = vn - sample_field(disp, vn)                    # old, first order
            v_fixed, res, snaps = invert_to_sample(disp, vn, args.n_iter, args.alpha,
                                                   snapshots=args.sweep_iters)
            disp_inv = svf_integrate(-vel, m.int_steps, idg)          # exact SVF inverse
            v_exact = vn + sample_field(disp_inv, vn)

            f_a, f_f, f_e = (flip_mask(vn, v_approx), flip_mask(vn, v_fixed),
                             flip_mask(vn, v_exact))
            fl_approx.append(f_a.float().mean().item() * 100)
            fl_fixed.append(f_f.float().mean().item() * 100)
            fl_exact.append(f_e.float().mean().item() * 100)
            # Same solver, warm-started from the SVF inverse: does it stay there
            # (svfexact already near-optimal) or walk to the cold-start answer?
            v_warm, res_w, _ = invert_to_sample(disp, vn, args.n_iter, args.alpha,
                                                init=v_exact)
            f_w = flip_mask(vn, v_warm)
            fl_warm.append(f_w.float().mean().item() * 100)

            res_e = (v_exact + sample_field(disp, v_exact) - vn).norm(dim=1)
            VOX = (128 - 1) / 2.0                       # normalized -> voxels
            for k, rr in (("fixedpt", res), ("svfexact", res_e)):
                rq[k].append(np.percentile(rr.cpu().numpy() * VOX, [50, 95, 99, 100]))
            res_warm_q.append(np.percentile(res_w.cpu().numpy() * VOX, [50, 95, 99, 100]))

            if args.surfcheck:
                tree = cKDTree(subject_wm_surface_norm(fix[0, 3].cpu().numpy()))
                for k, vv in (("approx", v_approx), ("fixedpt", v_fixed),
                              ("svfexact", v_exact), ("warm", v_warm)):
                    surf[k].append(tree.query(vv.cpu().numpy())[0].mean() * MM_PER_NORM)
            res_exact_max.append(res_e.max().item())
            res_exact_frac.append((res_e > args.res_tol).float().mean().item() * 100)
            res_max.append(res.max().item())
            res_frac.append((res > args.res_tol).float().mean().item() * 100)
            inv_err.append((v_fixed - v_exact).norm(dim=1).mean().item())
            for k, ov in snaps.items():
                sweep[k].append(flip_mask(vn, ov).float().mean().item() * 100)

            if args.figdir and n_done < args.n_fig and args.final:
                subj = os.path.basename(os.path.dirname(fp))
                if args.sample_surf:
                    sv, sfc = load_sample_white(subj)
                    if sv is None:
                        print(f"    [skip fig] {subj}: no lh/rh.white", flush=True)
                        continue
                    src = "FreeSurfer lh/rh.white"
                else:
                    sv, sfc = subject_wm_surface_norm(fix[0, 3].cpu().numpy(), want_faces=True)
                    src = "marching cubes"
                n_done += 1
                render_three(subj, name, v_tpl_vox, (norm_to_vox(sv), sfc),
                             norm_to_vox(v_exact.cpu().numpy()), f_e.cpu().numpy(),
                             os.path.join(args.figdir, f"mesh_{d}_{subj}.png"), src)
            elif args.figdir and si < args.n_fig:
                subj = os.path.basename(os.path.dirname(fp))
                fix_wm = fix[0, 3].cpu().numpy()
                render(subj, name, fix_wm, v_tpl_vox,
                       norm_to_vox(v_exact.cpu().numpy()),
                       norm_to_vox(v_approx.cpu().numpy()),
                       f_e.cpu().numpy(), f_a.cpu().numpy(), res.cpu().numpy(),
                       os.path.join(args.figdir, f"mesh_{d}_{subj}.png"))

    n = len(va)
    print(f"\n=== {name}  (dir {d}, epoch {ck.get('epoch')}, {n} val subjects) ===")
    print(f"  WM Dice: {pcs[3]/n:.4f} | logged folding: {ck.get('fold'):.4f}%")
    print(f"  triangle-flip %   approx  (v - disp(v), old eval_selfint.py) : {np.mean(fl_approx):.4f}%")
    print(f"  triangle-flip %   fixedpt (invert_to_sample, invertible-deform): {np.mean(fl_fixed):.4f}%")
    print(f"  triangle-flip %   svfexact(integrate(-vel), ground truth)     : {np.mean(fl_exact):.4f}%")
    print(f"  residual of fixedpt   max {np.mean(res_max):.2e} | "
          f"{np.mean(res_frac):.4f}% of verts over tol {args.res_tol:g}")
    print(f"  residual of svfexact  max {np.mean(res_exact_max):.2e} | "
          f"{np.mean(res_exact_frac):.4f}% of verts over tol {args.res_tol:g}")
    print("     NOTE: the residual is what the fixed point explicitly minimises, so it cannot")
    print("     adjudicate between the two -- read the percentile table and the surface")
    print("     distance below instead. svfexact carries a smooth sub-voxel bias everywhere;")
    print("     fixedpt is ~exact for >95% of vertices but strands ~1% by several voxels, and")
    print("     those strays are what flip triangles.")
    print(f"  triangle-flip %   fixedpt WARM-STARTED from svfexact           : {np.mean(fl_warm):.4f}%")
    print(f"  |fixedpt - svfexact| mean vertex gap: {np.mean(inv_err):.2e} normalized "
          f"({np.mean(inv_err)*(128-1)/2:.3f} vox)")
    print("  inverse residual, voxels        p50      p95      p99      max")
    for k in ("fixedpt", "svfexact"):
        a = np.mean(np.stack(rq[k]), 0)
        print(f"      {k:<24s} {a[0]:8.4f} {a[1]:8.4f} {a[2]:8.4f} {a[3]:8.4f}")
    a = np.mean(np.stack(res_warm_q), 0)
    print(f"      {'warm':<24s} {a[0]:8.4f} {a[1]:8.4f} {a[2]:8.4f} {a[3]:8.4f}")
    if args.surfcheck:
        print("  mean distance from pushed mesh to the SUBJECT's own WM isosurface (mm)")
        print("      -- independent of the residual metric; lower = the map is genuinely better")
        for k in ("approx", "fixedpt", "svfexact", "warm"):
            print(f"      {k:<24s} {np.mean(surf[k]):.4f} mm")
    if sweep:
        print("  fixed-point convergence (flip %% vs iterations, target = svfexact "
              f"{np.mean(fl_exact):.4f}%%):")
        for k in sorted(sweep):
            print(f"      n_iter={k:<5d} {np.mean(sweep[k]):.4f}%")
