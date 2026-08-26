"""Rigid-register each subject's FreeSurfer white surface into the neurite-OASIS
frame the KeyReg models work in.

The .surf files under oasis_mri_outputs/<SUBJ>_clinical come from a SEPARATE
FreeSurfer run on the clinical acquisition (conformed 266^3). neurite-OASIS ships
its own conformed/cropped volumes (aligned_*, 160x192x224). Both are LIA 1mm and
both carry a valid scanner-RAS affine, so surface vertices land inside the neurite
volume -- but the two scans are not registered to each other, so the surface is
offset by an unknown rigid transform. Plotting it without correcting that makes a
good deformation look bad.

This estimates that transform per subject by ICP between two representations of
the SAME anatomical boundary (the WM/GM interface):

    source: lh.white + rh.white vertices           (FS world / scanner RAS)
    target: marching cubes of the neurite WM label (neurite world)

Point-to-point ICP with a Kabsch solve each iteration, initialised on centroids.
Same subject, same boundary, so it converges quickly and tightly. Writes a
4x4 RAS->RAS matrix per subject into one npz.

    python reg_surf_to_neurite.py --val <full_val.txt> --out surf_reg.npz
"""
import argparse, os, re, sys
import numpy as np
import nibabel as nib
from skimage import measure
from scipy.spatial import cKDTree

_D = "/shared/scratch/0/home/v_vijay_bala_mahalingam"
WM_LABEL = 3

ap = argparse.ArgumentParser()
ap.add_argument("--val", default=f"{_D}/neurite_oasis/full_val.txt")
ap.add_argument("--surf_root", default=f"{_D}/oasis_mri_outputs")
ap.add_argument("--surf_suffix", default="_clinical")
ap.add_argument("--out", default="surf_reg.npz")
ap.add_argument("--iters", type=int, default=60)
ap.add_argument("--n_pts", type=int, default=30000)
args = ap.parse_args()


def kabsch(P, Q):
    """Rigid R,t minimising ||R@P.T + t - Q.T||, both (N,3)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def icp(src, dst, iters):
    tree = cKDTree(dst)
    R, t = np.eye(3), np.zeros(3)
    cur = src.copy()
    for _ in range(iters):
        _, idx = tree.query(cur)
        dR, dt = kabsch(cur, dst[idx])
        cur = cur @ dR.T + dt
        R, t = dR @ R, dR @ t + dt
    rms = float(np.sqrt((tree.query(cur)[0] ** 2).mean()))
    return R, t, rms


def main():
    subs = [os.path.basename(os.path.dirname(l.strip()))
            for l in open(args.val) if l.strip()]
    out, skipped = {}, []
    for subj in subs:
        m = re.search(r"(OAS1_\d+)", subj)
        sd = os.path.join(args.surf_root, (m.group(1) if m else "") + args.surf_suffix, "surf")
        lh, rh = os.path.join(sd, "lh.white"), os.path.join(sd, "rh.white")
        ref = os.path.join(f"{_D}/neurite_oasis", subj, "aligned_seg4.nii.gz")
        if not (m and os.path.exists(lh) and os.path.exists(rh) and os.path.exists(ref)):
            skipped.append(subj); continue

        # source: FS white surface in FS world (scanner RAS)
        pts = []
        for pth in (lh, rh):
            c, _, meta = nib.freesurfer.read_geometry(pth, read_metadata=True)
            pts.append(c + meta.get("cras", np.zeros(3)))
        src = np.concatenate(pts)

        # target: neurite WM isosurface in neurite world
        im = nib.load(ref)
        wm = (np.asarray(im.dataobj) == WM_LABEL)
        if wm.sum() == 0:
            skipped.append(subj); continue
        v, _, _, _ = measure.marching_cubes(wm.astype(np.float32), 0.5)
        dst = (np.c_[v, np.ones(len(v))] @ im.affine.T)[:, :3]

        rng = np.random.default_rng(0)
        s = src[rng.choice(len(src), min(args.n_pts, len(src)), replace=False)]
        d = dst[rng.choice(len(dst), min(args.n_pts, len(dst)), replace=False)]

        R, t, rms = icp(s, d, args.iters)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        out[subj] = T
        print(f"{subj}: ICP rms {rms:.3f} mm | translation {np.round(t,2)}", flush=True)

    np.savez(args.out, **out)
    print(f"\nwrote {len(out)} transforms -> {args.out}")
    if skipped:
        print(f"no FreeSurfer surface for {len(skipped)} subjects (of {len(subs)})")


if __name__ == "__main__":
    main()
