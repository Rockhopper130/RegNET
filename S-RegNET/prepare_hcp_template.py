"""
Prepare HCP 100206 as the Canonical Dual Template (WM + Pial / GM) for S-RegNET.

Performs:
1. Loads 100206.L.pial.native.surf.gii and 100206.R.pial.native.surf.gii -> combines & repairs -> template_pial_mesh_repaired.npz
2. Loads 100206.L.white.native.surf.gii and 100206.R.white.native.surf.gii -> combines & repairs -> template_wm_mesh_repaired.npz
3. Converts 100206/T1w/aparc+aseg.nii.gz into the 5-class one-hot template segmentation:
   template_pial_seg_onehot.npy and template_pial_seg.nii.gz

Usage:
    python prepare_hcp_template.py \
        --hcp_dir /shared/scratch/0/home/v_mahith_reddy/hcp_data/100206 \
        --out_dir /shared/scratch/0/home/v_mahith_reddy/hcp_data/template_100206
"""

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# Add parent directory for utils
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

from label_mapping import remap_to_5class
from repair_template_mesh import detect, repair
from visualize_mesh import world_to_norm


def to_one_hot(seg, num_classes=5):
    """(D, H, W) int labels -> (5, D, H, W) uint8 one-hot."""
    one_hot = np.zeros((num_classes, *seg.shape), dtype=np.uint8)
    for c in range(num_classes):
        one_hot[c] = (seg == c).astype(np.uint8)
    return one_hot


def load_gii_mesh(gii_path):
    """Load vertices and triangular faces from a GIFTI surface file."""
    gii = nib.load(gii_path)
    verts = np.asarray(gii.darrays[0].data, dtype=np.float64)
    faces = np.asarray(gii.darrays[1].data, dtype=np.int64)
    return verts, faces


def process_surface_pair(lh_path, rh_path, ref_img, out_npz_path, surf_name="surface"):
    print(f"\nProcessing {surf_name} mesh pair:\n  LH: {lh_path}\n  RH: {rh_path}")
    v_l, f_l = load_gii_mesh(lh_path)
    v_r, f_r = load_gii_mesh(rh_path)

    n_lh = len(v_l)
    print(f"  LH: {n_lh:,} vertices, {len(f_l):,} faces")
    print(f"  RH: {len(v_r):,} vertices, {len(f_r):,} faces")

    all_verts = np.concatenate([v_l, v_r], axis=0)
    all_faces = np.concatenate([f_l, f_r + n_lh], axis=0)

    print(f"  Testing self-intersections on raw {surf_name} mesh...")
    st0, _ = detect(all_verts, all_faces)
    print(f"  Initial crossing faces: {st0['si_faces']:,} ({st0['si_faces_pct']:.4f}%)")

    if st0['si_faces'] > 0:
        print(f"  Repairing {surf_name} crossings to guarantee 0 self-intersections...")
        repaired_verts, hist = repair(
            all_verts, all_faces,
            k=2, lam=0.3, inner=5, max_iter=60
        )
        st1, _ = detect(repaired_verts, all_faces)
        d = np.linalg.norm(repaired_verts - all_verts, axis=1)
        print(f"  Repaired crossing faces: {st1['si_faces']:,} ({st1['si_faces_pct']:.4f}%)")
        print(f"  Vertices moved: {int((d > 1e-6).sum()):,} ({(d > 1e-6).mean() * 100:.3f}%), "
              f"max move: {float(d.max()):.4f} mm")
    else:
        print(f"  Mesh is already 100% clean (0 crossings)!")
        repaired_verts = all_verts
        st1 = st0

    verts_norm = world_to_norm(repaired_verts, ref_img).astype(np.float32)

    np.savez(
        out_npz_path,
        verts_norm=verts_norm,
        faces=all_faces.astype(np.int64),
        n_lh=np.int64(n_lh),
        verts_world=repaired_verts.astype(np.float32)
    )
    print(f"  -> Saved {surf_name} mesh: {out_npz_path}")
    return st0, st1


def build_template_segmentation(hcp_subj_dir, out_dir):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ref_seg_path = Path(hcp_subj_dir) / "T1w" / "aparc+aseg.nii.gz"
    if not ref_seg_path.exists():
        raise FileNotFoundError(f"Reference segmentation {ref_seg_path} does not exist!")

    ref_img = nib.load(str(ref_seg_path))
    raw_seg = np.asanyarray(ref_img.dataobj).astype(np.int64)
    seg5 = remap_to_5class(raw_seg)

    template_seg_nii = out_path / "template_pial_seg.nii.gz"
    template_seg_npy = out_path / "template_pial_seg_onehot.npy"
    nib.save(
        nib.Nifti1Image(seg5.astype(np.uint8), ref_img.affine, ref_img.header),
        str(template_seg_nii),
    )
    np.save(str(template_seg_npy), to_one_hot(seg5))
    print(f"  -> Saved template seg NIfTI: {template_seg_nii}")
    print(f"  -> Saved template seg One-Hot: {template_seg_npy}")


def build_template_dual_mesh(hcp_subj_dir, out_dir):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    native_surf_dir = Path(hcp_subj_dir) / "Native_surf"
    t1w_dir = Path(hcp_subj_dir) / "T1w"

    lh_pial = next(native_surf_dir.glob("*L.pial.native.surf.gii"), None)
    rh_pial = next(native_surf_dir.glob("*R.pial.native.surf.gii"), None)
    lh_white = next(native_surf_dir.glob("*L.white.native.surf.gii"), None)
    rh_white = next(native_surf_dir.glob("*R.white.native.surf.gii"), None)
    ref_seg_path = t1w_dir / "aparc+aseg.nii.gz"

    if not ref_seg_path.exists():
        raise FileNotFoundError(f"Reference segmentation {ref_seg_path} does not exist!")

    ref_img = nib.load(str(ref_seg_path))

    # 1. Pial Mesh
    pial_npz = out_path / "template_pial_mesh_repaired.npz"
    st0_pial, st1_pial = process_surface_pair(lh_pial, rh_pial, ref_img, pial_npz, surf_name="Pial (GM)")

    # 2. White Matter Mesh
    wm_npz = out_path / "template_wm_mesh_repaired.npz"
    st0_wm, st1_wm = process_surface_pair(lh_white, rh_white, ref_img, wm_npz, surf_name="White Matter")

    # 3. Template Segmentation
    print("\nRemapping template aparc+aseg to 5-class one-hot segmentation...")
    build_template_segmentation(hcp_subj_dir, out_dir)

    report = {
        "subject": Path(hcp_subj_dir).name,
        "pial_self_intersections_before": st0_pial,
        "pial_self_intersections_after": st1_pial,
        "wm_self_intersections_before": st0_wm,
        "wm_self_intersections_after": st1_wm,
    }
    with open(out_path / "template_prep_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nTemplate preparation complete! Both Pial and White Matter meshes are ready.")


def main():
    parser = argparse.ArgumentParser(description="Build and repair HCP Dual Template Meshes (WM + Pial)")
    parser.add_argument("--hcp_dir", required=True, help="Path to HCP template subject dir (e.g. .../100206)")
    parser.add_argument("--out_dir", required=True, help="Output directory for repaired template meshes")
    parser.add_argument(
        "--seg_only", action="store_true",
        help="Regenerate only the remapped template segmentation, leaving meshes untouched",
    )
    args = parser.parse_args()

    if args.seg_only:
        build_template_segmentation(args.hcp_dir, args.out_dir)
    else:
        build_template_dual_mesh(args.hcp_dir, args.out_dir)


if __name__ == "__main__":
    main()
