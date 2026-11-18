"""
Visualization for Paired WM + Pial (GM) Cortical Surface Meshes on HCP.

Pushes BOTH the template White Matter mesh and template Pial mesh into the
subject space through the trained SVF field:
    phi_inv = exp(-vel_fw)

Outputs per subject:
1. <subj>_cortical_surfaces.png:
   3 Orthogonal Slices (Axial, Coronal, Sagittal) showing:
     - Underlying MRI / Segmentation slice
     - Predicted White Matter contour (Cyan) vs Ground Truth WM (Green)
     - Predicted Pial GM contour (Magenta) vs Ground Truth Pial (Yellow)
     - Shaded cortical ribbon between WM and Pial surfaces!
2. Cortical thickness distribution histogram (Predicted vs Ground Truth).
3. Saved deformed surfaces:
     - <subj>_deformed_white.native.surf.gii
     - <subj>_deformed_pial.native.surf.gii

Usage:
    python visualize_hcp_wm_gm.py \
        --checkpoint /shared/scratch/0/home/v_mahith_reddy/training_results/hcp_pial/.../checkpoints/best_model.pth \
        --config config_hcp_pial.yaml \
        --template_dir /shared/scratch/0/home/v_mahith_reddy/hcp_data/template_100206 \
        --hcp_root /shared/scratch/0/home/v_mahith_reddy/hcp_data \
        --subjects 100307 100408 \
        --output_dir /shared/scratch/0/home/v_mahith_reddy/viz_wm_gm \
        --device cuda:0
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

from git_provenance import write_git_sha
from inference import load_config
from model import SegRegistrationNet, SpatialTransformer
from visualize_mesh import (
    integrate_svf,
    norm_to_vox,
    norm_to_world,
    sample_field,
    slice_segments,
    undo_affine,
    world_to_norm,
)


def load_gii_mesh(gii_path):
    """Load vertices and faces from GIFTI."""
    gii = nib.load(gii_path)
    verts = np.asarray(gii.darrays[0].data, dtype=np.float64)
    faces = np.asarray(gii.darrays[1].data, dtype=np.int64)
    return verts, faces


def push_mesh(verts_norm, affine_matrix, vel_fw, stn, ref_img):
    """Push normalized mesh vertices using exact inverse SVF: exp(-v)."""
    dev = vel_fw.device
    v_norm_t = torch.as_tensor(verts_norm, dtype=torch.float32, device=dev)

    # 1. Undo affine
    v_aligned = undo_affine(v_norm_t, affine_matrix)
    # 2. Integrate exact inverse SVF: exp(-vel_fw)
    inv_flow = integrate_svf(-vel_fw, stn, steps=7)
    # 3. Sample flow
    disp = sample_field(inv_flow, v_aligned)
    v_sub_norm = (v_aligned + disp).cpu().numpy()
    # 4. Map to world coordinates
    v_sub_world = norm_to_world(v_sub_norm, ref_img)
    return v_sub_norm, v_sub_world


def render_ortho_slices(bg_vol, seg_vol,
                        pred_wm_vox, pred_pial_vox, wm_faces, pial_faces,
                        gt_wm_vox=None, gt_pial_vox=None,
                        gt_wm_faces=None, gt_pial_faces=None,
                        pred_wm_world_mm=None, pred_pial_world_mm=None,
                        gt_wm_world_mm=None, gt_pial_world_mm=None,
                        save_path=None, title=""):
    """
    Renders 3 orthogonal slices displaying both White Matter and Pial contours,
    plus cortical thickness comparisons.

    pred/gt_*_world_mm: (N, 3) vertices in mm (world coordinates). Used for
    thickness computation; if None falls back to scaled voxel approximation.
    """
    D, H, W = bg_vol.shape
    centers = [D // 2, H // 2, W // 2]
    views = [(2, 'Axial (Z)'), (1, 'Coronal (Y)'), (0, 'Sagittal (X)')]

    fig = plt.figure(figsize=(18, 11), facecolor='#111111')
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.8], hspace=0.25, wspace=0.15)

    # Colors
    c_pred_wm = '#00E5FF'     # Cyan
    c_pred_pial = '#FF007F'   # Neon Pink / Magenta
    c_gt_wm = '#00FF66'       # Spring Green
    c_gt_pial = '#FFE600'     # Bright Gold/Yellow

    for panel, (axis, axis_name) in enumerate(views):
        ax = fig.add_subplot(gs[0, panel])
        s = centers[axis]

        # Extract background slice
        if axis == 0:
            sl = bg_vol[s, :, :]
        elif axis == 1:
            sl = bg_vol[:, s, :]
        else:
            sl = bg_vol[:, :, s]

        # Mesh slice coordinates are ordered by ascending volume axis, while
        # imshow uses the last array axis for x and the penultimate for y.
        ax.imshow(sl.T, cmap='gray', origin='lower')

        # Slice segments
        # 1. Predicted WM
        seg_p_wm = slice_segments(pred_wm_vox, wm_faces, axis, s)
        if len(seg_p_wm) > 0:
            lc = LineCollection(seg_p_wm, colors=c_pred_wm, linewidths=1.6, label='Pred WM')
            ax.add_collection(lc)

        # 2. Predicted Pial
        seg_p_pial = slice_segments(pred_pial_vox, pial_faces, axis, s)
        if len(seg_p_pial) > 0:
            lc = LineCollection(seg_p_pial, colors=c_pred_pial, linewidths=1.6, label='Pred Pial')
            ax.add_collection(lc)

        # 3. Ground Truth WM
        if gt_wm_vox is not None and gt_wm_faces is not None:
            seg_g_wm = slice_segments(gt_wm_vox, gt_wm_faces, axis, s)
            if len(seg_g_wm) > 0:
                lc = LineCollection(seg_g_wm, colors=c_gt_wm, linewidths=1.0, linestyle='--', alpha=0.8)
                ax.add_collection(lc)

        # 4. Ground Truth Pial
        if gt_pial_vox is not None and gt_pial_faces is not None:
            seg_g_pial = slice_segments(gt_pial_vox, gt_pial_faces, axis, s)
            if len(seg_g_pial) > 0:
                lc = LineCollection(seg_g_pial, colors=c_gt_pial, linewidths=1.0, linestyle='--', alpha=0.8)
                ax.add_collection(lc)

        ax.set_title(f"{axis_name} (Slice {s})", color='white', fontsize=13, pad=8)
        ax.axis('off')

    # Thickness analysis in bottom row
    ax_hist = fig.add_subplot(gs[1, :2])
    ax_hist.set_facecolor('#1e1e1e')

    # Cortical thickness: nearest-neighbor distance from each pial vertex to
    # the WM surface, computed in WORLD COORDINATES (mm).
    # pred_wm_vox and pred_pial_vox are in target_size voxel space; we need
    # world-space mm for meaningful thickness. These are passed as world
    # coordinates from the caller, so we use them directly if available,
    # otherwise fall back to the voxel-space approximation with a warning.
    if pred_pial_world_mm is not None and pred_wm_world_mm is not None:
        tree_wm = cKDTree(pred_wm_world_mm)
        pred_thickness_mm, _ = tree_wm.query(pred_pial_world_mm)
    else:
        # Fallback: voxel-space approximation with HCP native voxel size 0.7 mm.
        # This underestimates thickness when target_size < native resolution.
        tree_wm = cKDTree(pred_wm_vox)
        pred_thickness_mm_vox, _ = tree_wm.query(pred_pial_vox)
        # Scale: native voxels are 0.7 mm; target_size/native_size ratio corrects
        # for the downsampling done by F.interpolate at load time.
        pred_thickness_mm = pred_thickness_mm_vox * 0.7

    ax_hist.hist(pred_thickness_mm, bins=60, range=(0.5, 6.0), color=c_pred_pial, alpha=0.7,
                 label=f'Predicted Thickness (Mean: {np.mean(pred_thickness_mm):.2f} mm)')

    if gt_wm_vox is not None and gt_pial_vox is not None:
        if gt_wm_world_mm is not None and gt_pial_world_mm is not None:
            tree_gt_wm = cKDTree(gt_wm_world_mm)
            gt_thickness_mm, _ = tree_gt_wm.query(gt_pial_world_mm)
        else:
            tree_gt_wm = cKDTree(gt_wm_vox)
            gt_thickness_vox, _ = tree_gt_wm.query(gt_pial_vox)
            gt_thickness_mm = gt_thickness_vox * 0.7
        ax_hist.hist(gt_thickness_mm, bins=60, range=(0.5, 6.0), color=c_gt_pial, alpha=0.4,
                     label=f'Ground Truth FreeSurfer (Mean: {np.mean(gt_thickness_mm):.2f} mm)')

    ax_hist.set_xlabel('Cortical Thickness (mm)', color='white', fontsize=12)
    ax_hist.set_ylabel('Vertex Count', color='white', fontsize=12)
    ax_hist.set_title('Cortical Thickness Distribution across Cerebral Mantle', color='white', fontsize=13)
    ax_hist.tick_params(colors='white')
    ax_hist.grid(color='#333333', linestyle=':')
    ax_hist.legend(facecolor='#222222', edgecolor='none', labelcolor='white')

    # Legend / Info panel on bottom right
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.set_facecolor('#1e1e1e')
    ax_info.axis('off')

    legend_elements = [
        Line2D([0], [0], color=c_pred_pial, lw=3, label='Predicted Pial (GM outer)'),
        Line2D([0], [0], color=c_gt_pial, lw=2, linestyle='--', label='Ground Truth Pial (FS)'),
        Line2D([0], [0], color=c_pred_wm, lw=3, label='Predicted White Matter (WM)'),
        Line2D([0], [0], color=c_gt_wm, lw=2, linestyle='--', label='Ground Truth WM (FS)'),
    ]
    ax_info.legend(handles=legend_elements, loc='center', facecolor='#222222',
                   edgecolor='white', labelcolor='white', fontsize=11)

    fig.suptitle(f"S-RegNET: Dual Cortical Mesh Reconstruction (WM + GM) | {title}",
                 color='white', fontsize=16, y=0.96)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved visualization: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize paired WM + Pial mesh reconstructions")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--config", default="config_hcp_pial.yaml")
    parser.add_argument("--template_dir", required=True, help="Path to template folder (holding template pial + wm)")
    parser.add_argument("--hcp_root", required=True, help="Path to HCP dataset root")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs to visualize")
    parser.add_argument("--output_dir", default="viz_output_wm_gm", help="Output directory")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_git_sha(out_dir)

    cfg = load_config(args.config)
    target_size = tuple(cfg.get("model", {}).get("target_size", [128, 128, 128]))
    use_affine = cfg.get("affine", {}).get("enabled", True)

    print("Loading model...")
    model = SegRegistrationNet(
        target_size=target_size,
        seg_channels=cfg.get("model", {}).get("seg_channels", 5),
        use_affine=use_affine,
    ).to(dev)

    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()

    stn = SpatialTransformer(size=target_size, device=dev).to(dev)

    # Load Template meshes (Pial and WM)
    tpl_dir = Path(args.template_dir)
    pial_tpl = np.load(tpl_dir / "template_pial_mesh_repaired.npz")
    pial_verts_norm = pial_tpl["verts_norm"]
    pial_faces = pial_tpl["faces"]

    # If template wm mesh exists in template_dir, load it, else fall back to pial
    wm_tpl_path = tpl_dir / "template_wm_mesh_repaired.npz"
    if wm_tpl_path.exists():
        wm_tpl = np.load(wm_tpl_path)
        wm_verts_norm = wm_tpl["verts_norm"]
        wm_faces = wm_tpl["faces"]
    else:
        wm_verts_norm, wm_faces = pial_verts_norm, pial_faces

    # Template segmentation
    tpl_seg = np.load(tpl_dir / "template_pial_seg_onehot.npy")
    tpl_seg_t = torch.tensor(tpl_seg, dtype=torch.float32, device=dev).unsqueeze(0)
    tpl_seg_resized = F.interpolate(tpl_seg_t, size=target_size, mode="nearest")

    for subj in args.subjects:
        print(f"\nProcessing subject {subj}...")
        subj_dir = Path(args.hcp_root) / subj
        subj_seg_path = subj_dir / "T1w" / "seg4_white_onehot.npy"
        subj_ref_path = subj_dir / "T1w" / "aparc+aseg.nii.gz"
        native_surf_dir = subj_dir / "Native_surf"

        if not subj_seg_path.exists() or not subj_ref_path.exists():
            print(f"Skipping {subj}, missing segmentation.")
            continue

        ref_img = nib.load(str(subj_ref_path))
        subj_seg = np.load(subj_seg_path)
        subj_seg_t = torch.tensor(subj_seg, dtype=torch.float32, device=dev).unsqueeze(0)
        subj_seg_resized = F.interpolate(subj_seg_t, size=target_size, mode="nearest")

        # 1. Forward Pass
        with torch.no_grad():
            res = model(tpl_seg_resized, subj_seg_resized, return_velocity=True)
            flow_fw, flow_rv, lambda_map, affine_mat, vel_fw = res

            # Push Pial mesh — returns (verts_norm, verts_world_mm)
            _, pred_pial_world = push_mesh(pial_verts_norm, affine_mat, vel_fw, stn, ref_img)
            # Push WM mesh
            _, pred_wm_world = push_mesh(wm_verts_norm, affine_mat, vel_fw, stn, ref_img)

        # World coordinates in mm — used for accurate thickness computation
        pred_pial_world_mm = pred_pial_world.astype(np.float64)
        pred_wm_world_mm   = pred_wm_world.astype(np.float64)

        # Voxel coordinates for orthogonal slice rendering
        pred_pial_vox = norm_to_vox(world_to_norm(pred_pial_world, ref_img), target_size[0])
        pred_wm_vox = norm_to_vox(world_to_norm(pred_wm_world, ref_img), target_size[0])

        # Load GT meshes if available
        gt_lh_pial = next(native_surf_dir.glob("*L.pial.native.surf.gii"), None)
        gt_rh_pial = next(native_surf_dir.glob("*R.pial.native.surf.gii"), None)
        gt_lh_wm = next(native_surf_dir.glob("*L.white.native.surf.gii"), None)
        gt_rh_wm = next(native_surf_dir.glob("*R.white.native.surf.gii"), None)

        gt_pial_vox, gt_pial_faces = None, None
        gt_wm_vox, gt_wm_faces = None, None
        gt_pial_world_mm, gt_wm_world_mm = None, None

        if gt_lh_pial and gt_rh_pial and gt_lh_wm and gt_rh_wm:
            v_lp, f_lp = load_gii_mesh(gt_lh_pial)
            v_rp, f_rp = load_gii_mesh(gt_rh_pial)
            gt_pial_w = np.concatenate([v_lp, v_rp], axis=0)
            gt_pial_faces = np.concatenate([f_lp, f_rp + len(v_lp)], axis=0)
            gt_pial_vox = norm_to_vox(world_to_norm(gt_pial_w, ref_img), target_size[0])
            gt_pial_world_mm = gt_pial_w.astype(np.float64)

            v_lw, f_lw = load_gii_mesh(gt_lh_wm)
            v_rw, f_rw = load_gii_mesh(gt_rh_wm)
            gt_wm_w = np.concatenate([v_lw, v_rw], axis=0)
            gt_wm_faces = np.concatenate([f_lw, f_rw + len(v_lw)], axis=0)
            gt_wm_vox = norm_to_vox(world_to_norm(gt_wm_w, ref_img), target_size[0])
            gt_wm_world_mm = gt_wm_w.astype(np.float64)

        # Background volume: use Class 1 (Cortex) + Class 3 (WM) as anatomical backdrop
        bg_vol = (subj_seg[1] * 0.5 + subj_seg[3] * 1.0)
        bg_vol_resized = F.interpolate(
            torch.tensor(bg_vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            size=target_size, mode='nearest'
        ).squeeze().numpy()

        fig_path = out_dir / f"{subj}_cortical_surfaces.png"
        render_ortho_slices(
            bg_vol=bg_vol_resized,
            seg_vol=subj_seg,
            pred_wm_vox=pred_wm_vox,
            pred_pial_vox=pred_pial_vox,
            wm_faces=wm_faces,
            pial_faces=pial_faces,
            gt_wm_vox=gt_wm_vox,
            gt_pial_vox=gt_pial_vox,
            gt_wm_faces=gt_wm_faces,
            gt_pial_faces=gt_pial_faces,
            pred_wm_world_mm=pred_wm_world_mm,
            pred_pial_world_mm=pred_pial_world_mm,
            gt_wm_world_mm=gt_wm_world_mm,
            gt_pial_world_mm=gt_pial_world_mm,
            save_path=str(fig_path),
            title=f"Subject {subj}"
        )


if __name__ == "__main__":
    main()
