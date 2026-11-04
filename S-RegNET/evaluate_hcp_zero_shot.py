"""
Zero-Shot Evaluation of S-RegNET on HCP Pial (GM) Surface Meshes.

Performs:
1. Loads an existing trained S-RegNET checkpoint (e.g. OASIS baseline/distilled).
2. Uses HCP 100206 repaired pial mesh as the moving template.
3. For each HCP test subject:
   - Warps the template segmentation to the subject segmentation.
   - Pushes the template pial mesh using the exact inverse SVF: exp(-v).
   - Compares the deformed mesh against the subject's ground-truth FreeSurfer
     pial mesh (*pial.native.surf.gii).
4. Computes:
   - Symmetric Mean Distance (mm)
   - 95% Hausdorff Distance (HD95 mm)
   - Cortical triangle flips %
   - Self-intersection percentage (si_faces_pct)
   - Tissue Dice scores (C1 Cortex, C3 WM Interior)
5. Outputs a detailed CSV and JSON summary.

Usage:
    python evaluate_hcp_zero_shot.py \
        --checkpoint /shared/scratch/0/home/v_nishchay_nilabh/training_results/mahith_experiment/20260801_083414/checkpoints/best_model.pth \
        --config config.yaml \
        --template_dir /shared/scratch/0/home/v_mahith_reddy/hcp_data/template_100206 \
        --hcp_root /shared/scratch/0/home/v_mahith_reddy/hcp_data \
        --subjects 100307 100408 100610 101006 101107 101309 101410 101915 102008 \
        --output_dir /shared/scratch/0/home/v_mahith_reddy/zero_shot_eval \
        --device cuda:0
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm import tqdm

# Add root and utils
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

from git_provenance import write_git_sha
from inference import load_config
from losses import compute_dice_score
from model import SegRegistrationNet, SpatialTransformer, load_bandlimited_warm_start
from repair_template_mesh import self_intersections
from visualize_mesh import (
    integrate_svf,
    norm_to_world,
    sample_field,
    triangle_flip_fraction,
    undo_affine,
    world_to_norm,
)


def load_gii_mesh(gii_path):
    """Load vertices (world mm) and faces from GIFTI."""
    gii = nib.load(gii_path)
    verts = np.asarray(gii.darrays[0].data, dtype=np.float64)
    faces = np.asarray(gii.darrays[1].data, dtype=np.int64)
    return verts, faces


def sym_dist_hemi(mesh_w, gt_w, n_lh, gt_n_lh):
    """Per-hemisphere nearest-neighbor distances in mm."""
    m2g_lh = cKDTree(gt_w[:gt_n_lh]).query(mesh_w[:n_lh])[0]
    g2m_lh = cKDTree(mesh_w[:n_lh]).query(gt_w[:gt_n_lh])[0]

    m2g_rh = cKDTree(gt_w[gt_n_lh:]).query(mesh_w[n_lh:])[0]
    g2m_rh = cKDTree(mesh_w[n_lh:]).query(gt_w[gt_n_lh:])[0]

    all_dists = np.concatenate([m2g_lh, g2m_lh, m2g_rh, g2m_rh])
    return {
        "sym_mean_mm": float(np.mean(all_dists)),
        "sym_median_mm": float(np.median(all_dists)),
        "sym_hd95_mm": float(np.percentile(all_dists, 95)),
        "sym_max_mm": float(np.max(all_dists)),
        "lh_mean_mm": float(np.mean(np.concatenate([m2g_lh, g2m_lh]))),
        "rh_mean_mm": float(np.mean(np.concatenate([m2g_rh, g2m_rh]))),
    }


def push_pial_mesh(verts_norm, affine_matrix, vel_fw, stn, ref_img):
    """
    Push template pial mesh with the exact inverse SVF: exp(-vel_fw).
    Sequence: undo affine -> apply exp(-v) -> convert to subject world mm.
    """
    dev = vel_fw.device
    v_norm_t = torch.as_tensor(verts_norm, dtype=torch.float32, device=dev)

    # 1. Undo affine pre-alignment: template space -> affine-aligned space
    v_aligned = undo_affine(v_norm_t, affine_matrix)

    # 2. Integrate exact inverse SVF: exp(-vel_fw)
    inv_flow = integrate_svf(-vel_fw, stn, steps=7)

    # 3. Sample inverse flow at vertex coordinates
    disp = sample_field(inv_flow, v_aligned)
    v_subject_norm = (v_aligned + disp).cpu().numpy()

    # 4. Map from normalized coordinates to subject world mm
    v_subject_world = norm_to_world(v_subject_norm, ref_img)
    return v_subject_world


def main():
    parser = argparse.ArgumentParser(description="Zero-shot S-RegNET evaluation on HCP Pial surfaces")
    parser.add_argument("--checkpoint", required=True, help="Path to trained S-RegNET checkpoint (.pth)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--template_dir", required=True, help="Path to template folder holding template_pial_mesh_repaired.npz")
    parser.add_argument("--hcp_root", required=True, help="Root folder of HCP data")
    parser.add_argument("--subjects", nargs="+", required=True, help="List of subject IDs to evaluate")
    parser.add_argument("--output_dir", default="zero_shot_results", help="Directory to save evaluation results")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--self_int", action="store_true", help="Calculate self-intersection percentage (slow)")
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_git_sha(out_dir)

    cfg = load_config(args.config)
    target_size = tuple(cfg.get("model", {}).get("target_size", [128, 128, 128]))
    use_affine = cfg.get("affine", {}).get("enabled", True)
    head_type = cfg.get("model", {}).get("head", "default")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = SegRegistrationNet(
        target_size=target_size,
        seg_channels=cfg.get("model", {}).get("seg_channels", 5),
        use_affine=use_affine,
        head=head_type,
    ).to(dev)

    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    stn = SpatialTransformer(size=target_size, device=dev).to(dev)

    # Load Template Mesh and Segmentation
    tpl_dir = Path(args.template_dir)
    tpl_mesh_path = tpl_dir / "template_pial_mesh_repaired.npz"
    tpl_seg_path = tpl_dir / "template_pial_seg_onehot.npy"

    print(f"Loading Template Mesh: {tpl_mesh_path}")
    tpl_mesh = np.load(tpl_mesh_path)
    tpl_verts_norm = tpl_mesh["verts_norm"]
    tpl_faces = tpl_mesh["faces"]
    tpl_n_lh = int(tpl_mesh["n_lh"])

    print(f"Loading Template Seg: {tpl_seg_path}")
    tpl_seg = np.load(tpl_seg_path)
    tpl_seg_t = torch.tensor(tpl_seg, dtype=torch.float32, device=dev).unsqueeze(0)
    tpl_seg_resized = F.interpolate(tpl_seg_t, size=target_size, mode="nearest")

    results = []
    print(f"\nStarting Zero-Shot Evaluation on {len(args.subjects)} HCP subjects...")

    for subj in tqdm(args.subjects, desc="Zero-Shot Eval"):
        subj_dir = Path(args.hcp_root) / subj
        subj_seg_path = subj_dir / "T1w" / "seg4_white_onehot.npy"
        subj_ref_path = subj_dir / "T1w" / "aparc+aseg.nii.gz"
        native_surf_dir = subj_dir / "Native_surf"

        lh_gt_path = next(native_surf_dir.glob("*L.pial.native.surf.gii"), None)
        rh_gt_path = next(native_surf_dir.glob("*R.pial.native.surf.gii"), None)

        if not subj_seg_path.exists() or not subj_ref_path.exists() or not lh_gt_path or not rh_gt_path:
            print(f"[Skip] Missing files for subject {subj}")
            continue

        ref_img = nib.load(str(subj_ref_path))
        subj_seg = np.load(subj_seg_path)
        subj_seg_t = torch.tensor(subj_seg, dtype=torch.float32, device=dev).unsqueeze(0)
        subj_seg_resized = F.interpolate(subj_seg_t, size=target_size, mode="nearest")

        # 1. Forward Pass
        with torch.no_grad():
            res = model(tpl_seg_resized, subj_seg_resized, return_velocity=True)
            flow_fw, flow_rv, lambda_map, affine_mat, vel_fw = res

            # Voxel alignment scores
            warped_tpl = stn(tpl_seg_resized, flow_fw)
            dice_scores, mean_fg_dice = compute_dice_score(warped_tpl, subj_seg_resized, num_classes=5)

            # 2. Push Template Pial Mesh using exact inverse SVF
            pushed_verts_w = push_pial_mesh(tpl_verts_norm, affine_mat, vel_fw, stn, ref_img)

        # 3. Load Ground Truth Pial Meshes (LH + RH)
        gt_v_l, _ = load_gii_mesh(lh_gt_path)
        gt_v_r, _ = load_gii_mesh(rh_gt_path)
        gt_verts_w = np.concatenate([gt_v_l, gt_v_r], axis=0)
        gt_n_lh = len(gt_v_l)

        # 4. Compute Surface Distances
        dist_metrics = sym_dist_hemi(pushed_verts_w, gt_verts_w, tpl_n_lh, gt_n_lh)

        # Cortical flips %
        flip_pct = triangle_flip_fraction(tpl_mesh["verts_world"], pushed_verts_w, tpl_faces)

        # Optional: Self-intersections
        si_pct = None
        if args.self_int:
            si_res = self_intersections(pushed_verts_w, tpl_faces)
            si_pct = si_res["si_faces_pct"]

        row = {
            "subject": subj,
            "dice_c1_cortex": float(dice_scores[1]),
            "dice_c3_wm": float(dice_scores[3]),
            "dice_fg_mean": float(mean_fg_dice),
            "sym_mean_mm": dist_metrics["sym_mean_mm"],
            "sym_median_mm": dist_metrics["sym_median_mm"],
            "sym_hd95_mm": dist_metrics["sym_hd95_mm"],
            "sym_max_mm": dist_metrics["sym_max_mm"],
            "flip_pct": flip_pct,
            "si_faces_pct": si_pct,
        }
        results.append(row)

    # Save CSV
    csv_file = out_dir / "zero_shot_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Summary JSON
    summary = {
        "num_subjects": len(results),
        "mean_cortex_dice": float(np.mean([r["dice_c1_cortex"] for r in results])),
        "mean_wm_dice": float(np.mean([r["dice_c3_wm"] for r in results])),
        "mean_sym_distance_mm": float(np.mean([r["sym_mean_mm"] for r in results])),
        "mean_hd95_mm": float(np.mean([r["sym_hd95_mm"] for r in results])),
        "mean_flip_pct": float(np.mean([r["flip_pct"] for r in results])),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ Zero-Shot Evaluation Results ================")
    print(f"Subjects evaluated:     {summary['num_subjects']}")
    print(f"Cortical GM Dice (C1):  {summary['mean_cortex_dice']:.4f}")
    print(f"WM Interior Dice (C3):  {summary['mean_wm_dice']:.4f}")
    print(f"Mean Surface Distance:  {summary['mean_sym_distance_mm']:.4f} mm")
    print(f"95% Hausdorff Distance: {summary['mean_hd95_mm']:.4f} mm")
    print(f"Triangle Flips:         {summary['mean_flip_pct']:.4f} %")
    print(f"Results saved to: {csv_file}")
    print("==============================================================")


if __name__ == "__main__":
    main()

