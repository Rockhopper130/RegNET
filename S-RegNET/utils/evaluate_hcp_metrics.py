"""Evaluate one HCP subject's voxel and paired cortical-surface metrics.

Reports per-class Dice and voxel-surface HD95, plus vertex-to-vertex HD95,
self-intersection percentage, and triangle-flip percentage for pial and white
meshes. Mesh metrics are measured on the raw pushed surfaces (no repair).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

from git_provenance import write_git_sha
from inference import load_config
from losses import compute_dice_score
from mesh_flip_probe import self_intersections
from model import SegRegistrationNet, SpatialTransformer
from visualize_mesh import (
    integrate_svf,
    norm_to_world,
    sample_field,
    triangle_flip_fraction,
    undo_affine,
)


CLASS_NAMES = ("background", "cortex_gm", "other_brain", "white_matter_interior", "csf")


def load_gii_mesh(path):
    gii = nib.load(str(path))
    return (np.asarray(gii.darrays[0].data, dtype=np.float64),
            np.asarray(gii.darrays[1].data, dtype=np.int64))


def find_surface_pair(subject_dir, surface):
    surf_dir = subject_dir / "Native_surf"
    lh = next(surf_dir.glob(f"*L.{surface}.native.surf.gii"), None)
    rh = next(surf_dir.glob(f"*R.{surface}.native.surf.gii"), None)
    if lh is None or rh is None:
        raise FileNotFoundError(
            f"Expected both hemispheres of {surface} surface under {surf_dir}")
    return lh, rh


def combine_hemispheres(lh_path, rh_path):
    lh_verts, lh_faces = load_gii_mesh(lh_path)
    rh_verts, rh_faces = load_gii_mesh(rh_path)
    verts = np.concatenate((lh_verts, rh_verts), axis=0)
    faces = np.concatenate((lh_faces, rh_faces + len(lh_verts)), axis=0)
    return verts, faces, len(lh_verts)


def surface_hd95(pred, target, spacing):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not pred.any() or not target.any():
        return float("nan")

    pred_surface = pred ^ binary_erosion(pred, border_value=0)
    target_surface = target ^ binary_erosion(target, border_value=0)
    d_target = distance_transform_edt(~target_surface, sampling=spacing)
    d_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate((d_target[pred_surface], d_pred[target_surface]))
    return float(np.percentile(distances, 95))


def vertex_hd95(pred, target, n_pred_lh, n_target_lh):
    distances = []
    for pred_slice, target_slice in (
        (slice(0, n_pred_lh), slice(0, n_target_lh)),
        (slice(n_pred_lh, None), slice(n_target_lh, None)),
    ):
        pred_to_target = cKDTree(target[target_slice]).query(pred[pred_slice])[0]
        target_to_pred = cKDTree(pred[pred_slice]).query(target[target_slice])[0]
        distances.extend((pred_to_target, target_to_pred))
    return float(np.percentile(np.concatenate(distances), 95))


def push_mesh(verts_norm, affine_matrix, vel_fw, stn, ref_img):
    verts = torch.as_tensor(verts_norm, dtype=torch.float32, device=vel_fw.device)
    aligned = undo_affine(verts, affine_matrix)
    inverse_flow = integrate_svf(-vel_fw, stn, steps=7)
    displaced = aligned + sample_field(inverse_flow, aligned)
    return norm_to_world(displaced.cpu().numpy(), ref_img)


def evaluate_mesh(tag, template, gt_paths, affine_matrix, vel_fw, stn, ref_img):
    gt_verts, _, gt_n_lh = combine_hemispheres(*gt_paths)
    pred_verts = push_mesh(
        template["verts_norm"], affine_matrix, vel_fw, stn, ref_img)
    metrics, _ = self_intersections(pred_verts, template["faces"], tag)
    return {
        f"{tag}_vertex_hd95_mm": vertex_hd95(
            pred_verts, gt_verts, int(template["n_lh"]), gt_n_lh),
        f"{tag}_self_intersecting_faces_pct": metrics["si_faces_pct"],
        f"{tag}_triangle_flips_pct": triangle_flip_fraction(
            template["verts_world"], pred_verts, template["faces"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--template_dir", required=True)
    parser.add_argument("--hcp_root", required=True)
    parser.add_argument("--subject", required=True, help="Single HCP subject ID")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    target_size = tuple(cfg["model"]["target_size"])
    subject_dir = Path(args.hcp_root) / args.subject
    seg_path = subject_dir / "T1w" / "seg4_white_onehot.npy"
    ref_path = subject_dir / "T1w" / "aparc+aseg.nii.gz"
    if not seg_path.is_file() or not ref_path.is_file():
        raise FileNotFoundError(f"Missing segmentation or reference image for {args.subject}")
    ref_img = nib.load(str(ref_path))

    template_dir = Path(args.template_dir)
    pial_template = np.load(template_dir / "template_pial_mesh_repaired.npz")
    white_template = np.load(template_dir / "template_wm_mesh_repaired.npz")
    pial_gt = find_surface_pair(subject_dir, "pial")
    white_gt = find_surface_pair(subject_dir, "white")

    model = SegRegistrationNet(
        target_size=target_size,
        seg_channels=cfg.get("model", {}).get("seg_channels", 5),
        use_affine=cfg.get("affine", {}).get("enabled", True),
        head=cfg.get("model", {}).get("head", "default"),
        lowpass_size=cfg.get("model", {}).get("lowpass_size"),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    stn = SpatialTransformer(size=target_size, device=device).to(device)

    template_seg = np.load(template_dir / "template_pial_seg_onehot.npy")
    template_seg_t = torch.as_tensor(
        template_seg, dtype=torch.float32, device=device).unsqueeze(0)
    template_seg_t = F.interpolate(template_seg_t, size=target_size, mode="nearest")
    subject_seg = np.load(seg_path)
    subject_seg_t = torch.as_tensor(
        subject_seg, dtype=torch.float32, device=device).unsqueeze(0)
    subject_seg_t = F.interpolate(subject_seg_t, size=target_size, mode="nearest")

    with torch.no_grad():
        flow_fw, _, _, affine_matrix, vel_fw = model(
            template_seg_t, subject_seg_t, return_velocity=True)
        warped_seg = stn(template_seg_t, flow_fw)
        dice, _ = compute_dice_score(warped_seg, subject_seg_t, num_classes=5)

    pred_labels = warped_seg[0].argmax(dim=0).cpu().numpy()
    target_labels = subject_seg_t[0].argmax(dim=0).cpu().numpy()
    source_shape = np.asarray(subject_seg.shape[1:], dtype=np.float64)
    resized_spacing = (
        np.asarray(ref_img.header.get_zooms()[:3], dtype=np.float64)
        * source_shape / np.asarray(target_size, dtype=np.float64)
    )
    row = {"subject": args.subject}
    for class_id, name in enumerate(CLASS_NAMES):
        row[f"dice_{name}"] = float(dice[class_id])
        row[f"hd95_{name}_mm"] = surface_hd95(
            pred_labels == class_id, target_labels == class_id, resized_spacing)

    row.update(evaluate_mesh(
        "pial", pial_template, pial_gt, affine_matrix, vel_fw, stn, ref_img))
    row.update(evaluate_mesh(
        "white", white_template, white_gt, affine_matrix, vel_fw, stn, ref_img))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_git_sha(out_dir)
    csv_path = out_dir / f"{args.subject}_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    json_path = out_dir / f"{args.subject}_metrics.json"
    json_path.write_text(json.dumps(row, indent=2))

    print("\nClass metrics (native-equivalent voxel spacing):")
    print(f"{'Class':<26} {'Dice':>8} {'HD95 (mm)':>12}")
    for name in CLASS_NAMES:
        print(f"{name:<26} {row[f'dice_{name}']:8.4f} "
              f"{row[f'hd95_{name}_mm']:12.2f}")
    print("\nRaw mesh metrics:")
    print(f"{'Mesh':<12} {'Vertex HD95 (mm)':>18} "
          f"{'Self-intersecting faces':>25} {'Triangle flips':>16}")
    for tag in ("pial", "white"):
        print(f"{tag:<12} {row[f'{tag}_vertex_hd95_mm']:18.2f} "
              f"{row[f'{tag}_self_intersecting_faces_pct']:24.4f}% "
              f"{row[f'{tag}_triangle_flips_pct']:15.4f}%")
    print(f"\nCSV: {csv_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()
