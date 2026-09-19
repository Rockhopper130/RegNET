"""Export template, sample, and SVF-deformed WM meshes as FreeSurfer surfaces.

The input surface files are FreeSurfer's native ``lh.white``/``rh.white`` binary
geometry. The exported copies use a ``.surf`` suffix only to make the file type
obvious to recipients; FreeSurfer itself does not require that extension.
"""
import argparse
import os
import re
import shutil

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from keyreg import SVFReg, identity_grid_vol, load_seg, svf_integrate


ROOT = "/shared/scratch/0/home/v_vijay_bala_mahalingam"

ap = argparse.ArgumentParser()
ap.add_argument("--subject", required=True, help="target, e.g. OASIS_OAS1_0374_MR1")
ap.add_argument("--fixed_name", default="seg4_onehot_clinical.npy")
ap.add_argument("--template_subject", default="OASIS_OAS1_0001_MR1")
ap.add_argument("--run", default="svf_E_full")
ap.add_argument("--surf_root", default=f"{ROOT}/oasis_mri_outputs")
ap.add_argument("--surf_suffix", default="_clinical")
ap.add_argument("--neurite", default=f"{ROOT}/neurite_oasis")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()


def short_id(subject):
    match = re.search(r"(OAS1_\d+)", subject)
    if not match:
        raise ValueError(f"cannot extract OAS1 id from {subject}")
    return match.group(1)


def surf_paths(subject):
    directory = os.path.join(args.surf_root, short_id(subject) + args.surf_suffix, "surf")
    return {hemi: os.path.join(directory, f"{hemi}.white") for hemi in ("lh", "rh")}


def read_white(subject):
    parts = []
    for hemi, path in surf_paths(subject).items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        vertices, faces, metadata = nib.freesurfer.read_geometry(path, read_metadata=True)
        parts.append((hemi, vertices, faces.astype(np.int32), metadata, path))
    return parts


def surface_to_norm(vertices, metadata, subject):
    """FreeSurfer surface RAS -> scanner RAS -> model-normalized (x,y,z)."""
    ref = nib.load(os.path.join(args.neurite, subject, "aligned_seg4.nii.gz"))
    world = vertices + metadata.get("cras", np.zeros(3))
    vox = (np.c_[world, np.ones(len(world))] @ np.linalg.inv(ref.affine).T)[:, :3]
    nd = np.asarray(ref.shape, dtype=np.float64)
    return np.stack([vox[:, 2] / (nd[2] - 1) * 2 - 1,
                     vox[:, 1] / (nd[1] - 1) * 2 - 1,
                     vox[:, 0] / (nd[0] - 1) * 2 - 1], 1).astype(np.float32)


def norm_to_surface(vertices, metadata, subject):
    """Model-normalized (x,y,z) -> scanner RAS -> target surface RAS."""
    ref = nib.load(os.path.join(args.neurite, subject, "aligned_seg4.nii.gz"))
    nd = np.asarray(ref.shape, dtype=np.float64)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    vox = np.stack([(z + 1) / 2 * (nd[0] - 1),
                    (y + 1) / 2 * (nd[1] - 1),
                    (x + 1) / 2 * (nd[2] - 1)], 1)
    world = (np.c_[vox, np.ones(len(vox))] @ ref.affine.T)[:, :3]
    return (world - metadata.get("cras", np.zeros(3))).astype(np.float32)


def sample_field(field, points):
    grid = points.view(1, 1, 1, -1, 3)
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)[0, :, 0, 0, :].T


template_parts = read_white(args.template_subject)
sample_parts = read_white(args.subject)

template_norm_parts = [surface_to_norm(v, meta, args.template_subject)
                       for _, v, _, meta, _ in template_parts]
template_norm = np.concatenate(template_norm_parts)
points = torch.as_tensor(template_norm, dtype=torch.float32, device=args.device)

template_seg = load_seg(
    os.path.join(args.neurite, args.template_subject, "seg4_onehot.npy"),
    (128, 128, 128)).unsqueeze(0).to(args.device)
fixed_seg = load_seg(
    os.path.join(args.neurite, args.subject, args.fixed_name),
    (128, 128, 128)).unsqueeze(0).to(args.device)

checkpoint = torch.load(os.path.join(ROOT, "keyreg_runs", args.run, "best.pth"),
                        map_location=args.device, weights_only=False)
cfg = checkpoint["args"]
model = SVFReg(target=cfg["target"], delta=cfg["delta"],
               int_steps=cfg["int_steps"], K=cfg["K"]).to(args.device)
model.load_state_dict(checkpoint["model"])
model.eval()

with torch.no_grad():
    x = torch.cat([template_seg, fixed_seg], 1)
    if model.kp_guided:
        x = torch.cat([x, model._sal(template_seg), model._sal(fixed_seg)], 1)
    velocity = model.flow(x)
    identity = identity_grid_vol(128, args.device)
    inverse = svf_integrate(-velocity, model.int_steps, identity)
    deformed_norm = points + sample_field(inverse, points)
deformed_norm = deformed_norm.cpu().numpy()

os.makedirs(args.out, exist_ok=True)
offset = 0
for (hemi, _, faces, _, template_path), (_, _, _, target_meta, sample_path), part in zip(
        template_parts, sample_parts, template_norm_parts):
    count = len(part)
    deformed_ras = norm_to_surface(deformed_norm[offset:offset + count], target_meta, args.subject)
    nib.freesurfer.write_geometry(
        os.path.join(args.out, f"{hemi}.deformed.white.surf"),
        deformed_ras, faces, create_stamp="KeyReg SVF exact inverse",
        volume_info=target_meta)
    shutil.copy2(template_path, os.path.join(args.out, f"{hemi}.template.white.surf"))
    shutil.copy2(sample_path, os.path.join(args.out, f"{hemi}.sample.white.surf"))
    offset += count

print(f"wrote template/sample/deformed lh+rh surfaces -> {args.out}")
