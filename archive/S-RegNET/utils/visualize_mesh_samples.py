"""
Visualize N samples for the FFD model: push the genus-0 template mesh into
subject space (model.push_points_to_sample) and overlay three WM surfaces on
orthogonal slices.

    blue   - deformed template mesh
    green  - subject's GT white surface (FreeSurfer lh/rh.white)
    orange - SynthSeg WM marching-cubes surface (the noisy input, for context)

Per-vertex distance is from each deformed vertex to the nearest GT white-surface
vertex, in mm. Inputs follow config experiment routing.

Run from the S-RegNET directory (needs torch + nibabel):
    python utils/visualize_mesh_samples.py \\
        --checkpoint <.../checkpoints/best_model.pth> \\
        --output_dir images_mesh --num_samples 5 --device cuda:0 --use_affine
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

print("[viz] starting — importing numpy/torch/matplotlib "
      "(can take 30-60s on a cold/busy node, no output meanwhile)...", flush=True)

import numpy as np
import nibabel as nib
import torch
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "eval")]

from inference import setup_inference
from get_data import SegDataset
from wm_template import load_template_wm_mesh
from experiment_routing import build_step_inputs
from compare_wm_surface import wm_surface                       # marching-cubes WM extractor
from overlay_surface import slice_segments as surface_contour, world_to_norm

MM_PER_NORM = 256.0 / 2.0     # normalized surface distance -> mm
WM_LABEL = 3                  # 5-class: 0 bg, 1 cortex, 2 subGM, 3 WM, 4 CSF
GT_SURF_C = 'green'           # GT white surface
SS_SURF_C = 'orange'          # SynthSeg WM marching-cubes surface
DEFORM_C  = 'blue'            # deformed genus-0 template mesh


def norm_to_vox(verts, n):
    """Normalized (x,y,z) in [-1,1] -> voxel coords (i,j,k) of the n^3 grid
    (align_corners=False), with the axis map x<->k, y<->j, z<->i."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i = ((z + 1) * n - 1) / 2
    j = ((y + 1) * n - 1) / 2
    k = ((x + 1) * n - 1) / 2
    return np.stack([i, j, k], axis=1)


def slice_segments(vox, faces, scal, axis, s):
    """Mesh-plane contour at {axis == s}, interpolating a per-vertex scalar onto
    each segment (like overlay_surface.slice_segments, with the scalar carried at
    the same parameter t). Returns (segments (K,2,2) in-plane voxel coords,
    seg_scalar (K,))."""
    p, q = [ax for ax in range(3) if ax != axis]
    d = vox[:, axis] - s
    xy = vox[:, [p, q]]
    df, pf, sf = d[faces], xy[faces], scal[faces]      # (M,3) (M,3,2) (M,3)

    pts, svals, masks = [], [], []
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        di, dj = df[:, i], df[:, j]
        cross = (di > 0) != (dj > 0)
        denom = di - dj
        safe = np.abs(denom) > 1e-12
        t = np.where(safe, di / np.where(safe, denom, 1.0), 0.5)
        pts.append(pf[:, i, :] + t[:, None] * (pf[:, j, :] - pf[:, i, :]))
        svals.append(sf[:, i] + t * (sf[:, j] - sf[:, i]))
        masks.append(cross)

    P = np.stack(pts, axis=1)
    S = np.stack(svals, axis=1)
    Mk = np.stack(masks, axis=1)
    sel = Mk.sum(1) == 2
    seg = P[sel][Mk[sel]].reshape(-1, 2, 2)
    sval = S[sel][Mk[sel]].reshape(-1, 2).mean(1)
    return seg, sval


def load_subject_white_norm(subject_dir, ref_nii, lh_name, rh_name):
    """Load a subject's FreeSurfer lh/rh white surfaces in the subject's
    normalized [-1,1] (x,y,z) coords, the same frame the pushed template mesh
    lives in. Surfaces are read from /meshes/<subj>/{lh,rh}.white via the same
    surface-RAS + cras -> world_to_norm(ref) chain as wm_template. Returns
    (verts_norm (N,3) f32, faces (M,3) i64), or (None, None) if missing."""
    mesh_dir = subject_dir.replace('/scans/', '/meshes/')
    lh_p, rh_p = os.path.join(mesh_dir, lh_name), os.path.join(mesh_dir, rh_name)
    if not (os.path.exists(lh_p) and os.path.exists(rh_p)):
        return None, None
    ref = nib.load(ref_nii)

    def _hemi(path):
        coords, faces, meta = nib.freesurfer.read_geometry(path, read_metadata=True)
        world = coords + meta.get('cras', np.zeros(3))                 # surface-RAS -> scanner RAS
        return world_to_norm(world, ref).astype(np.float32), np.asarray(faces, dtype=np.int64)

    lv, lf = _hemi(lh_p)
    rv, rf = _hemi(rh_p)
    verts = np.concatenate([lv, rv], axis=0)
    faces = np.concatenate([lf, rf + len(lv)], axis=0)                 # rh indices offset
    return verts, faces


@torch.no_grad()
def deform_template_mesh(ctx, template_verts, input_seg):
    """Run the model and push the genus-0 template mesh into subject space via the
    field's numerical inverse. Returns the pushed vertices in subject normalized
    coords (N,3)."""
    model, template_seg = ctx['model'], ctx['template_seg']
    cps_list, affine_matrix = model(template_seg, input_seg)
    pushed, _res = model.push_points_to_sample(template_verts, cps_list, affine_matrix)  # (N,3) norm
    return pushed.cpu().numpy()


def render_sample(name, vox, faces, dist_mm, gt_wm, gt_surf, ss_surf, n_lh, output, supervision_target):
    """3 ortho slices + a distance histogram. Over the GT WM mask (gray) each slice
    shows the GT white surface (green) and the SynthSeg WM marching-cubes surface
    (orange) as plain contours, plus the deformed template mesh as a solid blue
    contour. gt_surf/ss_surf are (verts, faces) in the same n^3 voxel frame as vox."""
    # Slice centres: pooled centroid, but offset the L/R axis to the lh centroid
    # so the parasagittal cut passes through a hemisphere, not the midline.
    lh_c, rh_c = vox[:n_lh].mean(0), vox[n_lh:].mean(0)
    center = np.round(vox.mean(0)).astype(int)
    lr_axis = int(np.abs(lh_c - rh_c).argmax())
    center[lr_axis] = int(round(lh_c[lr_axis]))
    center = np.clip(center, 0, np.array(gt_wm.shape) - 1)

    axis_names = ['axial', 'coronal', 'sagittal']
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for a in range(3):
        ax = axes[a]
        s = int(center[a])
        ax.imshow(np.take(gt_wm, s, axis=a).T, cmap='gray', origin='lower', aspect='equal')
        # GT + SynthSeg WM marching-cubes surfaces (same smooth extractor), plain contours.
        for (sv, sf), col in [(gt_surf, GT_SURF_C), (ss_surf, SS_SURF_C)]:
            cseg = surface_contour(sv, sf, a, s)
            if len(cseg):
                ax.add_collection(LineCollection(cseg, colors=col, linewidths=1.0, alpha=0.9))
        # Deformed genus-0 template mesh, solid contour.
        seg, _ = slice_segments(vox, faces, dist_mm, a, s)
        ax.add_collection(LineCollection(seg, colors=DEFORM_C, linewidths=1.6))
        ax.set_title(f'{axis_names[a]} @ {s}')
        ax.axis('off')
        if a == 0:
            ax.legend(handles=[
                Line2D([0], [0], color=GT_SURF_C, lw=2, label='GT white surface'),
                Line2D([0], [0], color=SS_SURF_C, lw=2, label='SynthSeg WM (MC)'),
                Line2D([0], [0], color=DEFORM_C, lw=2, label='deformed mesh'),
            ], loc='lower right', fontsize=7, framealpha=0.6)

    axh = axes[3]
    axh.hist(dist_mm, bins=60, color='steelblue')
    for v, c, lab in [(float(dist_mm.mean()), 'red', 'mean'),
                      (float(np.median(dist_mm)), 'orange', 'median'),
                      (float(np.percentile(dist_mm, 95)), 'purple', 'hd95')]:
        axh.axvline(v, color=c, ls='--', lw=1.2, label=f'{lab} {v:.2f} mm')
    axh.set_xlabel('per-vertex distance to GT white surface (mm)')
    axh.set_ylabel('vertices')
    axh.set_title('distance distribution')
    axh.legend(fontsize=8)

    fig.suptitle(
        f'{name} — genus-0 deformed mesh vs GT white surface   '
        f'(mean {dist_mm.mean():.2f} / median {np.median(dist_mm):.2f} / '
        f'hd95 {np.percentile(dist_mm, 95):.2f} mm)   '
        f'[input per config; supervised-vs {supervision_target}; '
        f'green=GT white, orange=SynthSeg MC, blue=deformed mesh]',
        fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize N random samples for the WM-mesh FFD model")
    ap.add_argument('--checkpoint', required=True, help='model .pth (e.g. .../checkpoints/best_model.pth)')
    ap.add_argument('--config', default=None, help='config.yaml (default: ./config.yaml)')
    ap.add_argument('--output_dir', default='images_mesh', help='where PNGs + summary.json land')
    ap.add_argument('--num_samples', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--val_txt', default=None, help='override config data.val_txt')
    ap.add_argument('--use_affine', action='store_true',
                    help='set ONLY if the checkpoint was trained with affine enabled')
    ap.add_argument('--smooth', type=float, default=1.0,
                    help='Gaussian sigma (vox) for the SynthSeg WM marching-cubes extractor')
    ap.add_argument('--white_lh', default='lh.white',
                    help="subject GT lh white-surface basename under /meshes/<subj>/ "
                         "(pass e.g. lh.white.surf if that is how yours are named)")
    ap.add_argument('--white_rh', default='rh.white', help='subject GT rh white-surface basename')
    args = ap.parse_args()

    print(f"[viz] checkpoint={args.checkpoint}\n[viz] device={args.device} — "
          f"loading model + template (a busy GPU will be slow; --device cpu is fine here)...",
          flush=True)
    ctx = setup_inference(args.checkpoint, args.config, args.device,
                          use_affine=args.use_affine, verbose=True)
    cfg, device, target_size = ctx['cfg'], ctx['device'], ctx['target_size']
    n = target_size[0]
    exp = cfg.get('experiment', {})
    input_source = exp.get('input_source', 'synthseg')
    supervision_target = exp.get('supervision_target', 'gt')

    verts_np, faces_np, n_lh = load_template_wm_mesh(cfg['data']['template_wm_mesh_path'])
    template_verts = torch.tensor(verts_np, dtype=torch.float32, device=device)
    print(f"Mesh: {len(verts_np):,} verts ({n_lh:,} lh) | {len(faces_np):,} faces")

    d = cfg['data']
    ds = SegDataset(
        args.val_txt or d['val_txt'], d['template_seg_path'], target_size=target_size,
        seg_filename=d['seg_filename'], synthseg_filename=d['synthseg_filename'],
        preload=False)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(len(ds)), min(args.num_samples, len(ds))))
    print(f"Model input: {input_source} | supervised-vs: {supervision_target} | colour-vs: GT | "
          f"samples: {idxs}", flush=True)

    results = []
    for k, i in enumerate(idxs, 1):
        sample = ds[i]
        subject_dir = ds.subject_dirs[i]
        name = Path(subject_dir).name
        print(f"[viz] ({k}/{len(idxs)}) {name} — deform + render...", flush=True)
        synthseg_seg = sample['synthseg_seg'].unsqueeze(0).to(device)
        gt_seg = sample['gt_seg'].unsqueeze(0).to(device)
        input_seg, _ = build_step_inputs(
            synthseg_seg, gt_seg,
            input_source=input_source, supervision_target=supervision_target)

        deformed = deform_template_mesh(ctx, template_verts, input_seg)   # (N,3) subject-norm

        # GT reference: the subject's own white surface, in the same normalized
        # frame as the pushed mesh.
        ref_nii = os.path.join(subject_dir, 'seg4.nii.gz')
        gt_verts, gt_faces = load_subject_white_norm(
            subject_dir, ref_nii, args.white_lh, args.white_rh)
        if gt_verts is None:
            print(f"  [skip] no {args.white_lh}/{args.white_rh} under "
                  f"{subject_dir.replace('/scans/', '/meshes/')}", flush=True)
            continue

        # Per-vertex surface distance: each deformed vertex → nearest GT-white vertex (mm).
        dist_mm = cKDTree(gt_verts).query(deformed)[0] * MM_PER_NORM

        vox = norm_to_vox(deformed, n)
        gt_surf = (norm_to_vox(gt_verts, n), gt_faces)           # GT white surface (green)
        ss_lbl = sample['synthseg_seg'].argmax(0).numpy()
        gt_wm = (sample['gt_seg'].argmax(0).numpy() == WM_LABEL).astype(np.float32)  # gray context
        ss_surf = wm_surface(ss_lbl, smooth=args.smooth)         # SynthSeg WM marching-cubes (orange)

        png = out / f'mesh_{name}.png'
        render_sample(name, vox, faces_np, dist_mm, gt_wm, gt_surf, ss_surf, n_lh, png,
                      supervision_target)
        r = {'subject': name, 'mean_mm': float(dist_mm.mean()),
             'median_mm': float(np.median(dist_mm)),
             'hd95_mm': float(np.percentile(dist_mm, 95)),
             'max_mm': float(dist_mm.max())}
        results.append(r)
        print(f"  {name}: mean {r['mean_mm']:.3f}  median {r['median_mm']:.3f}  "
              f"hd95 {r['hd95_mm']:.3f}  max {r['max_mm']:.3f}  mm  -> {png}", flush=True)

    if results:
        means = np.array([r['mean_mm'] for r in results])
        print(f"\nN={len(results)}  mean-of-means {means.mean():.3f} mm  "
              f"(should track the logged val GT surface dist)")
        (out / 'summary.json').write_text(json.dumps(
            {'checkpoint': args.checkpoint, 'input_source': input_source,
             'supervision_target': supervision_target, 'seed': args.seed,
             'per_sample': results}, indent=2))
        print(f"Summary -> {out / 'summary.json'}")


if __name__ == '__main__':
    main()
