"""
Overlay FreeSurfer surface(s) as contour outlines on orthogonal volume slices.

Two modes:

  IDENTITY (--surf):  render a surface (lh.white / rh.white) on the volume it was
    reconstructed against — the TopoFit/FreeSurfer-style WM contour over a slice.
    The alignment check that must pass before any flow is applied to a mesh.

  WARP (--template_surf): propagate the TEMPLATE surface onto a SAMPLE through the
    model's deformation field and overlay it against the sample's own surface
    (ground truth). Because the flow is a backward field (sample→template), this
    requires inverting it (fixed-point); see the warp-mode section below.

The outline is a true mesh-plane intersection: for every triangle that straddles
the cut plane we emit the segment where it crosses, so the result is a clean
continuous contour rather than a rasterized band of vertices.

Coordinate chain (the only thing that can silently go wrong):

    surface coords are in FreeSurfer surface-RAS (a.k.a. tkrRAS), millimetres.
    surface-RAS + cras (stored in the surface metadata) = scanner RAS (world mm).
    voxel = inv(vol.affine) @ [scanner RAS, 1].

Because we route through *scanner RAS world coords*, the background volume does
not have to be the exact conformed volume the surface was built from — any
volume sharing the same world space works (e.g. orig_synthseg.nii.gz, which
SynthSeg writes in the input image's geometry). The diagnostic block prints the
surface vs. volume bounding boxes so misalignment (wrong volume, wrong space,
or an axis-order bug) is obvious at a glance.

Usage:
    # Identity overlay
    python overlay_surface.py \\
        --surf  ~/shared_scratch/oasis_data/meshes/OASIS_OAS1_0406_MR1/lh.white \\
                ~/shared_scratch/oasis_data/meshes/OASIS_OAS1_0406_MR1/rh.white \\
        --vol   .../OASIS_OAS1_0406_MR1/seg4.nii.gz \\
        --output overlay.png

    # Warp template surface onto a sample, compare to the sample's own surface
    python overlay_surface.py \\
        --template_surf ~/shared_scratch/oasis_data/meshes/OASIS_OAS1_0406_MR1/lh.white \\
                        ~/shared_scratch/oasis_data/meshes/OASIS_OAS1_0406_MR1/rh.white \\
        --sample_surf   ~/shared_scratch/oasis_data/meshes/OASIS_OAS1_0021_MR1/lh.white \\
                        ~/shared_scratch/oasis_data/meshes/OASIS_OAS1_0021_MR1/rh.white \\
        --sample_onehot .../scans/OASIS_OAS1_0021_MR1/seg4_onehot.npy \\
        --device cuda:0 --output warp_0021.png
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


SURF_COLORS = ['cyan', 'yellow', 'magenta', 'lime']


def load_surface_voxels(surf_path, vol_affine):
    """Read a FreeSurfer surface; return vertices in volume voxel coords + faces.

    surface-RAS (tkrRAS) + cras -> scanner RAS (world mm) -> voxel via inv(affine).
    """
    coords, faces, meta = nib.freesurfer.read_geometry(surf_path, read_metadata=True)
    cras = meta.get('cras', np.zeros(3))           # surface->scanner RAS offset
    world = coords + cras                            # scanner RAS, mm
    vox = apply_affine(np.linalg.inv(vol_affine), world)
    return vox, faces, world


def slice_segments(vox, faces, axis, s):
    """Intersect the triangle mesh with the plane {axis == s}.

    Returns an (K, 2, 2) array of line segments in the in-plane (p, q) voxel
    coords (the two axes other than `axis`), matching the imshow x/y convention.
    A triangle crossing the plane has exactly two edges that straddle it; each
    such triangle contributes one segment joining the two edge-crossing points.
    """
    p, q = [ax for ax in range(3) if ax != axis]
    d = vox[:, axis] - s                            # signed distance to plane
    xy = vox[:, [p, q]]                              # in-plane coords per vertex

    df = d[faces]                                    # (M, 3) at triangle verts
    pf = xy[faces]                                   # (M, 3, 2)

    pts, masks = [], []
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        di, dj = df[:, i], df[:, j]
        cross = (di > 0) != (dj > 0)                 # vertices on opposite sides
        denom = di - dj
        safe = np.abs(denom) > 1e-12                 # triangles parallel to the plane → denom 0
        t = np.where(safe, di / np.where(safe, denom, 1.0), 0.5)
        pts.append(pf[:, i, :] + t[:, None] * (pf[:, j, :] - pf[:, i, :]))
        masks.append(cross)

    P = np.stack(pts, axis=1)                        # (M, 3, 2)
    Mk = np.stack(masks, axis=1)                     # (M, 3)
    sel = Mk.sum(1) == 2                             # triangles cut by the plane
    return P[sel][Mk[sel]].reshape(-1, 2, 2)


def _print_diagnostics(name, world, vox, vol_shape):
    """Bounding boxes in world mm and in voxel index space — the alignment check."""
    print(f"\n  [{name}]")
    print(f"    vertices            : {len(world):,}")
    print(f"    world RAS bbox (mm) : "
          f"x[{world[:,0].min():7.1f},{world[:,0].max():7.1f}] "
          f"y[{world[:,1].min():7.1f},{world[:,1].max():7.1f}] "
          f"z[{world[:,2].min():7.1f},{world[:,2].max():7.1f}]")
    print(f"    voxel bbox          : "
          f"i[{vox[:,0].min():6.1f},{vox[:,0].max():6.1f}] "
          f"j[{vox[:,1].min():6.1f},{vox[:,1].max():6.1f}] "
          f"k[{vox[:,2].min():6.1f},{vox[:,2].max():6.1f}]")
    inside = ((vox >= 0) & (vox < np.array(vol_shape))).all(axis=1).mean() * 100
    print(f"    inside volume       : {inside:5.1f}%   "
          f"(volume voxel grid is {tuple(vol_shape)})")
    if inside < 90:
        print("    *** WARNING: many vertices fall OUTSIDE the volume — surface and "
              "volume are probably not in the same space. Overlay will be wrong. ***")


def load_volume(vol_path):
    """Load a volume as a 3D float array (one-hot/multi-frame collapsed to labels)."""
    vol = nib.load(vol_path)
    data = np.asanyarray(vol.dataobj).astype(np.float32)
    if data.ndim == 4:                               # one-hot / multi-frame -> collapse
        data = data.argmax(-1).astype(np.float32) if data.shape[-1] <= 8 else data[..., 0]
    return vol, data


def render(data, surfaces, output, title):
    """Draw mesh-plane contours for each surface on three orthogonal slices.

    surfaces: list of (name, vox, faces, color); vox is (N,3) in `data` voxel coords.
    """
    # Slice positions: pooled centroid per axis, EXCEPT the left-right axis (the
    # one separating the first two surface centroids most), offset to one
    # surface's centroid so the parasagittal cut shows gyri instead of midline.
    centroids = np.array([v.mean(0) for _, v, _, _ in surfaces])
    center = np.round(centroids.mean(0)).astype(int)
    if len(surfaces) >= 2:
        lr_axis = int(np.abs(centroids[0] - centroids[1]).argmax())
        center[lr_axis] = int(round(centroids[0, lr_axis]))
    center = np.clip(center, 0, np.array(data.shape) - 1)

    axis_names = ['Axis-0 slice', 'Axis-1 slice', 'Axis-2 slice']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for a in range(3):
        s = int(center[a])
        img = np.take(data, s, axis=a)               # in-plane axes are p<q
        axes[a].imshow(img.T, cmap='gray', origin='lower', aspect='equal')
        for name, vox, faces, color in surfaces:
            seg = slice_segments(vox, faces, a, s)   # true mesh-plane contour
            axes[a].add_collection(LineCollection(seg, colors=color, linewidths=0.6))
        axes[a].set_title(f'{axis_names[a]} @ {s}')
        axes[a].axis('off')
        if a == 0:
            axes[a].legend(handles=[
                Line2D([0], [0], color=color, lw=2, label=name)
                for name, _, _, color in surfaces
            ], loc='lower right', fontsize=8)

    fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {output}")


def overlay(surf_paths, vol_path, output):
    vol, data = load_volume(vol_path)
    print("=" * 70)
    print("Surface-on-slice overlay")
    print("=" * 70)
    print(f"Volume : {vol_path}\n         shape {data.shape}, zooms {vol.header.get_zooms()[:3]}")

    surfaces = []
    for ci, sp in enumerate(surf_paths):
        vox, faces, world = load_surface_voxels(sp, vol.affine)
        _print_diagnostics(Path(sp).name, world, vox, data.shape)
        surfaces.append((Path(sp).name, vox, faces, SURF_COLORS[ci % len(SURF_COLORS)]))

    render(data, surfaces, output, f'{Path(vol_path).parent.name} — surface overlay')


# =============================================================================
# Warp mode: propagate the TEMPLATE surface onto a SAMPLE via the model's flow,
# then overlay it against the sample's own (ground-truth) surface.
#
# The model's flow is a backward field: sample[o] ≈ template[o + flow(o)], so
# sample-space → template-space is o ↦ o + flow(o). Placing a template-space
# vertex v in sample space means solving o + flow(o) = v — i.e. inverting the
# field — via the fixed point  o ← v − flow(o).
#
# --forward flips this: it pushes the SAMPLE surface into template space with the
# forward map o ↦ o + flow(o) directly (one evaluation, no inversion) and compares
# it against the template surface — the clean, inversion-free direction. If that
# overlay is smooth the flow is good and the jaggedness was purely the inversion.
# =============================================================================

# Default checkpoint — the one used by temp_run_inference_all.py.
DEFAULT_CHECKPOINT = ("/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/"
                      "training_seg_acm/20260528_115247/checkpoints/best_model.pth")


def _ref_for(onehot_path):
    """seg4_onehot.npy -> seg4.nii.gz in the same directory (for affine + dims)."""
    return str(onehot_path).replace("seg4_onehot.npy", "seg4.nii.gz")


def world_to_norm(world, ref):
    """World mm -> normalized [-1,1] grid coords (x,y,z) of `ref`'s voxel grid.

    Axis map matches the model: x<->k (axis 2), y<->j (axis 1), z<->i (axis 0),
    since one-hot (5,Ni,Nj,Nk) -> (5,D,H,W) and grid_sample's (x,y,z) -> (W,H,D).
    Normalization uses `ref`'s NATIVE dims — the [-1,1] FOV is resolution-
    independent, so the 128³ flow can be sampled at the same coordinate.
    """
    Ni, Nj, Nk = ref.shape[:3]
    ijk = apply_affine(np.linalg.inv(ref.affine), world)      # native voxel (i,j,k)
    x = (2 * ijk[:, 2] + 1) / Nk - 1
    y = (2 * ijk[:, 1] + 1) / Nj - 1
    z = (2 * ijk[:, 0] + 1) / Ni - 1
    return np.stack([x, y, z], axis=1)


def norm_to_world(norm, ref):
    """Inverse of world_to_norm: normalized (x,y,z) -> world mm via `ref`."""
    Ni, Nj, Nk = ref.shape[:3]
    i = ((norm[:, 2] + 1) * Ni - 1) / 2
    j = ((norm[:, 1] + 1) * Nj - 1) / 2
    k = ((norm[:, 0] + 1) * Nk - 1) / 2
    return apply_affine(ref.affine, np.stack([i, j, k], axis=1))


def _sample_flow(flow, pts):
    """Sample (1,3,D,H,W) flow at pts (torch (N,3) normalized x,y,z) -> (N,3).

    align_corners=False matches the model's SpatialTransformer; a mismatch here
    introduces a sub-voxel drift.
    """
    import torch.nn.functional as F
    g = pts.view(1, -1, 1, 1, 3)
    s = F.grid_sample(flow, g, mode='bilinear', padding_mode='border',
                      align_corners=False)                    # (1,3,N,1,1)
    return s.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0)  # (N,3), order (x,y,z)


def invert_to_sample(flow, v_norm, n_iter, alpha=0.5):
    """Wrapper around model.invert_to_sample (the one definition of the inverse),
    so this tool and the training/eval mesh-push share it. Returns o (N,3
    normalized) and per-vertex residual ||o+flow(o)-v||."""
    from model import invert_to_sample as _invert
    return _invert(flow, v_norm, n_iter, alpha)


def warp_overlay(args):
    import torch
    from inference import setup_inference, load_seg_input

    ctx = setup_inference(args.checkpoint, args.config, args.device,
                          use_affine=False, verbose=True)
    cfg, model, template_seg = ctx['cfg'], ctx['model'], ctx['template_seg']
    target_size, num_classes, device = ctx['target_size'], ctx['num_classes'], ctx['device']

    template_ref = args.template_ref or _ref_for(cfg['data']['template_seg_path'])
    sample_ref = args.sample_ref or _ref_for(args.sample_onehot)
    vol_path = args.vol or sample_ref
    tref, sref = nib.load(template_ref), nib.load(sample_ref)

    # Shape asserts (Check #3) — a mismatched ref silently corrupts every coord.
    assert tuple(tref.shape[:3]) == tuple(
        np.load(cfg['data']['template_seg_path'], mmap_mode='r').shape[1:]), \
        f"template_ref {tref.shape[:3]} != template one-hot {np.load(cfg['data']['template_seg_path'], mmap_mode='r').shape[1:]}"
    assert tuple(sref.shape[:3]) == tuple(
        np.load(args.sample_onehot, mmap_mode='r').shape[1:]), \
        f"sample_ref {sref.shape[:3]} != sample one-hot {np.load(args.sample_onehot, mmap_mode='r').shape[1:]}"

    # Forward pass -> flow for THIS sample (template -> sample).
    sample_seg, _ = load_seg_input(args.sample_onehot, target_size, num_classes)
    sample_seg = sample_seg.unsqueeze(0).to(device)
    with torch.no_grad():
        cps_list, affine_matrix = model(template_seg, sample_seg)
        flow = model.dense_flow_from_cps(cps_list)
    assert affine_matrix is None, "checkpoint uses affine; warp replay not implemented here"

    f = flow.detach().float()
    mag = f.norm(dim=1)
    print(f"\nFlow (normalized units): mean |u| {mag.mean():.4f}, max |u| {mag.max():.4f}")

    # --- Forward mode: push the SAMPLE surface into TEMPLATE space (no inversion) ---
    # flow maps sample→template, so t = o + flow(o) is exact for sample vertices.
    # --template_surf is the GT here; overlay lands on the template volume.
    if args.forward:
        vol, data = load_volume(args.vol or template_ref)
        warped = []
        for ci, sp in enumerate(args.sample_surf):
            coords, faces, meta = nib.freesurfer.read_geometry(sp, read_metadata=True)
            sworld = coords + meta.get('cras', np.zeros(3))
            o_norm = world_to_norm(sworld, sref)
            o = torch.as_tensor(o_norm, dtype=torch.float32, device=f.device)
            t_norm = (o + _sample_flow(f, o)).cpu().numpy()
            world0 = norm_to_world(o_norm, tref)             # sample coords in template space, no flow
            worldN = norm_to_world(t_norm, tref)             # + flow (exact, no inversion)
            vox = apply_affine(np.linalg.inv(vol.affine), worldN)
            warped.append({'name': f'warped {Path(sp).name}', 'vox': vox, 'faces': faces,
                           'color': SURF_COLORS[ci % len(SURF_COLORS)],
                           'world0': world0, 'worldN': worldN})
        gt = []
        for sp in args.template_surf:
            vox, faces, world = load_surface_voxels(sp, vol.affine)
            gt.append({'name': f'GT {Path(sp).name}', 'vox': vox, 'faces': faces, 'world': world})
        if gt:
            tree = cKDTree(np.concatenate([g['world'] for g in gt], axis=0))
            print("\nCheck (forward) — warped sample→template GT mean nearest distance (mm), must DROP:")
            for w in warped:
                d0 = tree.query(w['world0'])[0].mean()
                dN = tree.query(w['worldN'])[0].mean()
                flag = "OK" if dN < d0 else "*** WRONG DIRECTION/AXES ***"
                print(f"    {w['name']:28s}: no-flow {d0:6.2f} -> warped {dN:6.2f}  {flag}")
        render_surfaces = [(w['name'], w['vox'], w['faces'], w['color']) for w in warped]
        render_surfaces += [(g['name'], g['vox'], g['faces'], 'red') for g in gt]
        render(data, render_surfaces, args.output,
               f"{Path(template_ref).parent.name} — forward-warped sample (color) vs GT template (red)")
        return

    vol, data = load_volume(vol_path)

    # --- Warp each template surface into sample space (field inversion) ---
    tol = 2.0 / 128                                          # ≈2-voxel inverse-converged tolerance
    warped = []
    for ci, sp in enumerate(args.template_surf):
        coords, faces, meta = nib.freesurfer.read_geometry(sp, read_metadata=True)
        tworld = coords + meta.get('cras', np.zeros(3))
        v_norm = world_to_norm(tworld, tref)
        o0, _ = invert_to_sample(f, v_norm, 0)               # iter0 = raw template coords
        oN, res = invert_to_sample(f, v_norm, args.n_iter, args.alpha)
        res = res.cpu().numpy()
        world0 = norm_to_world(o0.cpu().numpy(), sref)
        worldN = norm_to_world(oN.cpu().numpy(), sref)
        vox = apply_affine(np.linalg.inv(vol.affine), worldN)
        good = res < tol                                     # vertices the inverse actually reached
        faces_ok = faces[good[faces].all(axis=1)]            # drop faces touching a non-invertible vertex
        warped.append({'name': f'warped {Path(sp).name}', 'vox': vox, 'faces': faces_ok,
                       'color': SURF_COLORS[ci % len(SURF_COLORS)],
                       'world0': world0, 'worldN': worldN,
                       'res_mean': float(res.mean()), 'res_max': float(res.max()),
                       'conv_pct': float(good.mean() * 100),
                       'n_drop': int(len(faces) - len(faces_ok))})

    # --- Ground-truth sample surfaces (the already-validated overlay path) ---
    gt = []
    for sp in args.sample_surf:
        vox, faces, world = load_surface_voxels(sp, vol.affine)
        gt.append({'name': f'GT {Path(sp).name}', 'vox': vox, 'faces': faces, 'world': world})

    # Check #1 — the inverse must CONVERGE (residual → 0); distance dropping is
    # necessary but NOT sufficient (a scattered cloud on the dense GT surface
    # scores ~1.5 mm regardless). Gate on conv% so a non-converged warp can't pass.
    if gt:
        tree = cKDTree(np.concatenate([g['world'] for g in gt], axis=0))
        print(f"\nCheck #1 — warped→GT mean nearest distance (mm) + inverse convergence "
              f"(residual ≪ {tol:.4f}):")
        for w in warped:
            d0 = tree.query(w['world0'])[0].mean()
            dN = tree.query(w['worldN'])[0].mean()
            ok = dN < d0 and w['conv_pct'] > 99.0
            flag = "OK" if ok else "*** inverse did NOT converge — folding/oscillation, see conv% ***"
            print(f"    {w['name']:28s}: iter0 {d0:6.2f} -> iter{args.n_iter} {dN:6.2f}   "
                  f"[res mean {w['res_mean']:.4f} max {w['res_max']:.4f} | "
                  f"{w['conv_pct']:5.1f}% conv, {w['n_drop']} faces dropped]  {flag}")

    render_surfaces = [(w['name'], w['vox'], w['faces'], w['color']) for w in warped]
    render_surfaces += [(g['name'], g['vox'], g['faces'], 'white') for g in gt]
    render(data, render_surfaces, args.output,
           f"{Path(vol_path).parent.name} — warped template (color) vs GT sample (red)")


def main():
    ap = argparse.ArgumentParser(
        description='Overlay FreeSurfer surfaces on volume slices (identity or model-warped)')
    # Identity overlay
    ap.add_argument('--surf', nargs='+',
                    help='[identity mode] surfaces to overlay on --vol (e.g. lh.white rh.white)')
    ap.add_argument('--vol', help='Background volume (.nii.gz/.mgz) in scanner-RAS space')
    ap.add_argument('--output', required=True, help='Output PNG path')
    # Warp overlay
    ap.add_argument('--template_surf', nargs='+',
                    help='[warp mode] TEMPLATE surfaces to warp onto the sample')
    ap.add_argument('--sample_surf', nargs='+',
                    help='[warp mode] ground-truth SAMPLE surfaces to compare against')
    ap.add_argument('--sample_onehot',
                    help='[warp mode] sample seg4_onehot.npy (the model input)')
    ap.add_argument('--template_ref',
                    help='[warp mode] template seg4.nii.gz (affine+dims); '
                         'default: derived from config template_seg_path')
    ap.add_argument('--sample_ref',
                    help='[warp mode] sample seg4.nii.gz (affine+dims); '
                         'default: derived from --sample_onehot')
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='[warp mode] model .pth')
    ap.add_argument('--config', default=None, help='[warp mode] config.yaml (default: ./config.yaml)')
    ap.add_argument('--device', default='cuda:0', help='[warp mode] e.g. cuda:0, cpu')
    ap.add_argument('--n_iter', type=int, default=500, help='[warp mode] inversion iterations')
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='[warp mode] inversion damping in o ← o + alpha*(v-flow(o)-o); '
                         '1.0 = old plain iteration (oscillates where ‖∇flow‖≥1), <1 damps it')
    ap.add_argument('--forward', action='store_true',
                    help='[warp mode] flip direction: push the SAMPLE surface into template '
                         'space (t=s+flow(s), no inversion) and compare to --template_surf '
                         '(the GT here), overlaid on the template volume')
    args = ap.parse_args()

    if args.template_surf:
        if not (args.sample_onehot and args.sample_surf):
            ap.error("warp mode needs --template_surf, --sample_onehot, and --sample_surf")
        warp_overlay(args)
    elif args.surf:
        if not args.vol:
            ap.error("identity mode needs --vol")
        overlay(args.surf, args.vol, args.output)
    else:
        ap.error("provide --surf (identity overlay) or --template_surf (warp overlay)")


if __name__ == '__main__':
    main()
