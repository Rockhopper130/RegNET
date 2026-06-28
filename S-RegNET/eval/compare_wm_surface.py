"""
WM-surface soundness check (inversion-free).

Registers the template seg onto a sample's SynthSeg seg with the model's OWN
forward pass (SpatialTransformer — no field inversion), then extracts the
WHITE-MATTER isosurface via marching cubes from THREE 128^3 label volumes on
the SAME grid:

    generated (cyan)  — the warped template (model output)
    synthseg  (yellow)— the SynthSeg seg the model registered TO (its input)
    actual    (red)   — the ground-truth seg4_onehot

Because all three come from 128^3 one-hot volumes on the same grid and the
SAME marching-cubes extractor (`wm_surface`, identical smoothing), the three
pairwise mean/95th-pct surface distances are a direct, like-for-like read of
the compare_synthseg_vs_gt triangle in surface-distance space:

    generated ↔ synthseg : how well the warp fits its registration target
    synthseg  ↔ actual   : SynthSeg's own quality vs GT (a model-free floor)
    generated ↔ actual   : the model's true accuracy vs GT

A flat contour overlay can't resolve the ~0.1-voxel gaps these numbers show,
and it lets *smoothness* masquerade as accuracy: the warped template is a
smooth deformation of a clean template whose boundary runs parallel to GT,
while SynthSeg carries genuine local shape deviation — so at these sub-voxel
separations the eye can't rank three overlaid contours by closeness, and the
smoother one reads as "better". (Note both surfaces are island-cleaned by
`wm_surface` below, so this is NOT a stray-component artefact.) The figure this
writes therefore (a) colours each moving surface by its per-vertex distance to
GT — so you see WHERE it deviates, not just how smooth it is — and (b) plots
the per-vertex distance distributions, the honest arbiter of which surface
sits closer to GT.

It deliberately does NOT chase gyral correspondence — the segs are 128^3
(blocky, no fine folds), which a volumetric seg-flow neither targets nor can
deliver. Cross-check: the generated↔actual mean distance is the mesh version
of compare_synthseg_vs_gt's model_ahd_per_class[2] (WM); they should agree.

Usage:
    python compare_wm_surface.py \\
        --synthseg <.../orig_synthseg.nii.gz> --gt_onehot <.../seg4_onehot.npy> \\
        --checkpoint <best_model.pth> --output wm_compare.png --device cuda:0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import setup_inference, load_seg_input
from overlay_surface import DEFAULT_CHECKPOINT

WM_LABEL = 3  # 5-class: 0 bg, 1 cortex, 2 subcortical GM, 3 white matter, 4 CSF
HEAT_CMAP = 'turbo'  # blue (close) → red (far): "where does this surface deviate"

# Colours for the three surfaces in figure titles/legends.
GEN_C, SS_C, GT_C = 'cyan', 'gold', 'red'


def wm_surface(label_vol, smooth=0.7):
    """Marching-cubes WM isosurface from a 128^3 argmax label volume.

    The mesher (skimage default = Lewiner) is already watertight/manifold; the
    defects that matter come from the hard-thresholded mask. We clean the MASK
    identically for every surface: keep the largest connected component (stray
    islands inflate Hausdorff), fill cavities, then Gaussian-smooth before the
    0.5 crossing to de-staircase. Identical cleanup means a uniform sub-voxel
    shift cancels in the comparison — but it does NOT fully cancel when the
    inputs differ topologically (e.g. if one seg's L/R WM is disconnected,
    keep-largest drops a hemisphere for that surface only and its distance
    blows up; watch the batch scatter for such outliers). This is NOT
    genus-correction (use FreeSurfer/TopoFit for genus-0 surfaces) — it's the
    proportionate cleanup a geometric surface-distance metric needs.
    """
    mask = (label_vol == WM_LABEL)
    lbl, n = ndimage.label(mask)
    if n > 1:                                          # keep only the largest blob
        sizes = ndimage.sum(mask, lbl, range(1, n + 1))
        mask = lbl == (int(np.argmax(sizes)) + 1)
    mask = ndimage.binary_fill_holes(mask).astype(np.float32)
    if smooth:
        mask = ndimage.gaussian_filter(mask, smooth)
    verts, faces, _, _ = marching_cubes(mask, level=0.5)
    return verts, faces


def _symmetric(va, vb):
    """Symmetric per-vertex surface distances (voxels): the pooled a→b and b→a
    nearest-neighbour distances, plus (mean, 95th-pct Hausdorff)."""
    da = cKDTree(vb).query(va)[0]
    db = cKDTree(va).query(vb)[0]
    both = np.concatenate([da, db])
    return both, float(both.mean()), float(np.percentile(both, 95))


# =============================================================================
# Warp + analysis (importable by the batch runner)
# =============================================================================

@torch.no_grad()
def warp_to_synthseg(ctx, synthseg_path, gt_path):
    """Register template → SynthSeg and return the three 128^3 argmax label
    volumes (generated, synthseg, actual) as numpy arrays."""
    model, template_seg, stn = ctx['model'], ctx['template_seg'], ctx['stn']
    target_size, num_classes, device = ctx['target_size'], ctx['num_classes'], ctx['device']

    synthseg_seg, _ = load_seg_input(synthseg_path, target_size, num_classes)
    synthseg_seg = synthseg_seg.unsqueeze(0).to(device)
    gt_seg, _ = load_seg_input(gt_path, target_size, num_classes)
    gt_seg = gt_seg.unsqueeze(0).to(device)

    cps_list, affine_matrix = model(template_seg, synthseg_seg)
    assert affine_matrix is None, "checkpoint uses affine; warp replay not handled here"
    # Volume warp + marching cubes here; the genus-0 mesh path lives in
    # utils/visualize_mesh_samples.py.
    flow = model.dense_flow_from_cps(cps_list)
    warped_seg = stn(template_seg, flow)

    gen_lbl = warped_seg.squeeze(0).argmax(0).cpu().numpy()
    ss_lbl = synthseg_seg.squeeze(0).argmax(0).cpu().numpy()
    gt_lbl = gt_seg.squeeze(0).argmax(0).cpu().numpy()
    return gen_lbl, ss_lbl, gt_lbl


def analyze_wm(gen_lbl, ss_lbl, gt_lbl, smooth):
    """Extract the three WM surfaces (same extractor + smoothing) and all
    pairwise distances. Returns a dict consumed by render_compare and the
    batch runner."""
    vg, fg = wm_surface(gen_lbl, smooth)
    vs, fs = wm_surface(ss_lbl, smooth)
    va, fa = wm_surface(gt_lbl, smooth)

    # Directed per-vertex distance TO GT — colours the moving surfaces.
    d_gen = cKDTree(va).query(vg)[0]
    d_ss = cKDTree(va).query(vs)[0]

    pairs, dists = {}, {}
    for key, a, b in [('gen_gt', vg, va), ('ss_gt', vs, va), ('gen_ss', vg, vs)]:
        both, mean_d, hd95 = _symmetric(a, b)
        pairs[key] = (mean_d, hd95)
        dists[key] = both

    return {
        'surf': {'gen': (vg, fg), 'ss': (vs, fs), 'gt': (va, fa)},
        'd_gen': d_gen, 'd_ss': d_ss,   # directed-to-GT, for the heatmap
        'pairs': pairs, 'dists': dists,
    }


# =============================================================================
# Figure
# =============================================================================

def _slice_segments(vox, faces, scal, axis, s):
    """Intersect the triangle mesh with the plane {axis == s} and carry a
    per-vertex scalar `scal` onto each segment (mean of its two endpoints).

    Returns (segments (K,2,2) in-plane voxel coords, seg_scalar (K,)). Same
    triangle-edge interpolation as overlay_surface.slice_segments, extended to
    interpolate the scalar at each plane crossing with the same parameter t.
    """
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

    P = np.stack(pts, axis=1)                          # (M,3,2)
    S = np.stack(svals, axis=1)                        # (M,3)
    Mk = np.stack(masks, axis=1)                       # (M,3)
    sel = Mk.sum(1) == 2
    seg = P[sel][Mk[sel]].reshape(-1, 2, 2)
    sval = S[sel][Mk[sel]].reshape(-1, 2).mean(1)
    return seg, sval


def render_compare(analysis, gt_lbl, output, title, vox_mm):
    """3×3 figure: rows 0–1 = generated / synthseg WM contours coloured by
    per-vertex distance to GT (shared scale) over the GT volume, 3 ortho views;
    row 2 = distance histograms, a mean/hd95 bar chart, and a text summary."""
    (vg, fg), (vs, fs), (va, _) = (analysis['surf']['gen'],
                                   analysis['surf']['ss'],
                                   analysis['surf']['gt'])
    pairs, dists = analysis['pairs'], analysis['dists']
    d_gen, d_ss = analysis['d_gen'] * vox_mm, analysis['d_ss'] * vox_mm   # mm

    vmax = float(np.percentile(np.concatenate([d_gen, d_ss]), 95)) or 1.0
    norm = Normalize(0.0, vmax)
    center = np.clip(np.round(va.mean(0)).astype(int), 0, np.array(gt_lbl.shape) - 1)
    axis_names = ['axial', 'coronal', 'sagittal']

    fig, axes = plt.subplots(3, 3, figsize=(20, 19), constrained_layout=True)
    rows = [('generated', vg, fg, d_gen), ('synthseg', vs, fs, d_ss)]
    for r, (rname, verts, faces, dist_mm) in enumerate(rows):
        for a in range(3):
            ax = axes[r, a]
            s = int(center[a])
            ax.imshow(np.take(gt_lbl, s, axis=a).T, cmap='gray',
                      origin='lower', aspect='equal')
            seg, sval = _slice_segments(verts, faces, dist_mm, a, s)
            lc = LineCollection(seg, cmap=HEAT_CMAP, norm=norm, linewidths=1.4)
            lc.set_array(sval)
            ax.add_collection(lc)
            ax.axis('off')
            if r == 0:
                ax.set_title(f'{axis_names[a]} @ {s}')
            if a == 0:
                ax.set_ylabel(rname, fontsize=13, fontweight='bold')
                ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])

    fig.colorbar(ScalarMappable(norm=norm, cmap=HEAT_CMAP),
                 ax=[axes[0, 2], axes[1, 2]], shrink=0.8, pad=0.02,
                 label='WM surface distance to GT (mm)')

    # --- Row 2: histograms / bars / text -------------------------------------
    hist_max = float(np.percentile(np.concatenate(list(dists.values())), 99)) * vox_mm
    bins = np.linspace(0, max(hist_max, 1e-3), 60)
    pair_meta = [('gen_gt', 'generated ↔ actual', GEN_C),
                 ('ss_gt', 'synthseg ↔ actual', SS_C),
                 ('gen_ss', 'generated ↔ synthseg', 'dimgray')]

    ax_h = axes[2, 0]
    for key, label, color in pair_meta:
        mean_d = pairs[key][0] * vox_mm
        ax_h.hist(dists[key] * vox_mm, bins=bins, histtype='step', density=True,
                  color=color, lw=1.8, label=f'{label}  (μ={mean_d:.2f} mm)')
        ax_h.axvline(mean_d, color=color, ls='--', lw=1.0)
    ax_h.set_xlabel('per-vertex surface distance (mm)')
    ax_h.set_ylabel('density')
    ax_h.set_title('Distance distributions')
    ax_h.legend(fontsize=8)

    ax_b = axes[2, 1]
    x = np.arange(3)
    means = [pairs[k][0] * vox_mm for k, _, _ in pair_meta]
    hd95s = [pairs[k][1] * vox_mm for k, _, _ in pair_meta]
    ax_b.bar(x - 0.19, means, 0.38, label='mean', color='steelblue')
    ax_b.bar(x + 0.19, hd95s, 0.38, label='hd95', color='lightsteelblue')
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(['gen↔gt', 'ss↔gt', 'gen↔ss'])
    ax_b.set_ylabel('mm')
    ax_b.set_title('Surface distance (mm)')
    ax_b.legend(fontsize=8)

    ax_t = axes[2, 2]
    ax_t.axis('off')
    lines = [f"vertices  gen {len(vg):,}  ss {len(vs):,}  gt {len(va):,}", ""]
    for key, label, _ in pair_meta:
        m, h = pairs[key]
        lines.append(f"{label:22s}  mean {m * vox_mm:5.2f} mm   hd95 {h * vox_mm:5.2f} mm")
    lines += ["", "the smooth warp can look closer than it",
              "measures — the histogram is the arbiter.",
              "warp usually tracks above ss↔actual (no GT",
              "signal), but not always — see summary.png."]
    ax_t.text(0.0, 1.0, "\n".join(lines), va='top', ha='left',
              family='monospace', fontsize=10, transform=ax_t.transAxes)

    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {output}")


# =============================================================================
# Single-sample CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="WM-surface soundness check (no inversion)")
    ap.add_argument('--synthseg', required=True,
                    help='sample SynthSeg seg (orig_synthseg.nii.gz) — the model input')
    ap.add_argument('--gt_onehot', required=True,
                    help='ground-truth seg4_onehot.npy — the "actual" reference')
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='model .pth')
    ap.add_argument('--config', default=None, help='config.yaml (default: ./config.yaml)')
    ap.add_argument('--device', default='cuda:0', help='e.g. cuda:0, cpu')
    ap.add_argument('--output', required=True, help='output PNG path')
    ap.add_argument('--smooth', type=float, default=1.2,
                    help='Gaussian sigma (vox) to de-staircase the WM mask before '
                         'marching cubes; same for all three surfaces so it cancels. 0 = off.')
    args = ap.parse_args()

    ctx = setup_inference(args.checkpoint, args.config, args.device,
                          use_affine=False, verbose=True)
    vox_mm = 256.0 / ctx['target_size'][0]  # ~2 mm/vox: native conformed 256^3 → 128^3

    gen_lbl, ss_lbl, gt_lbl = warp_to_synthseg(ctx, args.synthseg, args.gt_onehot)
    analysis = analyze_wm(gen_lbl, ss_lbl, gt_lbl, args.smooth)

    pairs = analysis['pairs']
    print("\nWM surface distances (same 128^3 grid, same extractor):")
    print(f"    {'pair':22s} {'mean vox':>10s} {'mean mm':>9s} {'hd95 vox':>10s} {'hd95 mm':>9s}")
    for key, label in [('gen_ss', 'generated ↔ synthseg'),
                       ('ss_gt', 'synthseg  ↔ actual'),
                       ('gen_gt', 'generated ↔ actual')]:
        m, h = pairs[key]
        print(f"    {label:22s} {m:10.3f} {m * vox_mm:9.2f} {h:10.3f} {h * vox_mm:9.2f}")

    render_compare(analysis, gt_lbl, args.output,
                   f"{Path(args.gt_onehot).parent.name} — WM surface vs GT: "
                   f"generated / synthseg",
                   vox_mm)


if __name__ == '__main__':
    main()
