"""
Numbers-only evaluation over EVERY subject: per-subject voxel + mesh metrics to
CSV, then means/std/min/max. No figures.

The template subject is excluded automatically (it is the moving image — warping
it onto itself is not a data point). Train and val are reported as separate
blocks as well as pooled, because train-subject numbers are not generalization
numbers and pooling them silently would flatter the model.

Per subject:
  voxel  dice_c0..c4, dice_fg_mean, dice_wm, folding_pct, min_det, cycle,
         flow_mag_mean/max, lambda_mean            (the numbers training optimizes)
  mesh   sym_mean_mm, sym_hd95_mm, mesh_to_gt_mm, gt_to_mesh_mm,
         undeformed_mean_mm, flip_pct, flip_pct_dense_only, svf_parity_max,
         inverse_residual_{mm,max_mm}, inverse_over_0p2mm_pct
         (genus-0 template mesh pushed to the subject vs its own FreeSurfer
          white surface; skipped per subject when the surfaces are missing)

The default push is exp(-v), the exact inverse of an SVF (--push svf_inv). The
inverse_* columns describe the OLD fixed-point inverse and are kept as a
reference: it stalls (residual does not shrink with --inv_iter) wherever the
field is not contractive, and its per-vertex noise creates triangle flips of its
own on a mesh with ~1 mm triangles. svf_parity_max must stay ~1e-6 — it is the
guard that our velocity replication still matches model.py.

One model forward per subject drives both halves. Mesh metrics dominate the
runtime (a 500-iteration fixed-point inverse plus two KD-tree queries over all
vertices) — pass --no_mesh for a fast voxel-only sweep. Rows are appended to the
CSV as they are computed, so a long run that dies partway keeps its results.

Run from the S-RegNET directory:
    python evaluate_all.py \\
        --model ~/shared_scratch/training_results/mahith_experiment/20260801_083414 \\
        --device cuda:4
    # --split val | train | both   (default both)
    # --limit 5                    smoke-test on the first few subjects
    # --push numeric | rv | rv_noaffine    which mesh push to score (default numeric)
    # --no_mesh                    voxel metrics only (fast)
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from scipy.spatial import cKDTree

from git_provenance import write_git_sha
from get_data import SegDataset
from inference import load_config, setup_inference
from losses import compute_dice_score, cycle_consistency_loss
from visualize_run import resolve_checkpoint, detect_affine, warp_template, folding_stats
from visualize_mesh import (WM_LABEL, load_template_mesh, load_subject_white, mm_per_norm,
                            norm_to_world, push_from_flows, ref_for, svf_inverse_flow,
                            triangle_flip_fraction)


def svf_inverse(ctx, sample_seg):
    """Exact SVF inverse field exp(-v) for this subject, plus the parity check
    that our velocity replication still matches model.py."""
    return svf_inverse_flow(ctx['model'], ctx['template_seg'], sample_seg)


def voxel_metrics(ctx, sample_seg):
    """Replay training's warp and score it. Returns (row dict, flows, affine)."""
    warped, flow_fw, flow_rv, lambda_map, affine = warp_template(ctx, sample_seg)
    dice_pc, fg = compute_dice_score(warped, sample_seg, ctx['num_classes'])
    fold, min_det = folding_stats(flow_fw)
    mag = flow_fw.pow(2).sum(1).sqrt()
    row = {f'dice_c{c}': dice_pc[c] for c in range(len(dice_pc))}
    row.update({
        'dice_fg_mean': fg,
        'dice_wm': dice_pc[WM_LABEL],
        'folding_pct': fold,
        'min_det': min_det,
        'cycle': cycle_consistency_loss(flow_fw, flow_rv, ctx['stn']).item(),
        'flow_mag_mean': mag.mean().item(),
        'flow_mag_max': mag.max().item(),
        'lambda_mean': lambda_map.mean().item(),
    })
    return row, (flow_fw, flow_rv), affine


def mesh_metrics(verts_t, verts_np, faces, flows, affine, subject_dir, seg_filename, push,
                 n_iter=500, alpha=0.5, flow_inv=None, parity=None, all_pushes=False):
    """Push the template mesh with the already-computed flows and score it against
    the subject's own white surface. Returns {} when the subject has no surfaces."""
    ref_path = ref_for(Path(subject_dir) / seg_filename)
    if not Path(ref_path).is_file():
        return {}
    ref = nib.load(ref_path)
    gt_v, _ = load_subject_white(subject_dir, ref)
    if gt_v is None:
        return {}

    p = push_from_flows(verts_t, flows[0], flows[1], affine, n_iter, alpha,
                        flow_inv=flow_inv, parity=parity)
    v = p['pushes'][push]
    mm = mm_per_norm(ref)
    res_mm = p['inv_residual_norm'] * mm

    gt_w, w = norm_to_world(gt_v, ref), norm_to_world(v, ref)
    tree = cKDTree(gt_w)
    m2g = tree.query(w)[0]
    g2m = cKDTree(w).query(gt_w)[0]
    both = np.concatenate([m2g, g2m])

    extra = {'svf_parity_max': p['svf_parity']}
    if all_pushes:                                   # diagnostic: score every mode
        for mode, mv in p['pushes'].items():
            if mode == push:
                continue
            mw = norm_to_world(mv, ref)
            mb = np.concatenate([tree.query(mw)[0], cKDTree(mw).query(gt_w)[0]])
            extra[f'{mode}_sym_mean_mm'] = float(mb.mean())
            extra[f'{mode}_flip_pct'] = triangle_flip_fraction(verts_np, mv, faces)
    return {**extra,
        'sym_mean_mm': float(both.mean()),
        'sym_hd95_mm': float(np.percentile(both, 95)),
        'mesh_to_gt_mm': float(m2g.mean()),
        'gt_to_mesh_mm': float(g2m.mean()),
        'undeformed_mean_mm': float(tree.query(norm_to_world(verts_np, ref))[0].mean()),
        # flip vs the raw template includes the affine; vs the aligned verts
        # isolates the dense inverse. A near-identity affine makes them equal —
        # if they differ, the affine itself is reversing triangles.
        'flip_pct': triangle_flip_fraction(verts_np, v, faces),
        'flip_pct_dense_only': triangle_flip_fraction(p['v_aligned'], v, faces),
        # Solver noise, not just a convergence stat: the mesh has ~1 mm triangles,
        # so residual of this size can create flips by itself.
        'inverse_residual_mm': float(res_mm.mean()),
        'inverse_residual_max_mm': float(res_mm.max()),
        'inverse_over_0p2mm_pct': float((res_mm > 0.2).mean() * 100),
    }


def summarize(rows, label):
    """mean / std / min / max for every numeric column present in rows."""
    if not rows:
        return {}
    cols = [k for k in rows[0] if k != 'idx' and isinstance(rows[0][k], (int, float))
            and not isinstance(rows[0][k], bool)]
    out = {}
    for c in cols:
        vals = np.array([r[c] for r in rows if c in r and r[c] is not None], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out[c] = {'mean': float(vals.mean()), 'std': float(vals.std()),
                      'min': float(vals.min()), 'max': float(vals.max()), 'n': int(len(vals))}
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"{'metric':<22s} {'mean':>10s} {'std':>9s} {'min':>10s} {'max':>10s} {'n':>5s}")
    for c, s in out.items():
        print(f"{c:<22s} {s['mean']:10.4f} {s['std']:9.4f} {s['min']:10.4f} "
              f"{s['max']:10.4f} {s['n']:5d}")
    return out


def main():
    ap = argparse.ArgumentParser(description='Per-subject metrics over all samples, no figures')
    ap.add_argument('--model', required=True,
                    help='run dir, a best_model.pth path, or a run name under output.base_dir')
    ap.add_argument('--config', default=None, help='config.yaml (default: ./config.yaml)')
    ap.add_argument('--output_dir', default=None, help='default <run_dir>/metrics')
    ap.add_argument('--split', choices=['val', 'train', 'both'], default='val',
                    help='default val — the generalization numbers; train/both on request')
    ap.add_argument('--limit', type=int, default=None, help='first N subjects per split (smoke test)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--affine', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--push', choices=['svf_inv', 'numeric', 'rv', 'rv_noaffine'],
                    default='svf_inv',
                    help='svf_inv = exp(-v), the exact SVF inverse (no fixed point)')
    ap.add_argument('--all_pushes', action='store_true',
                    help='also score every other push mode per subject (diagnostic; '
                         'adds a KD-tree pair per mode, use with --limit)')
    ap.add_argument('--inv_iter', type=int, default=500,
                    help='fixed-point iterations for the numeric inverse. Raise it and '
                         'compare flip_pct: if flip falls with the residual, the flips '
                         'were solver noise, not the model.')
    ap.add_argument('--inv_alpha', type=float, default=0.5, help='fixed-point damping')
    ap.add_argument('--no_mesh', action='store_true', help='voxel metrics only (much faster)')
    ap.add_argument('--template_surf', nargs=2, metavar=('LH', 'RH'), default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(args.model, cfg)
    use_affine = detect_affine(ckpt) if args.affine == 'auto' else args.affine == 'on'
    out = Path(args.output_dir).expanduser() if args.output_dir \
        else Path(ckpt).parent.parent / 'metrics'
    out.mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    print(f"[eval] checkpoint = {ckpt}")
    print(f"[eval] device = {args.device} | affine = {use_affine} | mesh = {not args.no_mesh} | "
          f"push = {args.push} | output_dir = {out}", flush=True)

    ctx = setup_inference(ckpt, args.config, args.device, use_affine=use_affine, verbose=True)
    d = cfg['data']
    template_subject = Path(d['template_seg_path']).parent.name

    verts_t = verts_np = faces = None
    if not args.no_mesh:
        verts_np, faces, n_lh = load_template_mesh(cfg, args.template_surf)
        verts_t = torch.tensor(verts_np, dtype=torch.float32, device=ctx['device'])
        print(f"[eval] template mesh {len(verts_np):,} verts ({n_lh:,} lh) | "
              f"{len(faces):,} faces", flush=True)

    splits = {'val': d['val_txt'], 'train': d['train_txt']}
    if args.split != 'both':
        splits = {args.split: splits[args.split]}

    csv_path = out / 'metrics_per_subject.csv'
    rows, writer, fh, seen, skipped = [], None, None, set(), 0
    try:
        for split, txt in splits.items():
            ds = SegDataset(txt, d['template_seg_path'], target_size=ctx['target_size'],
                            seg_filename=d['seg_filename'], preload=False)
            n = len(ds) if args.limit is None else min(args.limit, len(ds))
            print(f"\n[eval] split={split}: {len(ds)} subjects, evaluating {n}", flush=True)

            for i in range(n):
                subject_dir = ds.subject_dirs[i]
                name = Path(subject_dir).name
                if name == template_subject or name in seen:
                    skipped += 1
                    print(f"[eval] skip {name} "
                          f"({'template subject' if name == template_subject else 'duplicate'})",
                          flush=True)
                    continue
                seen.add(name)

                sample_seg = ds[i]['sample_seg'].unsqueeze(0).to(ctx['device'])
                row = {'subject': name, 'split': split, 'idx': i}
                vox, flows, affine = voxel_metrics(ctx, sample_seg)
                row.update(vox)
                if not args.no_mesh:
                    flow_inv, _aff, parity = svf_inverse(ctx, sample_seg)
                    row.update(mesh_metrics(verts_t, verts_np, faces, flows, affine,
                                            subject_dir, d['seg_filename'], args.push,
                                            args.inv_iter, args.inv_alpha,
                                            flow_inv=flow_inv, parity=parity,
                                            all_pushes=args.all_pushes))

                if fh is None:                          # header from the first row
                    fh = open(csv_path, 'w', newline='')
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                assert writer is not None
                dropped = [k for k in row if k not in writer.fieldnames]
                if dropped:      # first row set the header; don't lose columns silently
                    print(f"[eval] WARNING {name}: columns absent from the CSV header "
                          f"(first subject lacked them): {dropped}", flush=True)
                writer.writerow({k: row.get(k) for k in writer.fieldnames})
                fh.flush()                              # partial results survive a crash
                rows.append(row)

                msg = (f"[eval] {len(rows):4d} {name:>20s} [{split}] WM {row['dice_wm']:.4f} | "
                       f"fg {row['dice_fg_mean']:.4f} | fold {row['folding_pct']:.4f}%")
                if 'sym_mean_mm' in row:
                    msg += (f" | mesh {row['sym_mean_mm']:6.2f} mm (undef "
                            f"{row['undeformed_mean_mm']:6.2f}) | flip {row['flip_pct']:.4f}%")
                elif not args.no_mesh:
                    msg += " | mesh: no GT surface"
                print(msg, flush=True)
    finally:
        if fh is not None:
            fh.close()

    if not rows:
        raise SystemExit("no subjects evaluated")

    summary = {'checkpoint': ckpt, 'affine': use_affine, 'push': args.push,
               'mesh': not args.no_mesh, 'template_subject_excluded': template_subject,
               'n_subjects': len(rows), 'n_skipped': skipped}
    for split in splits:
        sub = [r for r in rows if r['split'] == split]
        if sub:
            summary[split] = summarize(sub, f'split={split}')
    if len(splits) > 1:
        summary['pooled'] = summarize(rows, 'pooled train+val')

    n_mesh = sum('sym_mean_mm' in r for r in rows)
    if not args.no_mesh:
        print(f"\n[eval] mesh metrics available for {n_mesh}/{len(rows)} subjects "
              f"(rest had no FreeSurfer white surface)")
    (out / 'metrics_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"[eval] per-subject CSV -> {csv_path}")
    print(f"[eval] summary        -> {out / 'metrics_summary.json'}")


if __name__ == '__main__':
    main()
