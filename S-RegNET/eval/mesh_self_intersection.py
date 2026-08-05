"""Self-intersection % of the pushed genus-0 WM mesh — the number to look at is `warp pp`.

Examples:
    python eval/mesh_self_intersection.py --checkpoint /shared/scratch/0/home/v_nishchay_nilabh/training_results/oasis_invertible_deform/20260626_070438 --output_dir /shared/scratch/0/home/v_nishchay_nilabh/training_results/oasis_invertible_deform/20260626_070438/mesh_selfint_n5 --num_samples 5 --device cuda:4
    python eval/mesh_self_intersection.py --checkpoint /shared/scratch/0/home/v_nishchay_nilabh/training_results/oasis_invertible_deform/20260626_070438 --template_mesh /shared/scratch/0/home/v_nishchay_nilabh/oasis_data/meshes/OASIS_OAS1_0406_MR1/repaired/template_wm_mesh_repaired.npz --output_dir /shared/scratch/0/home/v_nishchay_nilabh/training_results/oasis_invertible_deform/20260626_070438/mesh_selfint_n5_repaired --num_samples 5 --device cuda:4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from git_provenance import write_git_sha
from get_data import SegDataset
from inference import load_config, setup_inference
from experiment_routing import build_step_inputs
from wm_template import load_template_wm_mesh, self_intersections


def resolve_checkpoint(spec, cfg):
    """--checkpoint as a .pth, a run dir, or a run name under output.base_dir."""
    sub = cfg['output'].get('checkpoint_subdir', 'checkpoints')
    p = Path(spec).expanduser()
    candidates = [p, p / sub / 'best_model.pth',
                  Path(cfg['output']['base_dir']) / spec / sub / 'best_model.pth']
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError(
        f"Could not resolve --checkpoint '{spec}'. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates))


def detect_affine(ckpt_path):
    """Whether the checkpoint carries an affine head (checkpoint, not config, decides)."""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model_state_dict']
    return any(k.startswith('affine_net') for k in sd)


@torch.no_grad()
def push_mesh(ctx, template_verts, input_seg, n_iter):
    """Returns (pushed verts (N,3) in subject normalized coords, inverse residual in voxels)."""
    model, n = ctx['model'], ctx['target_size'][0]
    cps_list, affine_matrix = model(ctx['template_seg'], input_seg)
    pushed, res = model.push_points_to_sample(template_verts, cps_list, affine_matrix,
                                              n_iter=n_iter)
    return pushed.cpu().numpy(), (res * (n / 2.0)).cpu().numpy()


def selfint_colors(faces, hit, base, n_verts):
    """Vertex colours: grey clean, amber inherited from template, red warp-created."""
    col = np.full((n_verts, 3), 200, np.uint8)
    for sel, rgb in ((hit & base, (255, 170, 0)), (hit & ~base, (220, 30, 30))):
        col[np.unique(faces[sel])] = rgb
    return col


def write_ply(path, verts, faces, color):
    """Binary little-endian PLY with per-vertex colour."""
    vd = np.empty(len(verts), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for k, i in (('x', 0), ('y', 1), ('z', 2)):
        vd[k] = verts[:, i]
    for k, i in (('red', 0), ('green', 1), ('blue', 2)):
        vd[k] = color[:, i]
    fd = np.empty(len(faces), dtype=[('n', 'u1'), ('a', '<i4'), ('b', '<i4'), ('c', '<i4')])
    fd['n'] = 3
    for k, i in (('a', 0), ('b', 1), ('c', 2)):
        fd[k] = faces[:, i]
    assert vd.itemsize == 15 and fd.itemsize == 13, 'structured dtype got padded'
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
            "end_header\n")
    with open(path, 'wb') as f:
        f.write(head.encode('ascii'))
        f.write(vd.tobytes())
        f.write(fd.tobytes())


def write_selfint_ply(path, verts_norm, faces, colors, ref_nii):
    """PLY in world mm via `ref_nii`, so before/after meshes load into one viewer."""
    import nibabel as nib
    from overlay_surface import norm_to_world
    write_ply(path, norm_to_world(verts_norm, nib.load(ref_nii)), faces, colors)


def main():
    ap = argparse.ArgumentParser(
        description="Self-intersection of the pushed genus-0 WM mesh")
    ap.add_argument('--checkpoint', required=True,
                    help='run dir, run name under output.base_dir, or best_model.pth')
    ap.add_argument('--config', default=None, help='config.yaml (default: ./config.yaml)')
    ap.add_argument('--output_dir', default=None, help='default <run_dir>/mesh_selfint')
    ap.add_argument('--val_txt', default=None, help='override config data.val_txt')
    ap.add_argument('--template_mesh', default=None,
                    help='override config data.template_wm_mesh_path (repaired-template A/B)')
    ap.add_argument('--num_samples', type=int, default=5)
    ap.add_argument('--sample_idxs', default=None, help='comma list of val indices')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--inv_iter', type=int, default=50,
                    help='fixed-point iterations for the field inverse')
    ap.add_argument('--chunk', type=int, default=500_000,
                    help='candidate pairs narrowed per batch (~1 GB peak at 500k)')
    ap.add_argument('--ply', action='store_true',
                    help='write a colour-coded PLY per subject (~13 MB each)')
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(args.checkpoint, cfg)
    use_affine = detect_affine(ckpt)
    out = Path(args.output_dir).expanduser() if args.output_dir \
        else Path(ckpt).parent.parent / 'mesh_selfint'
    out.mkdir(parents=True, exist_ok=True)
    write_git_sha(out)

    print(f"[DEBUG] checkpoint {ckpt}\n[DEBUG] affine {use_affine} | device "
          f"{args.device} | out {out}", flush=True)

    ctx = setup_inference(ckpt, args.config, args.device,
                          use_affine=use_affine, verbose=False)
    device = ctx['device']

    mesh_path = args.template_mesh or cfg['data']['template_wm_mesh_path']
    verts_np, faces, _ = load_template_wm_mesh(mesh_path)
    template_verts = torch.tensor(verts_np, dtype=torch.float32, device=device)
    print(f"[DEBUG] mesh {mesh_path} | {len(verts_np):,} verts | {len(faces):,} faces",
          flush=True)

    # Baseline once: the affine is linear, so it neither creates nor removes a crossing.
    tpl, tpl_hit = self_intersections(verts_np, faces, label='template')

    exp = cfg.get('experiment', {})
    input_source = exp.get('input_source', 'synthseg')
    supervision_target = exp.get('supervision_target', 'gt')

    d = cfg['data']
    ds = SegDataset(args.val_txt or d['val_txt'], d['template_seg_path'],
                    target_size=ctx['target_size'], seg_filename=d['seg_filename'],
                    synthseg_filename=d['synthseg_filename'], preload=False,
                    load_synthseg='synthseg' in (input_source, supervision_target))
    idxs = [int(x) for x in args.sample_idxs.split(',')] if args.sample_idxs else \
        sorted(set(np.linspace(0, len(ds) - 1,
                               min(args.num_samples, len(ds))).astype(int).tolist()))
    print(f"[DEBUG] input {input_source} | samples {idxs}", flush=True)

    rows = []
    for k, i in enumerate(idxs, 1):
        subject_dir = ds.subject_dirs[i]
        name = Path(subject_dir).name
        sample = ds[i]
        gt_seg = sample['gt_seg'].unsqueeze(0).to(device)
        ss_seg = (sample['synthseg_seg'].unsqueeze(0).to(device)
                  if sample['synthseg_seg'] is not None else None)
        input_seg, _ = build_step_inputs(
            ss_seg, gt_seg, input_source=input_source,
            supervision_target=supervision_target)

        pushed, res_vox = push_mesh(ctx, template_verts, input_seg, args.inv_iter)
        stats, hit = self_intersections(pushed, faces, label=name, chunk=args.chunk)

        r = {'subject': name, 'idx': int(i), **stats,
             'si_faces_pct_warp': stats['si_faces_pct'] - tpl['si_faces_pct'],
             'inv_res_max_vox': float(res_vox.max()),
             'inv_res_median_vox': float(np.median(res_vox)),
             'inv_res_over_0p1vox_pct': float((res_vox > 0.1).mean() * 100)}
        rows.append(r)
        print(f"[DEBUG] {k}/{len(idxs)} {name}: pushed {r['si_faces_pct']:.4f}% | "
              f"warp {r['si_faces_pct_warp']:+.4f} pp", flush=True)

        if args.ply:
            write_selfint_ply(out / f'{name}_pushed_selfint.ply', pushed, faces,
                              selfint_colors(faces, hit, tpl_hit, len(pushed)),
                              os.path.join(subject_dir, 'seg4.nii.gz'))

    pct = np.array([r['si_faces_pct'] for r in rows])
    res = np.array([r['inv_res_max_vox'] for r in rows])
    # res max vox must stay well under 1; if not, raise --inv_iter and re-read.
    print(f"[DEBUG] {'subject':<26}{'pushed %':>10}{'warp pp':>10}{'tris':>9}"
          f"{'res max vox':>13}")
    for r in rows:
        print(f"[DEBUG] {r['subject']:<26}{r['si_faces_pct']:>10.4f}"
              f"{r['si_faces_pct_warp']:>+10.4f}{r['si_faces']:>9,}"
              f"{r['inv_res_max_vox']:>13.4f}")
    print(f"[DEBUG] {'template (inherited)':<26}{tpl['si_faces_pct']:>10.4f}{'-':>10}"
          f"{tpl['si_faces']:>9,}{'-':>13}")

    (out / 'mesh_self_intersection.json').write_text(json.dumps(
        {'checkpoint': ckpt, 'affine': use_affine, 'template_wm_mesh': mesh_path,
         'input_source': input_source, 'inv_iter': args.inv_iter,
         'n_verts': len(verts_np), 'n_faces': len(faces),
         'template': tpl, 'per_sample': rows,
         'aggregate': {'si_faces_pct_mean': float(pct.mean()),
                       'si_faces_pct_std': float(pct.std()),
                       'si_faces_pct_warp_mean': float(pct.mean() - tpl['si_faces_pct']),
                       'inv_res_max_vox_worst': float(res.max())}}, indent=2))
    print(f"[DEBUG] json -> {out / 'mesh_self_intersection.json'}")

    print("=" * 78)
    print(f"SELF INTERSECTION (%) = {pct.mean():.4f}   "
          f"(template {tpl['si_faces_pct']:.4f} + warp-created "
          f"{pct.mean() - tpl['si_faces_pct']:+.4f})")


if __name__ == '__main__':
    main()
