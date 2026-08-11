"""
Build the genus-0 white-matter template mesh.

The template subject's FreeSurfer lh.white + rh.white surfaces are genus-0
spheres, so we use them directly (no marching cubes or handle removal). Output
is an .npz at data.template_wm_mesh_path:

    verts_norm : (N, 3) float32  vertices in normalized [-1,1] (x,y,z) grid
                 coords of the template seg grid (the frame the dense flow lives
                 in, so the FFD samples displacements at them directly).
    faces      : (M, 3) int64    triangles; rh indices offset by n_lh.
    n_lh       : int             number of left-hemisphere vertices.

Coordinate chain: surface-RAS + cras = scanner RAS -> world_to_norm(ref) ->
normalized [-1,1], where ref is the template seg4.nii.gz. nibabel is imported
lazily so the pure-numpy helpers import without it.

Usage:
    python wm_template.py
    python wm_template.py --lh_white <lh.white> --rh_white <rh.white> \\
                          --ref <seg4.nii.gz> --output <template_wm_mesh.npz>
"""

import argparse
from pathlib import Path

import numpy as np


# =============================================================================
# Topology checks (pure numpy)
# =============================================================================

def mesh_genus(faces, n_verts=None):
    """Genus of a single connected closed orientable triangle mesh.

    g = 1 - chi/2 with Euler characteristic chi = V - E + F. Sphere -> 0,
    torus -> 1. Assumes one connected component; check lh/rh separately (two
    disjoint spheres give chi=4 and a meaningless g=-1).

    Args:
        faces: (M, 3) int array of triangle vertex indices.
        n_verts: vertex count; defaults to the number of distinct indices used.
    """
    faces = np.asarray(faces)
    if n_verts is None:
        n_verts = int(np.unique(faces).size)
    V = int(n_verts)
    F = int(len(faces))
    E = int(len(mesh_edges(faces)))
    euler = V - E + F
    return (2 - euler) // 2


def mesh_edges(faces):
    """Unique undirected edges (E, 2) int64, each sorted (lo, hi), from triangle
    faces (M, 3)."""
    faces = np.asarray(faces)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0).astype(np.int64)


def _face_normals(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return np.cross(v1 - v0, v2 - v0)


def triangle_flip_fraction(verts_before, verts_after, faces):
    """Fraction of faces whose orientation reversed under a deformation
    (dot(normal_before, normal_after) < 0) — the surface analogue of a negative
    Jacobian, which voxel folding alone does not capture. A diffeomorphic warp
    drives this to ~0."""
    nb = _face_normals(np.asarray(verts_before), np.asarray(faces))
    na = _face_normals(np.asarray(verts_after), np.asarray(faces))
    return float(((nb * na).sum(1) < 0).mean())


# =============================================================================
# Build
# =============================================================================

def _derive_paths(cfg):
    """Derive (lh.white, rh.white, ref seg4.nii.gz, output npz) from config. The
    template seg lives under .../scans/<subject>/ and the FreeSurfer surfaces
    under .../meshes/<subject>/{lh,rh}.white."""
    template_seg_path = cfg['data']['template_seg_path']
    ref = template_seg_path.replace('seg4_onehot.npy', 'seg4.nii.gz')
    mesh_dir = str(Path(template_seg_path).parent).replace('/scans/', '/meshes/')
    lh = str(Path(mesh_dir) / 'lh.white')
    rh = str(Path(mesh_dir) / 'rh.white')
    out = cfg['data'].get('template_wm_mesh_path',
                          template_seg_path.replace('seg4_onehot.npy', 'template_wm_mesh.npz'))
    return lh, rh, ref, out


def build_template_wm_mesh(lh_white, rh_white, ref_path, output_path):
    """Load both hemispheres, map to normalized coords, concat, genus-check,
    save the .npz. Returns (verts_norm, faces, n_lh)."""
    import nibabel as nib
    from overlay_surface import world_to_norm   # lazy: pulls nibabel

    ref = nib.load(ref_path)

    def _hemi(path):
        coords, faces, meta = nib.freesurfer.read_geometry(path, read_metadata=True)
        world = coords + meta.get('cras', np.zeros(3))        # surface-RAS → scanner RAS
        return world_to_norm(world, ref).astype(np.float32), np.asarray(faces, dtype=np.int64)

    lh_verts, lh_faces = _hemi(lh_white)
    rh_verts, rh_faces = _hemi(rh_white)

    # Check each hemisphere before concatenation: a white surface must be genus-0.
    g_lh, g_rh = mesh_genus(lh_faces, len(lh_verts)), mesh_genus(rh_faces, len(rh_verts))
    assert g_lh == 0, f"lh.white genus {g_lh} != 0 — not a genus-0 surface"
    assert g_rh == 0, f"rh.white genus {g_rh} != 0 — not a genus-0 surface"

    n_lh = len(lh_verts)
    verts = np.concatenate([lh_verts, rh_verts], axis=0)
    faces = np.concatenate([lh_faces, rh_faces + n_lh], axis=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, verts_norm=verts, faces=faces, n_lh=np.int64(n_lh))

    # Normalized coords should sit inside the [-1,1] FOV; a large out-of-box
    # fraction means a wrong ref or axis order.
    inside = ((verts >= -1) & (verts <= 1)).all(1).mean() * 100
    print(f"Saved: {output_path}")
    print(f"  lh verts {n_lh:,} (genus {g_lh}) | rh verts {len(rh_verts):,} (genus {g_rh})")
    print(f"  total verts {len(verts):,} | faces {len(faces):,}")
    print(f"  normalized bbox  x[{verts[:,0].min():.3f},{verts[:,0].max():.3f}] "
          f"y[{verts[:,1].min():.3f},{verts[:,1].max():.3f}] "
          f"z[{verts[:,2].min():.3f},{verts[:,2].max():.3f}]")
    print(f"  inside [-1,1] FOV: {inside:.1f}%  (low → wrong ref or axis order)")
    return verts, faces, n_lh


def load_template_wm_mesh(path):
    """Load the saved template mesh. Returns (verts_norm (N,3) f32, faces (M,3)
    i64, n_lh int)."""
    d = np.load(path)
    return d['verts_norm'].astype(np.float32), d['faces'].astype(np.int64), int(d['n_lh'])


def main():
    ap = argparse.ArgumentParser(description="Build the genus-0 WM template mesh (.npz)")
    ap.add_argument('--config', default=None, help='config.yaml (default: ./config.yaml)')
    ap.add_argument('--lh_white', default=None, help='override lh.white path')
    ap.add_argument('--rh_white', default=None, help='override rh.white path')
    ap.add_argument('--ref', default=None, help='override template seg4.nii.gz ref')
    ap.add_argument('--output', default=None, help='override output .npz path')
    args = ap.parse_args()

    import yaml
    config_path = args.config or (Path(__file__).parent / "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    lh, rh, ref, out = _derive_paths(cfg)
    lh, rh = args.lh_white or lh, args.rh_white or rh
    ref, out = args.ref or ref, args.output or out

    print("Building genus-0 WM template mesh")
    print(f"  lh.white : {lh}")
    print(f"  rh.white : {rh}")
    print(f"  ref      : {ref}")
    build_template_wm_mesh(lh, rh, ref, out)


if __name__ == '__main__':
    main()
