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
    drives this to ~0.

    Beware: on a real pushed cortical mesh most of this count is a metric
    artifact — a normal can rotate past 90 degrees under pure shear at det > 0,
    and near-degenerate slivers reverse on interpolation noise alone. Score the
    deliverable with `self_intersections`, which is a genuine embedding failure."""
    nb = _face_normals(np.asarray(verts_before), np.asarray(faces))
    na = _face_normals(np.asarray(verts_after), np.asarray(faces))
    return float(((nb * na).sum(1) < 0).mean())


# =============================================================================
# Self-intersection — the embedding check (pure numpy)
# =============================================================================
# Moving vertices cannot change V - E + F, so a pushed mesh is combinatorially
# genus-0 however violent the warp; `mesh_genus` on it is a tautology. What a
# warp CAN destroy is the EMBEDDING: the surface passing through itself. That is
# the only mesh number the genus-0 deliverable can actually fail on, and this is
# it. Scale-free (the broadphase cell is sized off the triangles themselves), so
# it reads the same on normalized [-1,1] verts or world mm.

def _seg_hits_tri(P, Q, T, eps=1e-14):
    """Möller-Trumbore: does the segment P->Q pierce triangle T? All (N,3)/(N,3,3)."""
    D, e1, e2 = Q - P, T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    pv = np.cross(D, e2)
    det = (e1 * pv).sum(1)
    ok = np.abs(det) >= eps                       # parallel segment: no crossing
    inv = np.zeros_like(det)
    np.divide(1.0, det, out=inv, where=ok)
    tv = P - T[:, 0]
    qv = np.cross(tv, e1)
    u = (tv * pv).sum(1) * inv                    # barycentric in T
    w = (D * qv).sum(1) * inv
    t = (e2 * qv).sum(1) * inv                    # position along the segment
    return ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t >= 0) & (t <= 1)


def _tri_tri_cross(A, B):
    """Do the paired triangles A[i], B[i] cross? Two triangles intersect iff an
    edge of one pierces the other, so six segment tests decide it. Coplanar
    overlap is measure-zero and is not detected."""
    hit = np.zeros(len(A), dtype=bool)
    for X, Y in ((A, B), (B, A)):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            hit |= _seg_hits_tri(X[:, a], X[:, b], Y)
    return hit


def _cell_pairs(lo, hi, h):
    """Candidate pairs: triangles whose boxes share a cell of a uniform grid.

    Every triangle is inserted into each cell its own box overlaps. A triangle
    several times the typical size then pays for itself in extra cells, instead
    of forcing every other triangle to be queried at ITS radius — the failure
    mode that makes a single-radius neighbour query blow up on a stretched
    surface (311M candidates for 954K real pairs, measured)."""
    c_lo = np.floor(lo / h).astype(np.int64)
    c_hi = np.floor(hi / h).astype(np.int64)
    span = c_hi - c_lo + 1                                   # cells per axis
    cnt = span.prod(1)
    tri = np.repeat(np.arange(len(lo)), cnt)
    k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    nx, ny = np.repeat(span[:, 0], cnt), np.repeat(span[:, 1], cnt)
    off = np.stack([k % nx, (k // nx) % ny, k // (nx * ny)], 1)
    g = np.repeat(c_lo, cnt, axis=0) + off - c_lo.min(0)     # non-negative cell
    dim = c_hi.max(0) - c_lo.min(0) + 1
    ids = (g[:, 2] * dim[1] + g[:, 1]) * dim[0] + g[:, 0]

    order = np.argsort(ids, kind='stable')
    ids_s, tri_s = ids[order], tri[order]
    start = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])
    size = np.diff(np.r_[start, len(ids_s)])
    out = []
    for s in np.unique(size[size > 1]):                      # a few distinct sizes
        blk = tri_s[start[size == s][:, None] + np.arange(s)]
        a, b = np.triu_indices(s, 1)
        out.append(np.stack([blk[:, a].ravel(), blk[:, b].ravel()], 1))
    if not out:
        return np.zeros((0, 2), np.int64)
    p = np.vstack(out)
    p.sort(1)                                                # i<j, then dedupe
    key = p[:, 0] * (len(lo) + 1) + p[:, 1]
    return p[np.unique(key, return_index=True)[1]]


def _face_clusters(faces, sel):
    """(n_clusters, largest) over the SELECTED faces, adjacency = shared vertex.
    A few crossing patches read very differently from thousands of singletons.
    Union-find in plain python: the selection is a few hundred faces."""
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        return 0, 0
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f in idx:                                   # a face joins with its verts
        for node in (('f', int(f)), *(('v', int(v)) for v in faces[f])):
            parent.setdefault(node, node)
        rf = find(('f', int(f)))
        for v in faces[f]:
            rv = find(('v', int(v)))
            if rv != rf:
                parent[rv] = rf
                rf = find(rf)
    lab = {}
    for f in idx:
        root = find(('f', int(f)))
        lab[root] = lab.get(root, 0) + 1
    return len(lab), max(lab.values())


def self_intersections(verts, faces, cell=2.0, chunk=500_000, label='', quiet=True):
    """Faces of this mesh that cross a NON-ADJACENT face of the same mesh.

    Uniform-grid broadphase over triangle AABBs, narrowed by the boxes
    themselves, then the exact six-segment triangle test. Pairs sharing a vertex
    are dropped — neighbours always touch at it.

    FreeSurfer ?h.white is not guaranteed intersection-free, so the un-pushed
    template scores nonzero too (0.1007% on OASIS_OAS1_0406_MR1); subtract that
    baseline and only the remainder is the warp's. The bounded-FFD clamp makes
    every stage injective, so the remainder is expected to be ~0 and anything
    larger is either the inverse solver not converging (watch its residual) or
    sub-voxel non-injectivity of the trilinear interpolant.

    Args:
        verts: (N, 3) vertices, any consistent units.
        faces: (M, 3) triangles.
        cell:  broadphase cell size in units of the 99th-pct triangle radius.
        chunk: candidate pairs narrowed per batch. The narrowphase materialises
               ~150 bytes per pair (two float64 (3,3) triangles plus the cross
               products), so 500k is ~1 GB peak — the value the pushed-mesh runs
               were measured at. Raising it trades RAM for fewer iterations, and
               on a 72M-candidate pushed mesh 2M can push a small box into swap,
               which looks exactly like a hang.
    Returns:
        (stats dict, hit (M,) bool mask of crossing faces).
    """
    tri = np.asarray(verts)[faces].astype(np.float64)
    lo, hi = tri.min(1), tri.max(1)
    rad = np.linalg.norm(tri - tri.mean(1)[:, None, :], axis=2).max(1)
    h = cell * float(np.percentile(rad, 99.0))
    # Printed BEFORE the broadphase: it is one silent numpy block, and a mesh
    # whose extent blew up (a diverged inverse) shows here rather than after the
    # minutes it would then cost.
    if not quiet:
        print(f"      [self-int {label}] cell {h:.5f}, extent "
              f"{float(np.ptp(lo, axis=0).max()):.3f}, broadphase...",
              end='', flush=True)
    pairs = _cell_pairs(lo, hi, h)
    if not quiet:
        print(f" {len(pairs):,} candidates", end='', flush=True)

    hit, n_pairs, n_narrow = np.zeros(len(faces), dtype=bool), 0, 0
    for s in range(0, len(pairs), chunk):
        i, j = pairs[s:s + chunk].T
        m = (lo[i] <= hi[j]).all(1) & (lo[j] <= hi[i]).all(1)             # boxes overlap
        i, j = i[m], j[m]
        m = ~(faces[i][:, :, None] == faces[j][:, None, :]).any((1, 2))   # not neighbours
        i, j = i[m], j[m]
        n_narrow += len(i)
        x = _tri_tri_cross(tri[i], tri[j])
        n_pairs += int(x.sum())
        hit[i[x]] = True
        hit[j[x]] = True
    if not quiet:
        print(f" -> {n_narrow:,} narrowphase -> {n_pairs:,} crossings", flush=True)

    n_c, big = _face_clusters(faces, hit)
    return {'si_candidates': int(len(pairs)), 'si_narrowphase': int(n_narrow),
            'si_pairs': int(n_pairs), 'si_faces': int(hit.sum()),
            'si_faces_pct': float(hit.mean() * 100),
            'si_clusters': n_c, 'si_largest': big}, hit


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
