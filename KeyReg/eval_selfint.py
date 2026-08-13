"""Measure surface self-intersection (triangle-flip %) of the warped WM surface
for the SVF models. Extract template WM surface via marching cubes, push it by
each model's deformation for every val subject, count orientation-flipped
(self-intersecting) triangles."""
import sys, os, argparse
import numpy as np, torch, torch.nn.functional as F
from skimage import measure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keyreg import SVFReg, load_seg, read_list, dice_per_class, identity_grid_vol

_D = "/shared/scratch/0/home/v_vijay_bala_mahalingam"
ap = argparse.ArgumentParser()
ap.add_argument("--val", default=f"{_D}/neurite_oasis/val.txt")
ap.add_argument("--template", default=f"{_D}/neurite_oasis/OASIS_OAS1_0001_MR1/seg4_onehot.npy")
ap.add_argument("--runs", nargs="+", default=["svf_E:SVF-E (Dice 0.910)", "svf_F:SVF-F (Dice 0.841)"],
                help="one or more DIRNAME:LABEL under keyreg_runs/")
args = ap.parse_args()

dev = "cuda:0"; ts = (128, 128, 128)
TPL = args.template
VAL = args.val

# ---- template WM surface (channel 3) ----
tpl = load_seg(TPL, ts)                       # (5,D,H,W)
wm = tpl[3].numpy()
verts, faces, _, _ = measure.marching_cubes(wm, 0.5)   # verts in (i,j,k)=(D,H,W) voxel coords
D, H, W = wm.shape
vx = verts[:, 2] / (W - 1) * 2 - 1            # W->x
vy = verts[:, 1] / (H - 1) * 2 - 1            # H->y
vz = verts[:, 0] / (D - 1) * 2 - 1            # D->z
vn = torch.tensor(np.stack([vx, vy, vz], 1), dtype=torch.float32, device=dev)  # (N,3) (x,y,z)
faces_t = torch.tensor(faces.astype(np.int64), device=dev)
print(f"template WM surface: {len(vn):,} verts, {len(faces):,} faces")

def flip_pct(v0, v1):
    a0, b0, c0 = v0[faces_t[:, 0]], v0[faces_t[:, 1]], v0[faces_t[:, 2]]
    a1, b1, c1 = v1[faces_t[:, 0]], v1[faces_t[:, 1]], v1[faces_t[:, 2]]
    n0 = torch.cross(b0 - a0, c0 - a0, dim=1)
    n1 = torch.cross(b1 - a1, c1 - a1, dim=1)
    return ((n0 * n1).sum(1) < 0).float().mean().item() * 100

def build(ck):
    a = ck["args"]
    m = SVFReg(target=a["target"], delta=a["delta"], int_steps=a["int_steps"], K=a["K"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval(); return m

idg = identity_grid_vol(128, dev)
tpl_b = tpl.unsqueeze(0).to(dev)
va = read_list(VAL)

for spec in args.runs:
    d, _, name = spec.partition(":")
    name = name or d
    p = f"{_D}/keyreg_runs/{d}/best.pth"
    ck = torch.load(p, map_location=dev, weights_only=False)
    m = build(ck)
    flips = []; pcs = np.zeros(5)
    with torch.no_grad():
        gsv = vn.view(1, 1, 1, -1, 3)
        for fp in va:
            fix = load_seg(fp, ts).unsqueeze(0).to(dev)
            warped, grid, _, _ = m(tpl_b, fix)
            pcs += np.array(dice_per_class(warped, fix))
            disp = (grid - idg).permute(0, 4, 1, 2, 3)                  # pull disp (B,3,D,H,W)
            d_at = F.grid_sample(disp, gsv, mode="bilinear", padding_mode="border",
                                 align_corners=True)[0, :, 0, 0, :].T   # (N,3)
            v_new = vn - d_at                                          # forward push
            flips.append(flip_pct(vn, v_new))
    print(f"=== {name} (epoch {ck.get('epoch')}) ===")
    print(f"  WM Dice: {pcs[3]/len(va):.4f} | logged folding: {ck.get('fold'):.4f}%")
    print(f"  self-intersection (triangle-flip %): {np.mean(flips):.4f}%")
