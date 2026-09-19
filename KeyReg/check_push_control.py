"""Control check: is the deformed mesh actually closer to the SAMPLE WM than the
unpushed template is? If not, the mesh push is not doing its job."""
import sys, os
import numpy as np, torch, torch.nn.functional as F
import nibabel as nib
from skimage import measure
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keyreg import SVFReg, load_seg, read_list, identity_grid_vol, svf_integrate, dice_per_class

_D = "/shared/scratch/0/home/v_vijay_bala_mahalingam"
dev, ts = "cuda:0", (128, 128, 128)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

tpl = load_seg(f"{_D}/neurite_oasis/OASIS_OAS1_0001_MR1/seg4_onehot.npy", ts)
wm = tpl[3].numpy(); D, H, W = wm.shape
v, f, _, _ = measure.marching_cubes(wm, 0.5)
vn = torch.tensor(np.stack([v[:,2]/(W-1)*2-1, v[:,1]/(H-1)*2-1, v[:,0]/(D-1)*2-1],1),
                  dtype=torch.float32, device=dev)
print(f"template WM surface: {len(vn):,} verts", flush=True)

def samp(field, pts):
    g = pts.view(1,1,1,-1,3)
    return F.grid_sample(field, g, mode="bilinear", padding_mode="border",
                         align_corners=True)[0,:,0,0,:].T


def norm_to_world(points, fp):
    """Model-normalized (x,y,z) -> scanner RAS millimetres.

    The 160x192x224 input is resized to 128^3, so normalized coordinates have
    a different physical scale on each axis. Converting through the original
    aligned volume affine avoids the invalid single-scale approximation.
    """
    subj_dir = os.path.dirname(fp)
    im = nib.load(os.path.join(subj_dir, "aligned_seg4.nii.gz"))
    nd = np.asarray(im.shape, dtype=np.float64)
    p = points.detach().cpu().numpy() if torch.is_tensor(points) else points
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    vox = np.stack([(z + 1) / 2 * (nd[0] - 1),
                    (y + 1) / 2 * (nd[1] - 1),
                    (x + 1) / 2 * (nd[2] - 1)], 1)
    return (np.c_[vox, np.ones(len(vox))] @ im.affine.T)[:, :3]

ck = torch.load(f"{_D}/keyreg_runs/svf_E_full/best.pth", map_location=dev, weights_only=False)
a = ck["args"]
m = SVFReg(target=a["target"], delta=a["delta"], int_steps=a["int_steps"], K=a["K"]).to(dev)
m.load_state_dict(ck["model"]); m.eval()
idg = identity_grid_vol(128, dev); tpl_b = tpl.unsqueeze(0).to(dev)
va = read_list(f"{_D}/neurite_oasis/full_val.txt")[:N]

acc = {k: [] for k in ("template(unpushed)", "push v-disp(v)", "push exp(-vel)", "push +disp(v)")}
dices = []
with torch.no_grad():
    for fp in va:
        fix = load_seg(fp, ts).unsqueeze(0).to(dev)
        x = torch.cat([tpl_b, fix], 1)
        if m.kp_guided:
            x = torch.cat([x, m._sal(tpl_b), m._sal(fix)], 1)
        vel = m.flow(x)
        disp = svf_integrate(vel, m.int_steps, idg)
        grid = idg + disp.permute(0,2,3,4,1)
        warped = F.grid_sample(tpl_b, grid, mode="bilinear", padding_mode="border", align_corners=True)
        dices.append(dice_per_class(warped, fix)[3])

        sv, _, _, _ = measure.marching_cubes(fix[0,3].cpu().numpy(), 0.5)
        svn = np.stack([sv[:,2]/(W-1)*2-1, sv[:,1]/(H-1)*2-1, sv[:,0]/(D-1)*2-1],1)
        tree = cKDTree(norm_to_world(svn, fp))

        cand = {
            "template(unpushed)": vn,
            "push v-disp(v)":     vn - samp(disp, vn),
            "push exp(-vel)":     vn + samp(svf_integrate(-vel, m.int_steps, idg), vn),
            "push +disp(v)":      vn + samp(disp, vn),
        }
        for k, pv in cand.items():
            acc[k].append(tree.query(norm_to_world(pv, fp))[0].mean())

print(f"\nWM Dice of warped template vs sample: {np.mean(dices):.4f}   ({N} subjects)")
print("\nmean distance to the SAMPLE's WM surface (mm) -- lower is better:")
for k in acc:
    print(f"   {k:<22s} {np.mean(acc[k]):7.3f}")
b = np.mean(acc["template(unpushed)"]); e = np.mean(acc["push exp(-vel)"])
print(f"\ndeformation closes {100*(b-e)/b:.1f}% of the template->sample gap "
      f"({b:.3f} -> {e:.3f} mm)")
