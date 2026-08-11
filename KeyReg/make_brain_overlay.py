"""Brain WM overlay comparison (axial/coronal/sagittal) for the keypoint-guided
registration: target WM vs warped-template WM (two configs), with HD95."""
import sys, os
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg")
from keyreg import SVFReg, load_seg, read_list

dev = "cuda:0"; ts = (128, 128, 128)
TPL = "/shared/scratch/0/home/v_vijay_bala_mahalingam/neurite_oasis/OASIS_OAS1_0001_MR1/seg4_onehot.npy"
VAL = "/shared/scratch/0/home/v_vijay_bala_mahalingam/neurite_oasis/val.txt"

def build(d):
    ck = torch.load(f"/shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/{d}/best.pth",
                    map_location=dev, weights_only=False)
    a = ck["args"]
    m = SVFReg(target=a["target"], delta=a["delta"], int_steps=a["int_steps"], K=a["K"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval(); return m

def wm(seg):  # (1,5,*) -> binary WM (class 3)
    return (seg.squeeze().argmax(0) == 3).cpu().numpy().astype(np.uint8)

def hd95(a, b):  # symmetric surface distance 95th pct (grid units)
    da = distance_transform_edt(1 - a); db = distance_transform_edt(1 - b)
    sa = a - (distance_transform_edt(a) > 1); sb = b - (distance_transform_edt(b) > 1)
    d_ab = db[sa > 0]; d_ba = da[sb > 0]
    alld = np.concatenate([d_ab, d_ba])
    return np.percentile(alld, 95), alld.mean()

tpl = load_seg(TPL, ts).unsqueeze(0).to(dev)
subj = read_list(VAL)[3]                      # pick a val subject
fix = load_seg(subj, ts).unsqueeze(0).to(dev)
name = os.path.basename(os.path.dirname(subj))

mE, mF = build("svf_E"), build("svf_F")
with torch.no_grad():
    wE = mE(tpl, fix)[0]; wF = mF(tpl, fix)[0]
wm_fix, wm_E, wm_F = wm(fix), wm(wE), wm(wF)
h95E, mE_d = hd95(wm_E, wm_fix)
h95F, mF_d = hd95(wm_F, wm_fix)

# best slice per axis (most WM)
za = wm_fix.sum((1, 2)).argmax(); ya = wm_fix.sum((0, 2)).argmax(); xa = wm_fix.sum((0, 1)).argmax()
views = [("axial (z)", lambda v: v[za, :, :], za),
         ("coronal (y)", lambda v: v[:, ya, :], ya),
         ("sagittal (x)", lambda v: v[:, :, xa], xa)]

fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="black")
fig.suptitle(f"{name} — WM overlay    balanced: mean {mE_d:.2f} / hd95 {h95E:.2f}    "
             f"fold-free: mean {mF_d:.2f} / hd95 {h95F:.2f}   (128³ grid units)",
             color="white", fontsize=12)
for a, (title, sl, idx) in zip(ax, views):
    a.set_facecolor("black")
    a.imshow(np.rot90(sl(wm_fix)), cmap="gray", vmin=0, vmax=1.6, interpolation="nearest")
    a.contour(np.rot90(sl(wm_fix)), levels=[0.5], colors="#43d17a", linewidths=1.1)  # target
    a.contour(np.rot90(sl(wm_E)),   levels=[0.5], colors="#4aa8ff", linewidths=1.0)  # balanced
    a.contour(np.rot90(sl(wm_F)),   levels=[0.5], colors="#ff5db1", linewidths=1.0)  # fold-free
    a.set_title(f"{title} @ {idx}/127", color="white", fontsize=11); a.axis("off")
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], color="#43d17a", lw=2, label="Target WM (GT)"),
       Line2D([0], [0], color="#4aa8ff", lw=2, label="Warped (balanced)"),
       Line2D([0], [0], color="#ff5db1", lw=2, label="Warped (fold-free)")]
ax[0].legend(handles=leg, loc="lower left", fontsize=8, framealpha=0.3, labelcolor="white")
out = "/shared/home/v_vijay_bala_mahalingam/vijay_brain_overlay_2026-08-07.png"
plt.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, facecolor="black", dpi=150); print("saved", out)
