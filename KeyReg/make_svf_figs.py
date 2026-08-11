"""Figures for the fold-free SVF model: (1) Dice+folding curve, (2) template/warped/fixed."""
import os, re, sys
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, "/shared/home/v_vijay_bala_mahalingam/RegNET/KeyReg")
from keyreg import SVFReg, load_seg, read_list

RUN = "/shared/scratch/0/home/v_vijay_bala_mahalingam/keyreg_runs/svf_B"
OUT = "/shared/home/v_vijay_bala_mahalingam"
dev = "cuda:0"; ts = (128, 128, 128)

# ---- Figure 1: Dice + folding over epochs ----
ep, wm, fold = [], [], []
for ln in open(os.path.join(RUN, "train.log")):
    m = re.search(r"Ep (\d+)/\d+ .*WM ([\d.]+) \| fold ([\d.]+)%", ln)
    if m:
        ep.append(int(m.group(1))); wm.append(float(m.group(2))); fold.append(float(m.group(3)))
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(ep, wm, color="purple", lw=2, label="WM Dice (↑)")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("WM Dice", color="purple"); ax1.set_ylim(0.5, 1.0)
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(ep, fold, color="crimson", lw=1.5, ls="--", label="Folding % (↓)")
ax2.set_ylabel("Folding % (interior, det<0)", color="crimson"); ax2.set_ylim(0, 2)
ax2.axhline(0.25, color="gray", ls=":", lw=1)
plt.title("Keypoint-guided diffeomorphic SVF — Dice vs Folding")
f1 = os.path.join(OUT, "svf_curve.png"); plt.tight_layout(); plt.savefig(f1, dpi=140); plt.close()
print("saved", f1)

# ---- Figure 2: template / warped / fixed ----
ck = torch.load(os.path.join(RUN, "best.pth"), map_location=dev, weights_only=False)
a = ck["args"]
m = SVFReg(target=a["target"], delta=a["delta"], int_steps=a["int_steps"], K=a["K"]).to(dev)
m.load_state_dict(ck["model"]); m.eval()
tpl = load_seg("/shared/scratch/0/home/v_vijay_bala_mahalingam/neurite_oasis/OASIS_OAS1_0001_MR1/seg4_onehot.npy", ts).unsqueeze(0).to(dev)
val0 = read_list("/shared/scratch/0/home/v_vijay_bala_mahalingam/neurite_oasis/val.txt")[0]
fix = load_seg(val0, ts).unsqueeze(0).to(dev)
with torch.no_grad():
    warped, _, _, _ = m(tpl, fix)

def lab(x):
    a_ = x.squeeze().argmax(0).cpu().numpy()
    return a_[:, a_.shape[1] // 2, :]
cmap = ListedColormap(["#3b6ea5", "#6db33f", "#9e9e9e", "#7c3a3a", "#7fd4e0"])
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
for aa, t, v in zip(ax, ["Template (Moving)", "Warped (fold-free)", "Fixed"], [tpl, warped, fix]):
    aa.imshow(np.rot90(lab(v)), cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    aa.set_title(t); aa.axis("off")
f2 = os.path.join(OUT, "svf_overlay.png"); plt.tight_layout(); plt.savefig(f2, dpi=140); plt.close()
print("saved", f2)
