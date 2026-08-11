"""Render a presentation-style summary slide for the keypoint-guided registration work."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG = "#f4efe1"; INK = "#1a1a1a"
fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

def T(x, y, s, size=13, weight="normal", color=INK, style="normal", va="top"):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=size, fontweight=weight,
            color=color, style=style, va=va, ha="left", wrap=True)

# Title
T(0.06, 0.94, "Vijay", size=22, weight="bold")
T(0.06, 0.885, "Keypoint-guided diffeomorphic registration", size=15, weight="bold", color="#3a3a3a")

# Approach
T(0.06, 0.80, "Approach", size=13, weight="bold")
T(0.06, 0.755,
  "A 3D CNN locates matching anatomical keypoints on the moving and fixed segmentations.\n"
  "Their saliency conditions a U-Net that predicts a stationary velocity field, integrated by\n"
  "scaling-and-squaring so the deformation is invertible by construction. A diffusion-smoothness\n"
  "term on the velocity sets the accuracy–invertibility balance.", size=12)

# Journey / tuning
T(0.06, 0.575, "Behaviour", size=13, weight="bold")
T(0.06, 0.53,
  "• Lighter smoothing keeps the flow sharp → higher accuracy and very low volumetric folding,\n"
  "   but the white-matter surface deforms more (higher self-intersection).\n"
  "• Heavier smoothing keeps velocity gradients low so the discretized Jacobian stays positive\n"
  "   everywhere → folding driven toward zero, at some accuracy cost.", size=12)

# Results box
box = FancyBboxPatch((0.055, 0.06), 0.89, 0.29, boxstyle="round,pad=0.01,rounding_size=0.015",
                     transform=ax.transAxes, facecolor="#ece3cc", edgecolor="#c9bd9c", linewidth=1.2)
ax.add_patch(box)
T(0.075, 0.325, "Results   (neurite-OASIS, 40 train / 10 val)", size=13, weight="bold")
T(0.075, 0.265, "Configuration", size=11.5, weight="bold", color="#555")
T(0.42,  0.265, "WM Dice", size=11.5, weight="bold", color="#555")
T(0.575, 0.265, "Folding (Jacobian) %", size=11.5, weight="bold", color="#555")
T(0.83,  0.265, "Self-intersection %", size=11.5, weight="bold", color="#555")

rows = [
    ("Balanced  (int=12, smooth=3000)", "0.9112", "0.0064", "1.09"),
    ("Fold-free (int=14, smooth=8000)", "0.8458", "0.0008", "0.168"),
]
y = 0.205
for cfg, d, f, s in rows:
    T(0.075, y, cfg, size=12)
    T(0.42,  y, d, size=12, weight="bold")
    T(0.575, y, f, size=12, weight="bold")
    T(0.83,  y, s, size=12, weight="bold")
    y -= 0.058
T(0.075, 0.085, "Inference: ~0.19 s / registration (GPU)  •  diffeomorphic warp by construction",
  size=11, style="italic", color="#555")

out = "/shared/home/v_vijay_bala_mahalingam/vijay_registration_2026-08-07.png"
fig.savefig(out, facecolor=BG, bbox_inches="tight"); print("saved", out)
