"""Render the paired GT / recon-all-clinical audit as a shareable PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BG = "#07071a"
GRID = "#45445f"
TEXT = "#e7e7ee"
ACCENT = "#cfd3ff"

fig = plt.figure(figsize=(13.5, 6.4), dpi=170, facecolor=BG)
ax = fig.add_axes([0.03, 0.08, 0.94, 0.84])
ax.set_facecolor(BG)
ax.axis("off")

fig.text(0.5, 0.95, "SVF-E full — paired held-out audit (82 subjects)",
         ha="center", va="top", color=TEXT, fontsize=18, fontweight="bold")

rows = [
    ["WM Dice", "0.9111", "0.7224"],
    ["Triangle orientation flips (MC mesh; 128³ model grid)", "0.1085%", "0.5314%"],
    ["Distance to matching marching-cubes WM surface", "0.9514 mm", "3.8468 mm"],
]
table = ax.table(cellText=rows,
                 colLabels=["Primary paired metrics", "neurite GT", "recon-all-clinical"],
                 cellLoc="left", colLoc="left", bbox=[0.0, 0.53, 1.0, 0.36],
                 colWidths=[0.58, 0.20, 0.22])

rows_surf = [
    ["Triangle orientation flips (.surf; physical scanner RAS)", "0.0337%", "0.2845%"],
    ["Distance to clinical sample lh/rh.white: template → deformed",
     "4.6325 → 4.7201 mm", "4.6325 → 4.2360 mm"],
]
table_surf = ax.table(cellText=rows_surf,
                      colLabels=["Real FreeSurfer .surf audit", "GT-driven field",
                                 "clinical-driven field"],
                      cellLoc="left", colLoc="left", bbox=[0.0, 0.19, 1.0, 0.27],
                      colWidths=[0.58, 0.20, 0.22])

for tbl in (table, table_surf):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (row, _), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.9)
        cell.set_facecolor(BG)
        cell.get_text().set_color(ACCENT if row == 0 else TEXT)
        if row == 0:
            cell.get_text().set_fontweight("bold")

fig.text(0.04, 0.10,
         "0.1083% is the original 83-subject GT value on the marching-cubes template "
         "in the 128³ model grid; "
         "0.1085% is the same metric on the 82 paired completed cases.\n"
         "Triangle orientation flips are a normal-reversal proxy, not an exact "
         "triangle–triangle self-intersection test. The available .surf targets are "
         "recon-all-clinical outputs.",
         color="#b8b8c9", fontsize=9.5, va="bottom")

out = "/shared/home/v_vijay_bala_mahalingam/paired_gt_clinical_audit.png"
fig.savefig(out, facecolor=BG, bbox_inches="tight")
print("saved", out)
