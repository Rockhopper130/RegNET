# KeyReg — SVF sweep results (neurite-OASIS, 40 train / 10 val)

Raw logs and figures for the keypoint-guided diffeomorphic SVF runs. Every run
was launched by the matching script in `../runs/` and trained by `../keyreg.py`.

## Sweep

Smoothness weight trades WM Dice against folding, monotonically:

| Run | Script | int_steps | smooth_w | delta | fold_w | best WM Dice | folding @ best |
|-----|--------|-----------|----------|-------|--------|--------------|----------------|
| SVF-A | `run_svfA.sh` | 8  | 150  | —    | —  | **0.9652** | 0.5159% |
| SVF-B | `run_svfB.sh` | 10 | 400  | —    | —  | 0.9584 | 0.2369% |
| SVF-C | `run_svfC.sh` | 12 | 600  | 0.30 | 10 | *did not run* | — |
| SVF-D | `run_svfD.sh` | 14 | 1200 | 0.28 | 20 | 0.9392 | 0.0288% |
| SVF-E | `run_svfE.sh` | 12 | 3000 | 0.20 | 40 | 0.9122 | 0.0070% |
| SVF-F | `run_svfF.sh` | 14 | 8000 | 0.18 | 60 | 0.8532 | **0.0006%** |

All 250 epochs except SVF-A/B (300). `svfC.log` contains only its START line —
the job never produced a training step; it is kept so the sweep has no silent gap.

Folding is `folding_pct()` in `keyreg.py`: interior voxels only, strict `det<0`.
This is the corrected metric — it does not boundary-pad, which previously counted
the image shell as folded. The `logged folding (old, boundary-padded)` values in
`eval_fold.log` belong to the earlier hybrid models, which pre-date that fix, and
are not comparable to the table above.

## Full-dataset run (supersedes the 50-subject numbers)

`run_svfE_full.sh` reruns the balanced config on the whole corpus — 330 train /
83 val, with `OASIS_OAS1_0001_MR1` held out as the template (413 = 414 - 1),
split with a fixed seed. Logs in `svfE_full.log`. All three columns come from the
same final checkpoint (epoch 229), not a mid-training one.

| | Train / Val | WM Dice | Folding % | Self-intersection % |
|---|---|---|---|---|
| SVF-E, original | 40 / 10 | 0.9122 | 0.0070 | 0.1434 |
| **SVF-E, full** | **330 / 83** | **0.9111** | **0.0046** | **0.1083** |

The headline Dice is unchanged (0.9122 -> 0.9111) on 6.6x the training data and
8.3x the validation set, so the 0.91 was not an artifact of the small subset.
Both folding and self-intersection improved.

The self-intersection column is measured by `eval_selfint_mesh.py`, not the older
`eval_selfint.py` — see below.

Reproduce with:

    ./runs/run_svfE_full.sh
    python eval_selfint_mesh.py --val <...>/neurite_oasis/full_val.txt \
                                --runs svf_E_full:"SVF-E FULL" --surfcheck

## Self-intersection was overstated ~8x by `eval_selfint.py`

`eval_selfint.py` pushes template-mesh vertices into subject space with

    v_new = v - disp(v)

but `disp` is the *pull* field: `keyreg.py` builds `grid = idg + disp` and samples
`moving[p + disp(p)]`, so the map it encodes is `Phi: subject -> template`,
`Phi(p) = p + disp(p)`. Carrying a *template* vertex the other way needs
`Phi^-1(v)`, the `o` with `o + disp(o) = v`. `v - disp(v)` is only the
first-order (one-step Picard) approximation of that, and under SVF-E-scale
deformations its own error bends triangles and fabricates orientation flips.

`eval_selfint_mesh.py` pushes the mesh three ways and reports all three:

| method | formula | |
|---|---|---|
| `approx` | `v - disp(v)` | what `eval_selfint.py` did |
| `fixedpt` | solve `o + disp(o) = v` | `invert_to_sample`, ported from `S-RegNET/model.py` on the `invertible-deform-SRegNET` branch |
| `svfexact` | `v + disp_inv(v)`, `disp_inv = svf_integrate(-vel)` | exact: for a stationary velocity field `Phi = exp(vel)`, so `Phi^-1 = exp(-vel)` |

Corrected triangle-flip %, and the check that settles which to believe — mean
distance from the pushed mesh to the subject's **own** WM isosurface, a metric
none of the three methods optimises:

| Run | WM Dice | `approx` (old) | `fixedpt` | **`svfexact`** | surf. dist: approx / fixedpt / svfexact |
|-----|---------|--------|---------|----------|------------------|
| SVF-E (40/10)    | 0.9122 | 1.1139% | 0.3404% | **0.1434%** | 1.631 / 1.266 / **1.253** mm |
| SVF-E full (330/83) | 0.9111 | 1.1431% | 0.1625% | **0.1083%** | 1.603 / 1.288 / **1.280** mm |
| SVF-F (40/10)    | 0.8532 | 0.1565% | 0.0964% | **0.0111%** | 1.699 / 1.452 / **1.443** mm |

The `approx` column reproduces the old numbers exactly (1.1431% for the full run;
1.1139% vs the 1.0904% in `selfint.log`, which was scored on the mid-run epoch-191
checkpoint), confirming the two scripts share that code path.

On surface agreement `approx` is clearly the worst map and `fixedpt`/`svfexact`
are equally accurate — so the drop from ~1.14% to ~0.11% is a real correction, not
a different measurement. `fixedpt` still reports 1.5-3x more flips than `svfexact`
because flipping is a *local* property: it drives the residual to ~0 for >95% of
vertices but strands ~1% of them by up to 4.6 voxels, and each stray flips every
triangle touching it. `svfexact` has no strays — `exp(-vel)` is a diffeomorphism
by construction, so its error is smooth and sub-voxel (p50 0.15 vox) and cannot
flip a triangle. Warm-starting the fixed point from `svfexact` confirms the
mechanism: flips fall 0.3404% -> 0.2399% while surface accuracy is unchanged.

## Gap closure — quote this next to self-intersection

Self-intersection alone flatters a heavily-regularised model: a deformation that
barely moves is trivially flip-free. `check_push_control.py` measures how much of
the template→sample surface distance the deformation actually removes, on the same
val subjects:

| | mean distance to the sample's WM surface |
|---|---|
| template, undeformed | 2.290 mm |
| **deformed template** (`exp(-vel)`) | **1.298 mm** |
| old first-order push | 1.646 mm |
| deformation reversed (control) | 2.955 mm |

SVF-E closes **43 %** of the gap. The reversed-direction control lands at 2.955 mm
— worse than not deforming at all — which confirms the push direction is right.

For scale, the subject's own FreeSurfer `lh/rh.white` sits 1.355 mm from the
`seg4` WM boundary, so 1.298 mm is near the label's own noise floor; the remaining
error is gyral detail the smoothing removes, not misalignment.

**Quote `svfexact`.** Self-intersection is no longer the weak point it appeared to
be: SVF-E on the full corpus holds WM Dice 0.9111 at 0.108% flipped triangles —
about 1 triangle in 1000, ~10x better than the 1.14% previously reported. The
Dice-vs-topology trade-off across the sweep survives in ratio (E is ~10x F), but
every absolute number is far lower than `selfint.log` states.

Mesh visualisations: `eval_selfint_mesh.py --figdir <dir>` renders, per subject,
three anatomically-oriented ortho slices (the volumes are LIA, so array axis 0 is
sagittal, axis 1 axial, axis 2 coronal — the naive axial/coronal/sagittal ordering
mislabels all three) with the subject's WM mask in grey, the unpushed template
surface in orange, the old `v - disp(v)` push dashed magenta, and the correct push
in cyan with flipped triangles picked out in red.

## Provenance of `selfint.log` — do not quote it

`selfint.log` was written at 14:43 on 2026-08-07, while SVF-E and SVF-F were
**still training** (they finished 15:40 and 16:11). It scored whichever `best.pth`
existed mid-run, and says so in its headers:

```
=== SVF-E (Dice 0.910) (epoch 191) ===
=== SVF-F (Dice 0.841) (epoch 104) ===
```

| | `selfint.log` (mid-run) | Final run |
|---|---|---|
| SVF-E | 0.9112 @ epoch 191, fold 0.0064% | **0.9122** @ epoch 214, fold 0.0070% |
| SVF-F | 0.8458 @ epoch 104, fold 0.0008% | **0.8532** @ epoch 214, fold 0.0006% |

Both configs finished better than `selfint.log` records. Its self-intersection
figures — 1.0904% (E) and 0.1680% (F) — are wrong on both counts: mid-run
checkpoints *and* the first-order push described above. Superseded by
`selfint_mesh.log` / `selfint_mesh_full.log`.

## Files

| File | Contents |
|------|----------|
| `svf{A..F}.log` | Full training logs, per-epoch val Dice / folding / per-class |
| `eval_svf.log` | Post-hoc TRUE-folding + inference speed for SVF-A/B (188 ms/registration) |
| `eval_fold.log` | TRUE-folding recompute for the older TPS+flow hybrid models |
| `selfint.log` | Superseded — mid-run checkpoints, first-order mesh push |
| `selfint_mesh.log` | **Corrected** triangle-flip self-intersection, SVF-E / SVF-F |
| `selfint_mesh_full.log` | **Corrected** triangle-flip self-intersection, SVF-E full |
| `svf_curve.png` | Dice / folding training curves |
| `svf_overlay.png` | Warped-vs-fixed segmentation overlay |
