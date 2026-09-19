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

| | Train / Val | WM Dice | Folding % | Triangle-orientation flips % (MC mesh) |
|---|---|---|---|---|
| SVF-E, original | 40 / 10 | 0.9122 | 0.0070 | 0.1434 |
| **SVF-E, full** | **330 / 83** | **0.9111** | **0.0046** | **0.1083** |

The headline Dice is unchanged (0.9122 -> 0.9111) on 6.6x the training data and
8.3x the validation set, so the 0.91 was not an artifact of the small subset.
Both folding and the triangle-flip proxy improved.

The last column is measured by `eval_selfint_mesh.py` on a marching-cubes template
WM mesh in the 128^3 model grid, not on a FreeSurfer `.surf` mesh. It counts
triangle normal reversals; it is not an exact triangle-triangle self-intersection
test.

Reproduce with:

    ./runs/run_svfE_full.sh
    python eval_selfint_mesh.py --val <...>/neurite_oasis/full_val.txt \
                                --runs svf_E_full:"SVF-E FULL" --surfcheck

## The triangle-flip proxy was overstated ~8x by `eval_selfint.py`

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

Corrected triangle-flip percentages:

| Run | WM Dice | `approx` (old) | `fixedpt` | **`svfexact`** |
|-----|---------|--------|---------|----------|
| SVF-E (40/10)    | 0.9122 | 1.1139% | 0.3404% | **0.1434%** |
| SVF-E full (330/83) | 0.9111 | 1.1431% | 0.1625% | **0.1083%** |
| SVF-F (40/10)    | 0.8532 | 0.1565% | 0.0964% | **0.0111%** |

The `approx` column reproduces the old numbers exactly (1.1431% for the full run;
1.1139% vs the 1.0904% in `selfint.log`, which was scored on the mid-run epoch-191
checkpoint), confirming the two scripts share that code path.

`fixedpt` reports more flips than `svfexact` because flipping is a local property:
it drives the residual to approximately zero for most vertices but strands a small
fraction, and each stray vertex affects every incident triangle. `svfexact` avoids
that iterative-solver failure mode by integrating the negated stationary velocity.

## Gap closure — quote this next to triangle flips

Triangle flips alone flatter a heavily regularised model: a deformation that
barely moves is trivially flip-free. `check_push_control.py` measures how much of
the template→sample surface distance the deformation actually removes, on the same
val subjects:

| | mean distance to the sample's WM surface |
|---|---|
| template, undeformed | 1.686 mm |
| **deformed template** (`exp(-vel)`) | **0.951 mm** |
| old first-order push | 1.190 mm |
| deformation reversed (control) | 2.224 mm |

SVF-E closes **43.6 %** of the gap. The reversed-direction control lands at 2.224 mm
— worse than not deforming at all — which confirms the push direction is right.

These corrected distances are computed in scanner-RAS millimetres through each
subject's aligned-volume affine. The earlier values used one scalar for normalized
coordinates even though the 160x192x224 input is resized anisotropically to 128^3.

**Quote `svfexact`.** SVF-E on the full corpus holds WM Dice 0.9111 at 0.108%
orientation-flipped triangles on the marching-cubes mesh —
about 1 triangle in 1000, ~10x better than the 1.14% previously reported. The
Dice-vs-topology trade-off across the sweep survives in ratio (E is ~10x F), but
every absolute number is far lower than `selfint.log` states.

Mesh visualisations: `eval_selfint_mesh.py --figdir <dir>` renders, per subject,
three anatomically-oriented ortho slices (the volumes are LIA, so array axis 0 is
sagittal, axis 1 axial, axis 2 coronal — the naive axial/coronal/sagittal ordering
mislabels all three) with the subject's WM mask in grey, the unpushed template
surface in orange, the old `v - disp(v)` push dashed magenta, and the correct push
in cyan with flipped triangles picked out in red.

## Paired GT / recon-all-clinical audit and real `.surf` meshes

The original clinical summary evaluated 413 available segmentations, mixing the
training and held-out cohorts and including the template while omitting failed
subject `OAS1_0288`. It is not directly comparable with the 83-subject GT row.
The corrected comparison uses the same 82 completed held-out subjects:

| Fixed input | Mesh used for flip metric | WM Dice | Triangle flips | Template -> deformed distance to matching target |
|---|---|---:|---:|---:|
| neurite GT | marching cubes (128^3 model grid) | 0.9111 | 0.1085% | 1.6853 -> **0.9514 mm** (MC target) |
| recon-all-clinical | marching cubes (128^3 model grid) | 0.7224 | 0.5314% | 5.3415 -> **3.8468 mm** (MC target) |
| neurite GT deformation | FreeSurfer `lh/rh.white` (scanner RAS) | 0.9111 | 0.0337% | 4.6325 -> **4.7201 mm** (clinical `.surf` target) |
| recon-all-clinical deformation | FreeSurfer `lh/rh.white` (scanner RAS) | 0.7224 | 0.2845% | 4.6325 -> **4.2360 mm** (clinical `.surf` target) |

The `.surf` rows use the real FreeSurfer template topology (201,817 vertices,
403,626 faces), measure normals in physical scanner-RAS coordinates, and compare
against the subjects' real recon-all-clinical surfaces. There is no
separate neurite-GT `.surf` set in this workspace, so the third row must not be
described as distance to a “GT `.surf`”; its target is still the clinical surface.
That distinction also explains why the GT-driven deformation can improve its own
GT label boundary while not improving the independently generated clinical mesh.

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
| `selfint_mesh.log` | Triangle-orientation-flip diagnostics, SVF-E / SVF-F |
| `selfint_mesh_full.log` | Published marching-cubes result for SVF-E full (0.1083%) |
| `selfint_marching_cubes_{gt,clinical}_paired.log` | Corrected paired 82-subject MC evaluation |
| `selfint_freesurfer_{gt,clinical}_paired.log` | Paired evaluation on real FreeSurfer mesh topology |
| `push_control_physical.log` | Full 83-subject direction control in physical RAS millimetres |
| `selfint_clinical_unpaired_legacy.log` | Superseded mixed-cohort clinical run; retained with an explicit warning |
| `svf_curve.png` | Dice / folding training curves |
| `svf_overlay.png` | Warped-vs-fixed segmentation overlay |
