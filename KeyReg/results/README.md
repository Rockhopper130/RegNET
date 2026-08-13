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
split with a fixed seed. Logs in `svfE_full.log`, self-intersection in
`selfint_full.log`. All three columns come from the same final checkpoint
(epoch 229), not a mid-training one.

| | Train / Val | WM Dice | Folding % | Self-intersection % |
|---|---|---|---|---|
| SVF-E, original | 40 / 10 | 0.9122 | 0.0070 | 1.0904 * |
| **SVF-E, full** | **330 / 83** | **0.9111** | **0.0046** | **1.1431** |

\* measured on the epoch-191 checkpoint, not the finished run — see below.

The headline Dice is unchanged (0.9122 -> 0.9111) on 6.6x the training data and
8.3x the validation set, so the 0.91 was not an artifact of the small subset.
Folding improved; self-intersection is marginally worse.

**Self-intersection remains the weak point.** 1.14% of surface triangles flip
orientation, so near-zero *volumetric* folding (0.0046%) does not deliver a
topologically clean *surface*. SVF-F reaches 0.168% but only by giving up Dice
(0.8532). Nothing in the sweep is simultaneously accurate and flip-free.

Reproduce with:

    ./runs/run_svfE_full.sh
    python eval_selfint.py --val <...>/neurite_oasis/full_val.txt \
                           --runs svf_E_full:"SVF-E FULL"

## Known discrepancy — read before quoting `selfint.log`

`selfint.log` was written at 14:43 on 2026-08-07, while SVF-E and SVF-F were
**still training** (they finished 15:40 and 16:11). It therefore scored whichever
`best.pth` existed mid-run, and states so in its headers:

```
=== SVF-E (Dice 0.910) (epoch 191) ===
=== SVF-F (Dice 0.841) (epoch 104) ===
```

| | `selfint.log` (mid-run) | Final run |
|---|---|---|
| SVF-E | 0.9112 @ epoch 191, fold 0.0064% | **0.9122**, fold 0.0070% |
| SVF-F | 0.8458 @ epoch 104, fold 0.0008% | **0.8532**, fold 0.0006% |

Both configs finished better than `selfint.log` records. The self-intersection
figures in it — 1.0904% (E) and 0.1680% (F) — were measured on those mid-run
checkpoints and have **not** been recomputed against the final models. Re-run
`../eval_selfint.py` against `keyreg_runs/svf_{E,F}/best.pth` before using them;
the checkpoints on disk are the final ones.

## Files

| File | Contents |
|------|----------|
| `svf{A..F}.log` | Full training logs, per-epoch val Dice / folding / per-class |
| `eval_svf.log` | Post-hoc TRUE-folding + inference speed for SVF-A/B (188 ms/registration) |
| `eval_fold.log` | TRUE-folding recompute for the older TPS+flow hybrid models |
| `selfint.log` | Triangle-flip self-intersection — see the discrepancy note above |
| `svf_curve.png` | Dice / folding training curves |
| `svf_overlay.png` | Warped-vs-fixed segmentation overlay |
