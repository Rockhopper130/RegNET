# KeyReg — keypoint-guided diffeomorphic registration

The current model. See the [root README](../README.md) for headline numbers and
[`results/README.md`](results/README.md) for the full sweep.

## The final model

**SVF-E on the full corpus** — `run_svfE_full.sh`, checkpoint
`keyreg_runs/svf_E_full/best.pth` (epoch 229).
WM Dice **0.9111** · folding **0.0046 %** · self-intersection **0.1083 %**.

```bash
./runs/run_svfE_full.sh
```

The script is chained so it survives a dropped session: it builds the 330/83 split
with a fixed seed, one-hots any missing subjects, then trains. Each stage aborts
the run on failure.

## Layout

| File | Purpose |
|---|---|
| `keyreg.py` | Model, losses, metrics, training loop. `SVFReg` is the final architecture; `KeyReg`/`HybridReg` are the earlier TPS and TPS+flow hybrids. |
| `runs/run_svfE_full.sh` | **The final run.** Others are the sweep (`run_svf{A..F}.sh`) and earlier hybrids. |
| `results/` | Training logs, evaluation logs, figures. See its README. |

## Evaluation

| Script | Measures |
|---|---|
| `eval_selfint_mesh.py` | **Self-intersection** (triangle-flip %) of the warped WM surface, plus mesh figures. Use this one. |
| `check_push_control.py` | **Gap closure** — how much of the template→sample surface distance the deformation actually removes. Quote alongside self-intersection. |
| `eval_fold.py` | Volumetric folding (det(J) < 0) and inference speed. |
| `eval_selfint.py` | **Deprecated** — first-order mesh push, overstates self-intersection ~8×. Kept only to explain the older numbers in `results/selfint.log`. |
| `reg_surf_to_neurite.py` | Rigid-fits FreeSurfer `lh/rh.white` onto the neurite-OASIS frame (needed only for `.surf`-based figures). |

```bash
# self-intersection + mesh figures for the final model
python eval_selfint_mesh.py --runs svf_E_full:"SVF-E FULL" \
    --val <...>/neurite_oasis/full_val.txt --surfcheck \
    --final --figdir figs --n_fig 3

# gap closure (the honesty check on the number above)
python check_push_control.py 10
```

## Why self-intersection needs care

The warp is stored as a **pull** field: `grid = idg + disp`, sampled as
`moving[p + disp(p)]`, so the map runs sample → template. Carrying a *template*
mesh vertex the other way needs the true inverse — the `o` with `o + disp(o) = v`.
`v - disp(v)` is only its first-order approximation, and its own error flips
triangles, which is what inflated the older 1.14 % figure.

For a stationary velocity field the inverse is exact and closed-form: the warp is
`exp(vel)`, so its inverse is `exp(-vel)` — integrate the negated velocity.
`eval_selfint_mesh.py` does this, and reports the first-order and fixed-point
alternatives alongside it for comparison.
