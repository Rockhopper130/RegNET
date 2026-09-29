# Self-supervised loss terms: experiment and result

**Verdict: no improvement. Both terms are off by default (`ssl` weights unset = 0),
so this branch reproduces the baseline exactly unless they are switched on.**

## What was added

Two label-free loss terms in `ssl_terms.py`, hooked into `train.py` after the
existing loss. The input contract is unchanged: the network still sees exactly
the genus-0 template and the SynthSeg one-hot, GT is still only the supervision
target, and `get_data.py` is untouched.

- **`mesh_injectivity`** — `det(I + d flow/dp)` evaluated at the 327,684 template
  vertex positions at a quarter-voxel step, on `flow_rv` (the field the
  deliverable push uses). Motivation: `utils/mesh_flip_probe.py` found 0.013%
  voxel folding coexisting with 10.49% cortical triangle flips, because the
  lattice determinant test cannot see non-injectivity between its own samples
  and the pushed mesh carries ~0.8 mm triangles on a 2 mm lattice.
- **`equivariance`** — a random band-limited warp `T` is applied to the
  already-loaded SynthSeg input; `phi(tpl, T.x)` is compared against
  `T.phi(tpl, x)`, the former as a stop-gradient target so the second forward
  pass stores no activations.

## Result

Both runs: `config_meshsup` recipe, 192³, warm-started, 30 epochs,
330 train / 83 val, SynthSeg input, evaluated with `utils/evaluate_all.py
--self_int` over all 83 validation subjects.

| metric | baseline | + SSL terms |
|---|---|---|
| WM Dice | **0.8038** | 0.7986 |
| foreground mean Dice | **0.7863** | 0.7790 |
| surface distance, mean | **1.017 mm** | 1.046 mm |
| surface distance, HD95 | **2.603 mm** | 2.666 mm |
| self-intersecting faces | 0.000338 % | 0.000349 % |
| folding (det < 0) | 0.000023 % | 0.000013 % |

## Why it did not help

**Self-intersection was already solved.** The baseline averages 2.2
self-intersecting faces out of 655,360, i.e. 0.0003%. The injectivity term was
built to remove exactly those, so there was almost nothing left for it to
remove, and it cost ~0.005 WM Dice and ~0.03 mm of surface accuracy to do it.
It did halve voxel folding (0.000023% to 0.000013%), but that was already
negligible.

**The equivariance term was inert.** Its raw value sat at ~2e-5, so at weight 50
it contributed ~0.001 against a Dice term of ~0.39. The weight was too small by
orders of magnitude; the run does not test the idea, only that weight.

## What this rules out, and where the gap is

Topology is not the bottleneck in the current recipe. Self-intersection,
folding, and triangle flips are all at or near zero before any of this. The
remaining gap is **Dice (0.80) and surface distance (1.02 mm mean, 2.60 mm
HD95)**, so that is where the next attempt should aim.

## Caveat on comparing to the README numbers

The 0.804 here is not comparable to the 0.856 in `README.md`. Both are
GT-supervised with SynthSeg input, but that one comes from the full pipeline
whereas these are 30 epochs warm-started from the `res192_v1` checkpoint. The
baseline row exists as the control for the SSL row, not as a reproduction.
