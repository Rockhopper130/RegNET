# RegNET

Deformable registration for brain imaging.

---

## The final model: KeyReg SVF-E (full corpus)

**The current model is [`KeyReg/`](KeyReg/). Everything else has been moved to
[`archive/`](archive/) — see [Project history](#project-history).**

A keypoint-guided **diffeomorphic** registration network. A single full-resolution
stationary velocity field, produced by a U-Net conditioned on the moving/fixed
segmentations and their keypoint saliency, is integrated by scaling-and-squaring,
so the warp is a diffeomorphism by construction.

| | |
|---|---|
| Config | SVF-E — `int_steps 12`, `smooth_w 3000`, `delta 0.20`, `fold_w 40` |
| Data | neurite-OASIS, 330 train / 83 val (`OASIS_OAS1_0001_MR1` held out as template) |
| Checkpoint | `keyreg_runs/svf_E_full/best.pth` (epoch 229) |
| **WM Dice** | **0.9111** |
| **Folding** | **0.0046 %** of voxels with det(J) < 0 |
| **Self-intersection** | **0.1083 %** of surface triangles flipped |

Train and evaluate:

```bash
./KeyReg/runs/run_svfE_full.sh                     # builds the split, then trains

python KeyReg/eval_selfint_mesh.py \
    --runs svf_E_full:"SVF-E FULL" \
    --val <...>/neurite_oasis/full_val.txt --surfcheck
```

Full numbers, the sweep that led here, and the metric definitions:
[`KeyReg/results/README.md`](KeyReg/results/README.md).

### Known limitation — read before quoting the Dice

`smooth_w 3000` buys the near-zero folding and clean surface, but it regularises
hard. Measured against each subject's own WM surface, the deformed template mesh
closes only **43 %** of the template→sample gap (2.290 mm → 1.298 mm):

| | mean distance to the sample's WM surface |
|---|---|
| template, undeformed | 2.290 mm |
| **deformed template** | **1.298 mm** |
| deformation reversed (control) | 2.955 mm |

So the model aligns globally but does not follow individual gyri. WM Dice does not
show this — it is dominated by the interior of the WM, not the boundary — which is
why **self-intersection should always be quoted alongside gap closure**: a
deformation that barely moves is trivially flip-free. Reproduce with
`python KeyReg/check_push_control.py`.

The higher-Dice end of the sweep (SVF-A, WM Dice 0.9652) trades this away: 0.5159 %
folding, ~75× SVF-E, on the 40/10 subset only.

---

## Project history

| Folder | Input | Status |
|---|---|---|
| [`KeyReg/`](KeyReg/) | Segmentation | **Current — this is the model.** Keypoint-guided diffeomorphic SVF. |
| [`archive/S-RegNET/`](archive/S-RegNET/) | Segmentation | Superseded. Seg-only registration, B-spline FFD cascade, genus-0 WM mesh deliverable. |
| [`archive/S-RegNET_exp4/`](archive/S-RegNET_exp4/) | Segmentation | Superseded. Experiment-4 variant of S-RegNET. |
| [`archive/M-RegNET/`](archive/M-RegNET/) | MRI | Superseded. Dense field on MRI, seg-guided attention. |

Each archived pipeline still has its own README, model, losses, training loop and
`config.yaml`, and should still run — only the paths have changed.
