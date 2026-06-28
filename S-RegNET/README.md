# S-RegNET - invertible deformation of a genus-0 template

S-RegNET takes a brain segmentation (e.g. from [SynthSeg](https://github.com/BBillot/SynthSeg))
and produces a white-matter surface that fits the subject **and** is guaranteed to
have the correct topology (a genus-0 surface - one closed sheet, no holes or handles).

The trick is to never edit the noisy input directly. We start from a *template*
whose white-matter surface is already a clean genus-0 mesh, and learn an
**invertible deformation** that morphs the template onto the subject. Because the
deformation cannot tear or fold, the deformed surface keeps the template's
topology no matter how messy the input segmentation is.

```
SynthSeg seg ─┐
              ├─► network ─► bounded B-spline deformation ─► warp template ─► fits subject
template seg ─┘                                                   │
                                                                  └─► push genus-0 template mesh
                                                                      through the inverse deformation
                                                                      ─► genus-0 surface for the subject
```

## How it works

1. **Inputs** - two 5-class one-hot segmentations (template + sample) at `128³`:
   background, cortex, subcortical GM, white matter, CSF.
2. **Network** - a 3D U-Net reads both segs and predicts control-point
   displacements on a coarse lattice. An optional affine stage pre-aligns the
   template first.
3. **Deformation** - a cubic **B-spline free-form deformation (FFD)** turns the
   control points into a dense displacement field (see below).
4. **Training** - the field warps the template seg onto the sample and is scored
   with multi-class Dice (white-matter-weighted) + cross-entropy. The mesh is not
   part of the loss; topology is guaranteed by construction (next section).
5. **Output mesh** - after training, the genus-0 template mesh is pushed into the
   subject's space using the *inverse* of the learned field, giving a genus-0
   white-matter surface for that subject.

Folding %, mesh triangle-flip %, and the mesh-inversion residual are logged every
epoch as sanity diagnostics - all expected to stay near zero.

## The B-spline FFD, and why it stays invertible

A **free-form deformation** lays a coarse grid of *control points* over the image
and moves them around; every voxel in between follows along, interpolated
smoothly. With a **cubic B-spline** as the interpolant you get a deformation field
that is smooth (C²) and depends only on the nearby control points, so a small,
low-dimensional set of parameters can describe a rich warp. This is the classic
Rueckert et al. registration model.

- Free-form deformation: https://en.wikipedia.org/wiki/Free-form_deformation
- B-spline basis functions: https://en.wikipedia.org/wiki/B-spline
- Rueckert et al., *Nonrigid registration using free-form deformations*, IEEE
  Transactions on Medical Imaging, 1999 - the FFD-for-registration reference.

The key property we need is **invertibility** (the deformation is a
diffeomorphism: smooth, one-to-one, smoothly invertible). A B-spline FFD is
invertible as long as the control points don't move too far relative to their
spacing. Choi & Lee worked out exactly how far is safe: if every control-point
displacement stays below a fixed fraction of the control-point spacing, the warp
is guaranteed injective (no folding).

- Choi & Lee, *Injectivity conditions of 2D and 3D uniform cubic B-spline
  functions*, Journal of Mathematical Imaging and Vision, 2000.

We enforce this directly. Each predicted displacement is squashed with

```
cp = delta_max * tanh(cp_raw / delta_max),    delta_max = injectivity_k * cp_spacing * (2/N)
```

where displacements live in normalized `[-1,1]` coordinates over an `N`-voxel axis,
so `cp_spacing * (2/N)` is the control-point spacing in those units. The clamp means
the displacement can never exceed `delta_max`. With `injectivity_k ≤ 0.40` (a fraction of the
Choi-Lee bound), every stage is a diffeomorphism - and composing several stages
(`n_stages`) is still one, which lets the warp reach further without ever folding.
Two consequences:

- The deformed surface keeps the template's genus exactly (the mesh faces never
  change, so its Euler characteristic is fixed).
- The field is invertible, so the template mesh can be pushed into subject space
  by numerically inverting the field (`push_points_to_sample`).

`cp_spacing`, `n_stages`, and `injectivity_k` live in `config.yaml` under
`transform`. Use `cp_spacing` / `n_stages` for more deformation capacity; keep
`injectivity_k ≤ 0.40` for the guarantee.

## Repository layout

Core pipeline (top level):

| File | Role |
|------|------|
| `model.py` | U-Net + `BoundedBSplineFFD` + `SpatialTransformer`; the warp and its inverse |
| `losses.py` | `SegRegistrationLoss` (Dice + CE + bending + affine) and metrics |
| `get_data.py` | dataset: serves template / GT / SynthSeg one-hot segs |
| `train.py` | training loop |
| `config.yaml` | **single source of truth** for paths, hyperparameters, loss weights |
| `inference.py` | run a trained checkpoint on one segmentation |
| `wm_template.py` | build the genus-0 WM template mesh from FreeSurfer `lh/rh.white` |
| `label_mapping.py` | FreeSurfer/SynthSeg integer labels to 5 classes |
| `experiment_routing.py` | picks which seg is the model input vs. the supervision target |
| `convert_one_hot.py`, `convert_synthseg_one_hot.py` | segmentation to 5-channel one-hot `.npy` |
| `overlay_surface.py` | shared surface/coordinate helpers used by the eval + viz tools |

`eval/` - evaluation scripts (surface comparison, self-intersection, SynthSeg-vs-GT).
`utils/` - visualization and one-off dev scripts (mesh figures, smoke test,
transform ceiling probe).

The eval/utils scripts add the parent directory to `sys.path`, so run them from
the `S-RegNET` directory as e.g. `python eval/compare_wm_surface_all.py`.

## Usage

`config.yaml` holds all paths and hyperparameters - edit it rather than passing
flags. Install: `pip install torch numpy nibabel scipy scikit-image matplotlib monai tqdm pyyaml`.

```bash
# Data prep
python convert_one_hot.py                       # GT .nii.gz seg -> 5-channel one-hot .npy
python convert_synthseg_one_hot.py --lists train.txt val.txt \
       --synthseg_dir <dir> --synthseg_name orig_synthseg.nii.gz \
       --out_name synthseg_onehot_v1.npy      # must match data.synthseg_filename
python wm_template.py                            # build the genus-0 WM template mesh

# Train
python train.py                                  # or --config <alt.yaml>

# Inference on one segmentation
python inference.py --input_seg <seg.nii.gz> --checkpoint <best_model.pth> \
       --output_dir <out/> --device cuda:0 [--use_affine]

# Evaluation / visualization
python eval/compare_wm_surface_all.py
python eval/compute_self_intersection.py --which_timestamp <YYYYMMDD_HHMMSS> --num_samples 10
python utils/visualize_mesh_samples.py --checkpoint <best_model.pth> \
       --output_dir images_mesh --num_samples 5 --device cuda:0 [--use_affine]
```

## Notes

- **Warp order is affine first, then the FFD.** Every consumer (training,
  inference, the mesh push) replays that order - inverted, for the mesh push.
- Segmentations are resized with nearest-neighbour interpolation so they stay
  one-hot; `target_size = [128,128,128]` is baked into the model grid.
- Each run stamps its git commit (`git_provenance.py`) and writes to a timestamped
  directory under `output.base_dir`.
