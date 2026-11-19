# HCP dataset access on the cluster

This guide covers the HCP data layout used by the S-RegNET WM/GM (pial)
training workflow. The repository does not contain HCP data or download it.
You need approved access to the dataset and an account on the cluster where
your lab has staged it. Use the cluster hostname and login method provided by
your institution; neither is configured in this repository.

## 1. Find the staged dataset

Connect to the cluster using your institution's instructions, then use a
login or compute environment allowed by its policy. The current
`config_hcp_pial.yaml` refers to:

```bash
/shared/scratch/0/home/v_mahith_reddy/hcp_data
```

If that shared path is available to you, set it as the dataset root:

```bash
export HCP_ROOT=/shared/scratch/0/home/v_mahith_reddy/hcp_data
```

Otherwise, set `HCP_ROOT` to the location provided by your lab or cluster
administrator. Verify the template segmentation is readable and inspect the
surface files present:

```bash
test -r "$HCP_ROOT/100206/T1w/aparc+aseg.nii.gz" \
  && find "$HCP_ROOT/100206/Native_surf" -maxdepth 1 \
       \( -name '*L.pial.native.surf.gii' -o -name '*R.pial.native.surf.gii' \
          -o -name '*L.white.native.surf.gii' -o -name '*R.white.native.surf.gii' \) \
       -print
```

Each subject used for training needs `T1w/aparc+aseg.nii.gz` and at least one
`*pial.native.surf.gii` file under `Native_surf` to be included by the split
generator. The template preparation step needs the left and right pial and
white surfaces, plus the template segmentation:

```text
$HCP_ROOT/
  100206/
    T1w/aparc+aseg.nii.gz
    Native_surf/
      *L.pial.native.surf.gii
      *R.pial.native.surf.gii
      *L.white.native.surf.gii
      *R.white.native.surf.gii
  <other-subject-id>/
    T1w/aparc+aseg.nii.gz
    Native_surf/
      *pial.native.surf.gii
```

If the checks fail, ask your lab or cluster administrator to confirm the
dataset mount and your permissions. Do not copy large datasets into your
home directory; use the approved shared or scratch storage. Keep any access
credentials out of the repository.

## 2. Prepare the template, splits, and segmentations

From the S-RegNET directory in your checkout, activate the environment
provided for the cluster (the project README uses `~/venv/bin/activate`):

```bash
cd /path/to/RegNET/S-RegNET
source ~/venv/bin/activate
```

Build the canonical template from HCP subject `100206`. Store generated files
under the dataset/scratch area rather than in the code checkout:

```bash
export TEMPLATE_DIR="$HCP_ROOT/template_100206"
python3 prepare_hcp_template.py \
  --hcp_dir "$HCP_ROOT/100206" \
  --out_dir "$TEMPLATE_DIR"
```

Create reproducible train/validation lists and convert the listed
segmentations to the 5-class format expected by training:

```bash
python3 create_hcp_splits.py \
  --hcp_root "$HCP_ROOT" \
  --val_count 100 \
  --seed 42

python3 convert_hcp_one_hot.py \
  --hcp_root "$HCP_ROOT" \
  --subjects_file "$HCP_ROOT/val.txt" \
  --workers 4

python3 convert_hcp_one_hot.py \
  --hcp_root "$HCP_ROOT" \
  --subjects_file "$HCP_ROOT/train.txt" \
  --workers 4
```

The split generator excludes `100206` and writes `train.txt` and `val.txt`
under `HCP_ROOT`. The converter reads each subject's
`T1w/aparc+aseg.nii.gz` and writes:

```text
T1w/seg4_white.nii.gz
T1w/seg4_white_onehot.npy
```

The one-hot array has shape `(5, D, H, W)`. Adjust `--workers` to suit the
cluster's CPU and memory allocation. Re-running conversion skips subjects
whose two output files already exist; pass `--overwrite` to regenerate them.

## 3. Configure and run training

`config_hcp_pial.yaml` currently contains paths specific to the original
cluster account. Make a private copy outside the checkout and edit its
absolute paths:

```bash
cp config_hcp_pial.yaml "$HCP_ROOT/config_hcp_pial_local.yaml"
```

In that copy, set these values to match your storage locations:

- `data.train_txt`: `$HCP_ROOT/train.txt`
- `data.val_txt`: `$HCP_ROOT/val.txt`
- `data.template_seg_path`: `$TEMPLATE_DIR/template_pial_seg_onehot.npy`
- `data.template_wm_mesh_path`: `$TEMPLATE_DIR/template_pial_mesh_repaired.npz`
- `output.base_dir`: a writable directory on your scratch allocation

Replace the examples with the fully expanded absolute paths; YAML does not
expand shell variables. Use a GPU assigned to your job for `device.gpu`.
Follow your cluster's job-submission and GPU-allocation rules, and do not run
training on a login node unless the cluster explicitly allows it.

Launch training from the S-RegNET directory:

```bash
python3 train.py --config "$HCP_ROOT/config_hcp_pial_local.yaml"
```

Check `output.base_dir` for the run's checkpoints, logs, and visualizations.
The source config is kept unchanged, so your account-specific paths stay out
of version control.

## 4. Evaluate and visualize a trained HCP model

The evaluation config must match the checkpoint's input resolution. For the
96³ Stage 1 checkpoint, make a private 96³ copy and choose one validation
subject for an initial smoke test:

```bash
RUN_DIR=/shared/scratch/0/home/v_mahith_reddy/training_results/hcp_ft_stage1/<run_timestamp>
OUT_DIR="$RUN_DIR/validation_preview"
mkdir -p "$OUT_DIR"
cp config_hcp_ft.yaml "$OUT_DIR/config_hcp_ft_96.yaml"
sed -i 's/target_size: \[128, 128, 128\]/target_size: [96, 96, 96]/' \
  "$OUT_DIR/config_hcp_ft_96.yaml"
SUBJECT=$(awk -F/ 'NR == 1 {print $(NF-2)}' "$HCP_ROOT/val.txt")
DEVICE=cuda:0  # use the CUDA-visible index assigned to your job
```

Run the HCP metrics evaluator. It reports per-class Dice and voxel-surface
HD95, plus raw pial and white mesh vertex HD95, self-intersecting-face
percentage, and triangle-flip percentage. It writes one CSV and JSON row:

```bash
python3 utils/evaluate_hcp_metrics.py \
  --checkpoint "$RUN_DIR/checkpoints/best_model.pth" \
  --config "$OUT_DIR/config_hcp_ft_96.yaml" \
  --template_dir "$HCP_ROOT/template_100206" \
  --hcp_root "$HCP_ROOT" \
  --subject "$SUBJECT" \
  --output_dir "$OUT_DIR/metrics" \
  --device "$DEVICE"
```

Generate paired WM/pial slice visualizations for the same subject:

```bash
python3 visualize_hcp_wm_gm.py \
  --checkpoint "$RUN_DIR/checkpoints/best_model.pth" \
  --config "$OUT_DIR/config_hcp_ft_96.yaml" \
  --template_dir "$HCP_ROOT/template_100206" \
  --hcp_root "$HCP_ROOT" \
  --subjects "$SUBJECT" \
  --output_dir "$OUT_DIR/visualizations" \
  --device "$DEVICE"
```

The evaluator measures meshes before Laplacian repair; the visualizer also
shows the un-repaired prediction unless `--postprocess` is explicitly passed.
The visualization output is a PNG overlay, while the metrics are stored in the
CSV/JSON files.
