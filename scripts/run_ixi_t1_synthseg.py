#!/usr/bin/env python3
"""
Batch SynthSeg for IXI-T1 (flat folder of *-T1.nii.gz).

Self-contained in this RegNET repo. Does not import sibling workspace code.
Pass --synthseg-dir to a SynthSeg *install* (weights + SynthSeg package),
same idea as FREESURFER_HOME for the recon-all arm.

Examples (from RegNET repo root):
  python scripts/run_ixi_t1_synthseg.py \\
    --input-dir /data/IXI-T1 \\
    --output-dir /data/IXI-T1_synthseg \\
    --synthseg-dir /opt/SynthSeg \\
    --gpu 0

  ./scripts/run_ixi_t1_synthseg.sh /data/IXI-T1 /data/IXI-T1_synthseg /opt/SynthSeg 0 --vol
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def check_environment() -> None:
    """SynthSeg requires NumPy 1.x and TensorFlow 2.x + Keras 2.x."""
    try:
        import numpy as np
    except ImportError:
        sys.exit("ERROR: numpy is not installed.")

    major = int(np.__version__.split(".", 1)[0])
    if major >= 2:
        sys.exit(
            f"ERROR: NumPy {np.__version__} is incompatible with SynthSeg "
            "(need NumPy 1.x in the SynthSeg Python env)."
        )

    try:
        import tensorflow as tf
    except ImportError as e:
        sys.exit(f"ERROR: tensorflow is not installed in this Python.\n  ({e})")

    tf_major = int(tf.__version__.split(".", 1)[0])
    if tf_major != 2:
        sys.exit(
            f"ERROR: tensorflow {tf.__version__} is unsupported (need 2.x)."
        )
    tf_minor = int(tf.__version__.split(".")[1])
    if tf_minor >= 10:
        print(
            f"NOTE: tensorflow {tf.__version__} on Python 3.10+ "
            "(TF 2.2 not always available via pip).",
            file=sys.stderr,
        )

    try:
        import keras
    except ImportError:
        sys.exit("ERROR: keras not found.")

    kv = getattr(keras, "__version__", "")
    if kv:
        kparts = kv.split(".")
        kmajor = int(kparts[0])
        kminor = int(kparts[1]) if len(kparts) > 1 else 0
        if kmajor >= 3:
            sys.exit(f"ERROR: keras {kv} is incompatible (need Keras 2.x).")
        if tf_minor >= 10 and kminor < 10:
            print(
                f"WARNING: tensorflow {tf.__version__} with keras {kv} may be mismatched.",
                file=sys.stderr,
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SynthSeg on all IXI-T1 T1 scans (flat input folder).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Path to IXI-T1 folder (flat *-T1.nii.gz files).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Path for *_synthseg.nii.gz outputs (created if missing).",
    )
    p.add_argument(
        "--synthseg-dir",
        required=True,
        type=Path,
        help="SynthSeg install root (contains SynthSeg/ package, models/, data/). "
        "Also accepted via env SYNTHSEG_DIR.",
    )
    p.add_argument(
        "--gpu",
        default="0",
        help="CUDA_VISIBLE_DEVICES (e.g. 0, 1). Use 'cpu' or --cpu for CPU only.",
    )
    p.add_argument(
        "--pattern",
        default="*-T1.nii.gz",
        help="Glob under input-dir (use '**/norm.nii.gz' for nested subject folders).",
    )
    p.add_argument(
        "--mirror-output",
        action="store_true",
        help="Mirror input subdirs under output-dir.",
    )
    p.add_argument("--robust", action="store_true", help="SynthSeg-robust model.")
    p.add_argument("--parc", action="store_true", help="Cortical parcellation.")
    p.add_argument("--fast", action="store_true", help="Faster predictions.")
    p.add_argument("--v1", action="store_true", help="SynthSeg 1.0 weights.")
    p.add_argument("--cpu", action="store_true", help="Force CPU.")
    p.add_argument("--threads", type=int, default=1, help="TensorFlow threads.")
    p.add_argument("--crop", nargs="+", type=int, default=None, help="Crop size (divisible by 32).")
    p.add_argument("--vol", action="store_true", help="Write volumes.csv in output-dir.")
    p.add_argument("--skip-existing", action="store_true", help="Skip existing outputs.")
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar.")
    return p.parse_args()


def discover_inputs(input_dir: Path, pattern: str) -> list[Path]:
    paths = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' under {input_dir}"
        )
    return paths


def synthseg_output_path(
    input_path: Path, output_dir: Path, input_dir: Path | None = None
) -> Path:
    name = input_path.name
    if input_dir is not None:
        try:
            rel = input_path.resolve().relative_to(input_dir.resolve())
            out = output_dir / rel
            for ext in (".nii.gz", ".nii", ".mgz", ".npz"):
                if out.name.endswith(ext):
                    return out.parent / out.name.replace(ext, f"_synthseg{ext}")
            return out.parent / f"{out.name}_synthseg.nii.gz"
        except ValueError:
            pass
    for ext in (".nii.gz", ".nii", ".mgz", ".npz"):
        if name.endswith(ext):
            return output_dir / name.replace(ext, f"_synthseg{ext}")
    return output_dir / f"{name}_synthseg.nii.gz"


def write_path_lists(
    inputs: list[Path],
    output_dir: Path,
    input_dir: Path | None = None,
) -> tuple[Path, Path]:
    input_list = output_dir / "_synthseg_input_paths.txt"
    output_list = output_dir / "_synthseg_output_paths.txt"
    with input_list.open("w", encoding="utf-8") as fin, output_list.open("w", encoding="utf-8") as fout:
        for p in inputs:
            fin.write(f"{p.resolve()}\n")
            out = synthseg_output_path(p, output_dir, input_dir=input_dir)
            out.parent.mkdir(parents=True, exist_ok=True)
            fout.write(f"{out.resolve()}\n")
    return input_list, output_list


def patch_loopinfo_with_tqdm() -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        print("tqdm not installed; using SynthSeg default progress.", file=sys.stderr)
        return

    import ext.lab2im.utils as lab2im_utils

    class TqdmLoopInfo:
        def __init__(self, n_iterations, spacing=10, text="predicting", print_time=False):
            del spacing, print_time
            self._pbar = tqdm(total=n_iterations, desc=text, unit="scan", dynamic_ncols=True)

        def update(self, idx: int) -> None:
            del idx
            self._pbar.update(1)
            if self._pbar.n >= self._pbar.total:
                self._pbar.close()

    lab2im_utils.LoopInfo = TqdmLoopInfo


def configure_tensorflow(threads: int) -> None:
    import tensorflow as tf

    print(f"TensorFlow: {threads} thread(s)")
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)


def model_weights_path(synthseg_dir: Path, filename: str) -> Path:
    """Resolve a .h5 file under models/, models/SynthSeg_models/, or SYNTHSEG_MODEL_DIR."""
    env_dir = os.environ.get("SYNTHSEG_MODEL_DIR")
    if env_dir:
        p = Path(env_dir).expanduser().resolve() / filename
        if p.is_file():
            return p
        sys.exit(
            f"ERROR: {filename} not found in SYNTHSEG_MODEL_DIR={env_dir}"
        )

    models_root = synthseg_dir / "models"
    for subdir in ("", "SynthSeg_models"):
        candidate = models_root / subdir / filename if subdir else models_root / filename
        if candidate.is_file():
            return candidate
    return models_root / filename


def main() -> None:
    check_environment()
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    synthseg_dir = args.synthseg_dir.expanduser().resolve()

    if not input_dir.is_dir():
        sys.exit(f"ERROR: input directory not found: {input_dir}")
    if not synthseg_dir.is_dir():
        sys.exit(f"ERROR: SynthSeg directory not found: {synthseg_dir}")
    if not (synthseg_dir / "SynthSeg").is_dir():
        sys.exit(
            f"ERROR: {synthseg_dir} does not look like a SynthSeg install "
            "(missing SynthSeg/ package directory)."
        )

    labels_dir = synthseg_dir / "data" / "labels_classes_priors"
    seg_model = "synthseg_robust_2.0.h5" if args.robust else "synthseg_2.0.h5"
    path_model_segmentation = model_weights_path(synthseg_dir, seg_model)

    if not path_model_segmentation.is_file():
        v1_weights = model_weights_path(synthseg_dir, "synthseg_1.0.h5")
        if not args.robust and v1_weights.is_file() and not args.v1:
            print(
                f"WARNING: {seg_model} missing; using SynthSeg 1.0.",
                file=sys.stderr,
            )
            args.v1 = True
            path_model_segmentation = v1_weights
        else:
            models_root = synthseg_dir / "models"
            sys.exit(
                f"ERROR: missing weights: {seg_model}\n"
                f"Looked in: {models_root}, {models_root / 'SynthSeg_models'}\n"
                f"(or set SYNTHSEG_MODEL_DIR to the folder containing the .h5 files)"
            )

    inputs = discover_inputs(input_dir, args.pattern)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu or str(args.gpu).lower() == "cpu":
        print("Device: CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        print(f"Device: GPU CUDA_VISIBLE_DEVICES={args.gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()

    # Runtime dependency: user-supplied SynthSeg install (not a workspace sibling).
    sys.path.insert(0, str(synthseg_dir))
    if not args.no_tqdm:
        patch_loopinfo_with_tqdm()
    configure_tensorflow(args.threads)

    from SynthSeg.predict_synthseg import predict  # noqa: E402

    if args.robust:
        args.fast = True
        assert not args.v1, "--v1 cannot be used with --robust"

    version = "SynthSeg-robust 2.0" if args.robust else ("SynthSeg 1.0" if args.v1 else "SynthSeg 2.0")
    if args.fast and not args.robust:
        version += " (fast)"
    print(f"\n{version}")
    print(f"SynthSeg root: {synthseg_dir}")
    print(f"Weights: {path_model_segmentation}")
    print(f"Input:  {input_dir} ({len(inputs)} scans)")
    print(f"Output: {output_dir}\n")

    labels_segmentation = labels_dir / "synthseg_segmentation_labels_2.0.npy"
    labels_denoiser = labels_dir / "synthseg_denoiser_labels_2.0.npy"
    labels_parcellation = labels_dir / "synthseg_parcellation_labels.npy"
    labels_qc = labels_dir / "synthseg_qc_labels_2.0.npy"
    names_segmentation = labels_dir / "synthseg_segmentation_names_2.0.npy"
    names_parcellation = labels_dir / "synthseg_parcellation_names.npy"
    names_qc = labels_dir / "synthseg_qc_names_2.0.npy"
    topology_classes = labels_dir / "synthseg_topological_classes_2.0.npy"
    n_neutral_labels = 19

    if args.v1:
        path_model_segmentation = model_weights_path(synthseg_dir, "synthseg_1.0.h5")
        labels_segmentation = labels_dir / "synthseg_segmentation_labels.npy"
        labels_qc = labels_dir / "synthseg_qc_labels.npy"
        names_segmentation = labels_dir / "synthseg_segmentation_names.npy"
        names_qc = labels_dir / "synthseg_qc_names.npy"
        topology_classes = labels_dir / "synthseg_topological_classes.npy"
        n_neutral_labels = 18

    mirror_root = input_dir if args.mirror_output else None
    input_list, output_list = write_path_lists(inputs, output_dir, input_dir=mirror_root)
    path_volumes = str(output_dir / "volumes.csv") if args.vol else None

    t0 = time.time()
    predict(
        path_images=str(input_list),
        path_segmentations=str(output_list),
        path_model_segmentation=str(path_model_segmentation),
        labels_segmentation=str(labels_segmentation),
        robust=args.robust,
        fast=args.fast,
        v1=args.v1,
        do_parcellation=args.parc,
        n_neutral_labels=n_neutral_labels,
        names_segmentation=str(names_segmentation),
        labels_denoiser=str(labels_denoiser),
        path_posteriors=None,
        path_resampled=None,
        path_volumes=path_volumes,
        path_model_parcellation=str(
            model_weights_path(synthseg_dir, "synthseg_parc_2.0.h5")
        ),
        labels_parcellation=str(labels_parcellation),
        names_parcellation=str(names_parcellation),
        path_model_qc=str(model_weights_path(synthseg_dir, "synthseg_qc_2.0.h5")),
        labels_qc=str(labels_qc),
        path_qc_scores=None,
        names_qc=str(names_qc),
        cropping=args.crop,
        topology_classes=str(topology_classes),
        ct=False,
        recompute=not args.skip_existing,
    )
    elapsed = time.time() - t0
    if args.mirror_output:
        n_out = sum(1 for _ in output_dir.rglob("*_synthseg.nii.gz"))
    else:
        n_out = len(list(output_dir.glob("*_synthseg.nii.gz")))
    print(f"\nFinished in {elapsed / 60:.1f} min")
    print(f"Segmentations: {output_dir} ({n_out} files)")


if __name__ == "__main__":
    main()
