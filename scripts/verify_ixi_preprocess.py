#!/usr/bin/env python3
"""
Verify IXI preprocessing code inside this RegNET repo only.

Does not import or require any sibling workspace repos.
Optional paths (--ixi-t1, --aseg) may point at dataset files anywhere.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
S_REGNET = REPO / "S-RegNET"
SCRIPTS = REPO / "scripts"
SUBJECT_RE = re.compile(r"^IXI\d+.+-T1\.nii\.gz$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK: {msg}")


def assert_no_sibling_imports() -> None:
    """Fail if scripts hard-code sibling workspace repos."""
    forbidden = (
        "SynthSeg-master",
        "CODE 12122025/scripts",
        "parents[2]",
        "runpy.run_path",
    )
    # Only audit the runnable preprocess entrypoints (not this verifier).
    for name in (
        "run_recon_all_clinical_ixi.sh",
        "run_ixi_t1_synthseg.sh",
        "run_ixi_t1_synthseg.py",
    ):
        path = SCRIPTS / name
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                fail(f"{path.name} references external code via {token!r}")
    ok("no sibling-repo imports/paths in preprocess entrypoints")


def check_scripts_present() -> None:
    for name in (
        "run_recon_all_clinical_ixi.sh",
        "run_ixi_t1_synthseg.sh",
        "run_ixi_t1_synthseg.py",
    ):
        if not (SCRIPTS / name).is_file():
            fail(f"missing {SCRIPTS / name}")

    for sh in (
        "run_recon_all_clinical_ixi.sh",
        "run_ixi_t1_synthseg.sh",
    ):
        r = subprocess.run(
            ["bash", "-n", str(SCRIPTS / sh)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            fail(f"bash -n {sh}: {r.stderr}")

    # Syntax-check Python without importing TensorFlow/SynthSeg
    src = (SCRIPTS / "run_ixi_t1_synthseg.py").read_text(encoding="utf-8")
    ast.parse(src)
    ok("preprocess scripts present; bash -n + ast.parse clean")


def check_discovery_unit() -> None:
    sys.path.insert(0, str(SCRIPTS))
    from run_ixi_t1_synthseg import discover_inputs, synthseg_output_path  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "IXI002-Guys-0828-T1.nii.gz").write_bytes(b"")
        (root / "IXI012-HH-1211-T1.nii.gz").write_bytes(b"")
        (root / "readme.txt").write_text("ignore", encoding="utf-8")
        found = discover_inputs(root, "*-T1.nii.gz")
        if len(found) != 2:
            fail(f"discover_inputs expected 2, got {len(found)}")
        for p in found:
            if not SUBJECT_RE.match(p.name):
                fail(f"unexpected name {p.name}")
        out = synthseg_output_path(found[0], root / "out")
        if out.name != "IXI002-Guys-0828-T1_synthseg.nii.gz":
            fail(f"bad synthseg output name: {out.name}")
    ok("discover_inputs + synthseg_output_path unit checks passed")


def check_label_mapping_unit() -> None:
    sys.path.insert(0, str(S_REGNET))
    import numpy as np
    from label_mapping import LABEL_MAPPING, remap_to_5class  # noqa: E402

    vol = np.zeros((4, 4, 4), dtype=np.int64)
    vol[0, 0, 0] = 3   # cortex
    vol[0, 0, 1] = 2   # WM
    vol[0, 0, 2] = 14  # CSF
    vol[0, 0, 3] = 999 # unlisted -> bg
    remapped = remap_to_5class(vol)
    if remapped[0, 0, 0] != 1 or remapped[0, 0, 1] != 3:
        fail(f"unexpected remap: {remapped[0, 0, :2]}")
    if remapped[0, 0, 2] != LABEL_MAPPING[14]:
        fail("CSF remap mismatch")
    if remapped[0, 0, 3] != 0:
        fail("unlisted label should map to background")
    ok("label_mapping.remap_to_5class unit check passed")


def check_optional_data(ixi_t1: Path | None, aseg: Path | None) -> None:
    if ixi_t1 is not None:
        if not ixi_t1.is_dir():
            fail(f"--ixi-t1 not a directory: {ixi_t1}")
        scans = sorted(ixi_t1.glob("*-T1.nii.gz"))
        if not scans:
            fail(f"No *-T1.nii.gz under {ixi_t1}")
        bad = [p.name for p in scans[:50] if not SUBJECT_RE.match(p.name)]
        if bad:
            fail(f"Unexpected T1 names: {bad[:5]}")
        ok(f"optional IXI-T1 discovery: {len(scans)} scans")

    if aseg is not None:
        if not aseg.is_file():
            fail(f"--aseg not a file: {aseg}")
        sys.path.insert(0, str(S_REGNET))
        import nibabel as nib
        import numpy as np
        from label_mapping import remap_to_5class  # noqa: E402

        vol = nib.load(str(aseg)).get_fdata().astype(np.int64)
        remapped = remap_to_5class(vol)
        classes = set(np.unique(remapped).tolist())
        if not classes.issubset({0, 1, 2, 3, 4}):
            fail(f"remap out-of-range: {sorted(classes)}")
        for need in (0, 1, 3):
            if need not in classes:
                fail(f"expected class {need}; got {sorted(classes)}")
        ok(f"optional aseg remap: classes={sorted(classes)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ixi-t1", type=Path, default=None, help="Optional real IXI-T1 dir")
    ap.add_argument("--aseg", type=Path, default=None, help="Optional aseg.mgz to remap")
    args = ap.parse_args()

    print(f"repo={REPO}")
    assert_no_sibling_imports()
    check_scripts_present()
    check_discovery_unit()
    check_label_mapping_unit()
    check_optional_data(args.ixi_t1, args.aseg)
    print("\nAll in-repo IXI preprocess verification checks passed.")


if __name__ == "__main__":
    main()
