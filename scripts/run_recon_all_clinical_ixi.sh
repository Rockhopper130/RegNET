#!/usr/bin/env bash
set -euo pipefail

# Run recon-all-clinical on the first N IXI-T1 scans (flat *-T1.nii.gz folder).
# Ported from workspace scripts/run_recon_all_clinical_ixi_first50.sh (minimal path edits).
#
# Usage:
#   export FREESURFER_HOME=...
#   source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
#   ./scripts/run_recon_all_clinical_ixi.sh <NIFTI_DIR> [OUTPUT_DIR] [THREADS] [MAX_SUBJECTS]
#
# Example (from this RegNET clone; paths are caller-supplied, no sibling repos):
#   ./scripts/run_recon_all_clinical_ixi.sh \
#     /data/IXI-T1 /data/IXI-T1_recon-all-clinical 2 50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NIFTI_DIR="${1:?Usage: $0 <nifti_dir> [output_dir] [threads] [max_subjects]}"
OUTPUT_DIR="${2:-${REPO_ROOT}/data/IXI-T1_recon-all-clinical}"
THREADS="${3:-2}"
MAX_SUBJECTS="${4:-50}"

NIFTI_DIR="$(cd "$NIFTI_DIR" && pwd)"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

if [[ -z "${FREESURFER_HOME:-}" ]]; then
  echo "ERROR: FREESURFER_HOME is not set."
  echo "Example:"
  echo "  export FREESURFER_HOME=/shared/scratch/0/home/v_soham_lahiri/freesurfer"
  echo "  source \"\$FREESURFER_HOME/SetUpFreeSurfer.sh\""
  exit 1
fi

if ! command -v recon-all-clinical.sh >/dev/null 2>&1; then
  echo "ERROR: recon-all-clinical.sh not found in PATH."
  exit 1
fi

resolve_fs_license() {
  local candidate
  local -a candidates=()

  if [[ -n "${FS_LICENSE:-}" ]]; then
    candidates+=("$FS_LICENSE")
  fi
  candidates+=(
    "$FREESURFER_HOME/.license"
    "$(pwd)/license.txt"
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/license.txt"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      export FS_LICENSE="$candidate"
      return 0
    fi
  done

  echo "ERROR: FreeSurfer license not found."
  echo "Set FS_LICENSE or place license.txt in the project root."
  echo "Expected default: $FREESURFER_HOME/.license"
  exit 1
}

resolve_fs_license

export FS_ALLOW_DEEP=1
export SINGULARITYENV_FS_ALLOW_DEEP=1
export APPTAINERENV_FS_ALLOW_DEEP=1

echo "FREESURFER_HOME=$FREESURFER_HOME"
echo "FS_LICENSE=$FS_LICENSE"
echo "NIFTI_DIR=$NIFTI_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "THREADS=$THREADS"
echo "MAX_SUBJECTS=$MAX_SUBJECTS"
echo

if ! mri_synthseg --help >/tmp/mri_synthseg_help.log 2>&1; then
  echo "ERROR: mri_synthseg failed preflight."
  cat /tmp/mri_synthseg_help.log
  exit 1
fi

LINK_ROOT="/tmp/freesurfer_clinical_links_$$"
mkdir -p "$LINK_ROOT"
ln -sfn "$OUTPUT_DIR" "$LINK_ROOT/subjects_out"

# Portable discovery (bash 3.2+; avoids mapfile from bash 4+)
T1_LIST="$(mktemp)"
trap 'rm -rf "$LINK_ROOT"; rm -f "$T1_LIST"' EXIT
find "$NIFTI_DIR" -maxdepth 1 -type f -name '*-T1.nii.gz' -print \
  | sort \
  | head -n "$MAX_SUBJECTS" > "$T1_LIST"

if [[ ! -s "$T1_LIST" ]]; then
  echo "ERROR: No *-T1.nii.gz files found in $NIFTI_DIR"
  exit 1
fi

N_SCANS="$(wc -l < "$T1_LIST" | tr -d ' ')"
echo "Found ${N_SCANS} scan(s) to process"
echo

PROCESSED=0
SKIPPED=0
FAILED=0

while IFS= read -r T1W; do
  [[ -n "$T1W" ]] || continue
  SUBJECT_ID="$(basename "$T1W" .nii.gz)"
  NORM_FILE="$OUTPUT_DIR/$SUBJECT_ID/mri/norm.mgz"

  if [[ -f "$NORM_FILE" ]]; then
    echo "SKIP $SUBJECT_ID (already done: $NORM_FILE)"
    ((SKIPPED++)) || true
    continue
  fi

  echo "=========================================="
  echo "Processing $SUBJECT_ID"
  echo "  Input:  $T1W"
  echo "  Output: $OUTPUT_DIR/$SUBJECT_ID"
  echo "=========================================="

  ln -sfn "$T1W" "$LINK_ROOT/${SUBJECT_ID}_input.nii.gz"
  rm -f "$OUTPUT_DIR/$SUBJECT_ID/scripts/IsRunning.lh+rh"

  if recon-all-clinical.sh \
      "$LINK_ROOT/${SUBJECT_ID}_input.nii.gz" \
      "$SUBJECT_ID" \
      "$THREADS" \
      "$LINK_ROOT/subjects_out"; then
    echo "OK   $SUBJECT_ID"
    ((PROCESSED++)) || true
  else
    echo "FAIL $SUBJECT_ID"
    ((FAILED++)) || true
  fi

  echo
done < "$T1_LIST"

echo "=========================================="
echo "Summary"
echo "  Processed: $PROCESSED"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"
echo "  Output:    $OUTPUT_DIR"
echo "=========================================="
