#!/usr/bin/env bash
# Self-contained IXI SynthSeg launcher (code stays in this RegNET repo).
#
# Usage:
#   ./scripts/run_ixi_t1_synthseg.sh <input_dir> <output_dir> <synthseg_dir> [gpu_ids] [flags...]
#   SYNTHSEG_DIR=/opt/SynthSeg ./scripts/run_ixi_t1_synthseg.sh <input_dir> <output_dir> [gpu_ids] [flags...]
#
# gpu_ids: one token, e.g. 0 or 5,6

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $0 <input_dir> <output_dir> <synthseg_dir> [gpu_ids] [--vol --skip-existing ...]" >&2
  echo "   or: SYNTHSEG_DIR=/path/to/SynthSeg $0 <input_dir> <output_dir> [gpu_ids] ..." >&2
  exit 1
}

[[ $# -ge 2 ]] || usage

INPUT_DIR="${1}"
OUTPUT_DIR="${2}"
shift 2

SYNTHSEG_DIR_ARG=""
if [[ $# -gt 0 && "${1}" != --* ]]; then
  # If next token is not a GPU id list, treat it as synthseg_dir.
  if [[ ! "${1}" =~ ^[0-9]+([,][0-9]+)*$ ]]; then
    SYNTHSEG_DIR_ARG="${1}"
    shift 1
  fi
fi
if [[ -z "${SYNTHSEG_DIR_ARG}" ]]; then
  SYNTHSEG_DIR_ARG="${SYNTHSEG_DIR:-}"
fi
if [[ -z "${SYNTHSEG_DIR_ARG}" ]]; then
  echo "ERROR: synthseg_dir required (3rd arg or SYNTHSEG_DIR)." >&2
  usage
fi

GPU_ID="0"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  GPU_ID="${1}"
  shift 1
fi
GPU_ID="${GPU_ID// /}"

PY_GPU=(--gpu "${GPU_ID}")
PASS_ARGS=()
skip_next=false
for arg in "$@"; do
  if $skip_next; then
    skip_next=false
    continue
  fi
  if [[ "${arg}" == --gpu ]]; then
    skip_next=true
    PY_GPU=()
    PASS_ARGS+=("${arg}")
    continue
  fi
  PASS_ARGS+=("${arg}")
done

PY="${SYNTHSEG_PYTHON:-python3}"

"${PY}" "${SCRIPT_DIR}/run_ixi_t1_synthseg.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --synthseg-dir "${SYNTHSEG_DIR_ARG}" \
  "${PY_GPU[@]}" \
  "${PASS_ARGS[@]}"
