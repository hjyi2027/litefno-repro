#!/usr/bin/env bash
#
# Train the FNO-S baseline on a single dataset.
# Data must already be downloaded and preprocessed.
#
# Usage:
#   scripts/run_fno_s.sh <dataset> [--gpu N] [--set key=value ...]
#
# Example:
#   scripts/run_fno_s.sh gray_scott_reaction_diffusion
#   scripts/run_fno_s.sh gray_scott_reaction_diffusion --gpu 1
#   scripts/run_fno_s.sh active_matter --set training.epochs=5 --set training.device=cpu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 1 || "$1" == -h || "$1" == --help ]]; then
  sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

DATASET="$1"; shift
CONFIG="configs/experiments/fno_s_${DATASET}.yaml"

# Optional: pin to a single GPU. --gpu N masks all other devices.
if [[ "${1:-}" == "--gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="$2"; shift 2
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  echo "Available datasets:" >&2
  ls configs/experiments/ | sed -n 's/^fno_s_\(.*\)\.yaml$/  \1/p' >&2
  exit 1
fi

echo ">>> Training FNO-S on ${DATASET} (${CONFIG})"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo ">>> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
litefno train --config "${CONFIG}" "$@"
