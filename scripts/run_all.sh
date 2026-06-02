#!/usr/bin/env bash
#
# Train both models (FNO-S and LiteFNO) across all 8 datasets.
# Data must already be downloaded and preprocessed (done manually).
#
# Usage:
#   scripts/run_all.sh                      # both models, all datasets
#   scripts/run_all.sh --model litefno      # only the LiteFNO model
#   scripts/run_all.sh --gpu 1              # pin all runs to GPU 1
#   scripts/run_all.sh --dataset active_matter gray_scott_reaction_diffusion
#   scripts/run_all.sh --set training.epochs=5 --set training.device=cpu
#
# Any --set overrides are forwarded verbatim to `litefno train`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ALL_DATASETS=(
  acoustic_scattering_discontinuous
  active_matter
  euler_multi_quadrants_openBC
  euler_multi_quadrants_periodicBC
  gray_scott_reaction_diffusion
  rayleigh_benard
  turbulent_radiative_layer_2D
  viscoelastic_instability
)

MODELS=(fno_s litefno)
DATASETS=("${ALL_DATASETS[@]}")
SET_OVERRIDES=()
GPU_ARGS=()

usage() {
  sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

# --- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODELS=("$2"); shift 2 ;;
    --gpu)
      GPU_ARGS=(--gpu "$2"); shift 2 ;;
    --dataset)
      shift
      DATASETS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    --set)
      SET_OVERRIDES+=(--set "$2"); shift 2 ;;
    -h|--help)
      usage 0 ;;
    *)
      echo "Unknown argument: $1" >&2; usage 1 ;;
  esac
done

for m in "${MODELS[@]}"; do
  if [[ "${m}" != "fno_s" && "${m}" != "litefno" ]]; then
    echo "Invalid --model '${m}' (expected: fno_s, litefno)" >&2; exit 1
  fi
done

echo "Repo:      ${REPO_ROOT}"
echo "Models:    ${MODELS[*]}"
echo "Datasets:  ${DATASETS[*]}"
[[ ${#SET_OVERRIDES[@]} -gt 0 ]] && echo "Overrides: ${SET_OVERRIDES[*]}"
echo

# --- train ------------------------------------------------------------------
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    "${SCRIPT_DIR}/run_${model}.sh" "${dataset}" "${GPU_ARGS[@]}" "${SET_OVERRIDES[@]}"
  done
done

echo
echo "All requested runs complete."
