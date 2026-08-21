#!/usr/bin/env bash
set -euo pipefail

provider="${GPU_LAB_PROFILE_PROVIDER:-triton}"
case "$provider" in
  cuda_extension|torch|triton) ;;
  *) echo "GPU_LAB_PROFILE_PROVIDER must be cuda_extension, torch, or triton" >&2; exit 2 ;;
esac

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu was not found; install Nsight Compute and add it to PATH" >&2
  exit 1
fi

mkdir -p artifacts/ncu
ncu \
  --force-overwrite \
  --set roofline \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section LaunchStats \
  --nvtx \
  --nvtx-include "residual_rmsnorm:${provider}/" \
  --export "artifacts/ncu/residual-rmsnorm-${provider}" \
  python -m gpu_systems_lab.profiling.residual_rmsnorm --provider "$provider"
