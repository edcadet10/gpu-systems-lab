#!/usr/bin/env bash
set -euo pipefail

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
  --nvtx-include 'residual_rmsnorm:triton/' \
  --output artifacts/ncu/residual-rmsnorm \
  python -m gpu_systems_lab.profiling.residual_rmsnorm --provider triton
