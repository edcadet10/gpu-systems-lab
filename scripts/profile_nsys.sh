#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys was not found; install Nsight Systems and add it to PATH" >&2
  exit 1
fi

mkdir -p artifacts/nsys
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --output=artifacts/nsys/residual-rmsnorm \
  python -m gpu_systems_lab.profiling.residual_rmsnorm --provider triton
