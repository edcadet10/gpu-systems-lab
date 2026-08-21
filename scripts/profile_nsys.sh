#!/usr/bin/env bash
set -euo pipefail

provider="${GPU_LAB_PROFILE_PROVIDER:-triton}"
case "$provider" in
  cuda_extension|torch|triton) ;;
  *) echo "GPU_LAB_PROFILE_PROVIDER must be cuda_extension, torch, or triton" >&2; exit 2 ;;
esac

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
  --output="artifacts/nsys/residual-rmsnorm-${provider}" \
  python -m gpu_systems_lab.profiling.residual_rmsnorm --provider "$provider"
