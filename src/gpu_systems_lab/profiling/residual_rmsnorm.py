"""Profiler target for the residual-RMSNorm implementations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from gpu_systems_lab.kernels.residual_rmsnorm import (
    residual_rmsnorm_reference,
    residual_rmsnorm_triton,
)
from gpu_systems_lab.kernels.residual_rmsnorm_cuda import residual_rmsnorm_cuda


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("torch", "cuda_extension", "triton"),
        default="triton",
    )
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-capable GPU is required for profiling")
    torch.manual_seed(17)
    input_tensor = torch.randn((args.rows, args.hidden), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(input_tensor)
    weight = torch.randn((args.hidden,), device="cuda", dtype=torch.float16)
    functions = {
        "torch": residual_rmsnorm_reference,
        "cuda_extension": residual_rmsnorm_cuda,
        "triton": residual_rmsnorm_triton,
    }
    function = functions[args.provider]

    for _ in range(5):
        function(input_tensor, residual, weight)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"residual_rmsnorm:{args.provider}")
    for _ in range(args.iterations):
        function(input_tensor, residual, weight)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
