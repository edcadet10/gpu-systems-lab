"""Correctness-gated benchmark for residual plus RMSNorm implementations."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from gpu_systems_lab.kernels.residual_rmsnorm import (
    residual_rmsnorm_reference,
    residual_rmsnorm_triton,
)


def _frameworks() -> tuple[Any, Any]:
    try:
        import torch
        import triton
    except ModuleNotFoundError as error:
        raise RuntimeError("install the 'gpu' extra before benchmarking") from error
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-capable GPU is required for this benchmark")
    return torch, triton


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _parse_positive_integers(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",")]
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return parsed


def _time_cuda(torch: Any, function: Callable[[], Any], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _device_metadata(torch: Any, triton: Any) -> dict[str, Any]:
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "device_name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        "total_memory_bytes": properties.total_memory,
        "pytorch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def run_benchmark(
    *,
    rows: Sequence[int],
    hidden_sizes: Sequence[int],
    dtype_name: str,
    providers: Sequence[str],
    warmup: int,
    repeats: int,
    seed: int,
    epsilon: float,
) -> dict[str, Any]:
    """Run correctness first, then collect raw CUDA-event timings."""

    torch, triton = _frameworks()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
    absolute_tolerance = {"float16": 2e-3, "bfloat16": 2e-2, "float32": 2e-5}[dtype_name]
    torch.manual_seed(seed)
    results: list[dict[str, Any]] = []

    for row_count in rows:
        for hidden_size in hidden_sizes:
            input_tensor = torch.randn((row_count, hidden_size), device="cuda", dtype=dtype)
            residual = torch.randn_like(input_tensor)
            weight = torch.randn((hidden_size,), device="cuda", dtype=dtype)

            eager = partial(residual_rmsnorm_reference, input_tensor, residual, weight, epsilon)
            triton_implementation = partial(
                residual_rmsnorm_triton,
                input_tensor,
                residual,
                weight,
                epsilon,
            )

            implementations: dict[str, Callable[[], Any]] = {
                "torch_eager": eager,
                "triton": triton_implementation,
            }
            if "torch_compile" in providers:
                compiled = torch.compile(eager, fullgraph=True)
                implementations["torch_compile"] = compiled

            reference = eager()
            for provider in providers:
                if provider not in implementations:
                    raise ValueError(f"unsupported provider: {provider}")
                candidate = implementations[provider]()
                max_absolute_error = float((candidate.float() - reference.float()).abs().max())
                if max_absolute_error > absolute_tolerance:
                    raise AssertionError(
                        f"{provider} failed correctness for {(row_count, hidden_size)}: "
                        f"{max_absolute_error} > {absolute_tolerance}"
                    )
                samples = _time_cuda(torch, implementations[provider], warmup, repeats)
                results.append(
                    {
                        "shape": [row_count, hidden_size],
                        "dtype": dtype_name,
                        "provider": provider,
                        "correctness": {
                            "max_absolute_error": max_absolute_error,
                            "absolute_tolerance": absolute_tolerance,
                        },
                        "latency_ms": {
                            "samples": samples,
                            "median": statistics.median(samples),
                            "observed_minimum": min(samples),
                            "observed_maximum": max(samples),
                        },
                    }
                )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": _device_metadata(torch, triton),
        "protocol": {
            "timer": "CUDA events",
            "warmup_iterations": warmup,
            "measured_iterations": repeats,
            "seed": seed,
            "epsilon": epsilon,
        },
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=_parse_positive_integers, default=[128, 1024])
    parser.add_argument("--hidden", type=_parse_positive_integers, default=[1024, 4096, 8192])
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument(
        "--providers",
        type=lambda value: [item.strip() for item in value.split(",")],
        default=["torch_eager", "triton"],
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warmup < 1 or args.repeats < 1:
        raise SystemExit("--warmup and --repeats must be positive")
    report = run_benchmark(
        rows=args.rows,
        hidden_sizes=args.hidden,
        dtype_name=args.dtype,
        providers=args.providers,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        epsilon=args.epsilon,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
