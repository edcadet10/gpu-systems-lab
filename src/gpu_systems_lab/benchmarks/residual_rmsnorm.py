"""Correctness-gated benchmark for residual plus RMSNorm implementations."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from gpu_systems_lab.benchmark_inputs import (
    choice_list,
    positive_float,
    positive_integer,
    positive_integer_list,
)
from gpu_systems_lab.correctness import CORRECTNESS_COMPARISON, CORRECTNESS_TOLERANCES
from gpu_systems_lab.kernels.residual_rmsnorm import (
    residual_rmsnorm_reference,
    residual_rmsnorm_triton,
)
from gpu_systems_lab.kernels.residual_rmsnorm_cuda import residual_rmsnorm_cuda
from gpu_systems_lab.reporting import (
    git_commit,
    git_dirty,
    nvidia_smi_metadata,
    runtime_metadata,
)

PROVIDERS = ("torch_eager", "cuda_extension", "triton", "torch_compile")


def _frameworks() -> tuple[Any, Any]:
    try:
        import torch
        import triton
    except ModuleNotFoundError as error:
        raise RuntimeError("install the 'gpu' extra before benchmarking") from error
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-capable GPU is required for this benchmark")
    return torch, triton


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


def _time_first_call(
    torch: Any,
    function: Callable[[], Any],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Any, float]:
    """Time first execution with a host clock and device synchronization."""

    torch.cuda.synchronize()
    started = clock()
    result = function()
    torch.cuda.synchronize()
    return result, (clock() - started) * 1000


def _require_correctness(
    provider: str,
    shape: tuple[int, int],
    max_absolute_error: float,
    max_error_ratio: float,
) -> None:
    if (
        not math.isfinite(max_absolute_error)
        or not math.isfinite(max_error_ratio)
        or max_error_ratio > 1
    ):
        raise AssertionError(
            f"{provider} failed correctness for {shape}: "
            f"max_absolute_error={max_absolute_error}, max_error_ratio={max_error_ratio}"
        )


def _correctness_metrics(
    candidate: Any,
    reference: Any,
    dtype_name: str,
) -> dict[str, float | str]:
    """Return scale-aware error metrics using the framework's documented defaults."""

    relative_tolerance, absolute_tolerance = CORRECTNESS_TOLERANCES[dtype_name]
    reference_values = reference.double()
    reference_magnitude = reference_values.abs()
    absolute_error = (candidate.double() - reference_values).abs()
    allowed_error = absolute_tolerance + relative_tolerance * reference_magnitude
    error_ratio = absolute_error / allowed_error
    flat_error = absolute_error.reshape(-1)
    flat_reference_magnitude = reference_magnitude.reshape(-1)
    flat_error_ratio = error_ratio.reshape(-1)
    max_absolute_index = int(flat_error.argmax())
    max_ratio_index = int(flat_error_ratio.argmax())
    return {
        "max_absolute_error": float(flat_error[max_absolute_index]),
        "max_absolute_error_reference_magnitude": float(
            flat_reference_magnitude[max_absolute_index]
        ),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "max_error_ratio": float(flat_error_ratio[max_ratio_index]),
        "max_error_ratio_absolute_error": float(flat_error[max_ratio_index]),
        "max_error_ratio_reference_magnitude": float(flat_reference_magnitude[max_ratio_index]),
        "comparison": CORRECTNESS_COMPARISON,
    }


def _device_metadata(torch: Any, triton: Any) -> dict[str, Any]:
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    compute_capability = list(torch.cuda.get_device_capability(device_index))
    return {
        "device_index": device_index,
        "device_name": properties.name,
        "compute_capability": compute_capability,
        "total_memory_bytes": properties.total_memory,
        "pytorch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        **_cuda_compiler_metadata(torch),
        **runtime_metadata(),
        **nvidia_smi_metadata(device_index),
    }


def _cuda_compiler_metadata(torch: Any) -> dict[str, str | None]:
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except (ImportError, RuntimeError):
        CUDA_HOME = None
    nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc") if CUDA_HOME else shutil.which("nvcc")
    compiler_version = None
    if nvcc:
        try:
            completed = subprocess.run(
                [nvcc, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            compiler_version = lines[-1] if lines else None
    architecture_override = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if architecture_override is None:
        visible_capabilities = {
            tuple(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        }
        architecture_list = ";".join(
            f"{major}.{minor}" for major, minor in sorted(visible_capabilities)
        )
    else:
        architecture_list = architecture_override
    return {
        "cuda_compiler_version": compiler_version,
        "cuda_arch_list": architecture_list or None,
    }


def _build_report(
    *,
    environment: dict[str, Any],
    results: list[dict[str, Any]],
    providers: Sequence[str],
    warmup: int,
    repeats: int,
    seed: int,
    epsilon: float,
    commit: str | None,
    dirty: bool | None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the versioned report independently of GPU execution."""

    return {
        "schema_version": 5,
        "benchmark_type": "residual_rmsnorm",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": environment,
        "protocol": {
            "timer": "CUDA events",
            "first_call_timer": "host wall clock with device synchronization",
            "warmup_iterations": warmup,
            "measured_iterations": repeats,
            "seed": seed,
            "epsilon": epsilon,
            "provider_order": list(providers),
        },
        "results": results,
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

    if not rows or len(rows) != len(set(rows)) or any(value < 1 for value in rows):
        raise ValueError("rows must be a non-empty unique sequence of positive integers")
    if (
        not hidden_sizes
        or len(hidden_sizes) != len(set(hidden_sizes))
        or any(value < 1 for value in hidden_sizes)
    ):
        raise ValueError("hidden_sizes must be a non-empty unique sequence of positive integers")
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not providers or len(providers) != len(set(providers)):
        raise ValueError("providers must be a non-empty unique sequence")
    unknown_providers = [provider for provider in providers if provider not in PROVIDERS]
    if unknown_providers:
        raise ValueError(f"unsupported provider: {unknown_providers[0]}")
    if dtype_name not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype: {dtype_name}")
    if "cuda_extension" in providers and dtype_name != "float16":
        raise ValueError("the CUDA extension provider supports float16 only")

    torch, triton = _frameworks()
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtypes[dtype_name]
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
                "cuda_extension": partial(
                    residual_rmsnorm_cuda,
                    input_tensor,
                    residual,
                    weight,
                    epsilon,
                ),
                "triton": triton_implementation,
            }
            if "torch_compile" in providers:
                compiled = torch.compile(eager, fullgraph=True)
                implementations["torch_compile"] = compiled

            reference, eager_first_call_ms = _time_first_call(torch, eager)
            for provider in providers:
                if provider == "torch_eager":
                    candidate = reference
                    first_call_latency_ms = eager_first_call_ms
                else:
                    candidate, first_call_latency_ms = _time_first_call(
                        torch, implementations[provider]
                    )
                correctness = _correctness_metrics(candidate, reference, dtype_name)
                _require_correctness(
                    provider,
                    (row_count, hidden_size),
                    float(correctness["max_absolute_error"]),
                    float(correctness["max_error_ratio"]),
                )
                samples = _time_cuda(torch, implementations[provider], warmup, repeats)
                results.append(
                    {
                        "shape": [row_count, hidden_size],
                        "dtype": dtype_name,
                        "provider": provider,
                        "correctness": correctness,
                        "first_call_latency_ms": first_call_latency_ms,
                        "latency_ms": {
                            "samples": samples,
                            "median": statistics.median(samples),
                            "observed_minimum": min(samples),
                            "observed_maximum": max(samples),
                        },
                    }
                )

    return _build_report(
        environment=_device_metadata(torch, triton),
        results=results,
        providers=providers,
        warmup=warmup,
        repeats=repeats,
        seed=seed,
        epsilon=epsilon,
        commit=git_commit(),
        dirty=git_dirty(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive_integer_list, default=[128, 1024])
    parser.add_argument("--hidden", type=positive_integer_list, default=[1024, 4096, 8192])
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument(
        "--providers",
        type=lambda value: choice_list(value, allowed=PROVIDERS, label="providers"),
        default=["torch_eager", "triton"],
    )
    parser.add_argument("--warmup", type=positive_integer, default=25)
    parser.add_argument("--repeats", type=positive_integer, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epsilon", type=positive_float, default=1e-6)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
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
        serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
        else:
            print(serialized)
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
