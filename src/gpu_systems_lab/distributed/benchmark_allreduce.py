"""Correctness-gated multi-process NCCL all-reduce benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_systems_lab.benchmark_inputs import element_count, positive_integer
from gpu_systems_lab.reporting import (
    git_commit,
    git_dirty,
    nvidia_smi_metadata,
    runtime_metadata,
)


def _rank_metadata(torch: Any, rank: int, local_rank: int) -> dict[str, Any]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "device_index": index,
        "device_name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "total_memory_bytes": properties.total_memory,
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "nccl_version": list(torch.cuda.nccl.version()),
        **runtime_metadata(),
        **nvidia_smi_metadata(index),
    }


def _allreduce_once(distributed: Any, tensor: Any) -> None:
    tensor.fill_(1)
    distributed.all_reduce(tensor)


def _tensor_extrema(torch: Any, tensor: Any) -> tuple[float, float]:
    """Return extrema so correctness covers every element without a full host copy."""

    observed_min, observed_max = torch.aminmax(tensor)
    return float(observed_min), float(observed_max)


def _assert_collective_correctness(
    torch: Any,
    distributed: Any,
    *,
    observed_min: float,
    observed_max: float,
    expected: float,
    device: Any,
    rank: int,
) -> None:
    """Make every rank agree that correctness passed before any rank raises."""

    local_ok = observed_min == expected and observed_max == expected
    all_ranks_ok = torch.tensor(
        1 if local_ok else 0,
        dtype=torch.int32,
        device=device,
    )
    distributed.all_reduce(all_ranks_ok, op=distributed.ReduceOp.MIN)
    if int(all_ranks_ok.item()) == 1:
        return
    if local_ok:
        detail = f"rank {rank} matched locally, but at least one peer rank failed"
    else:
        detail = (
            f"rank {rank} observed range [{observed_min}, {observed_max}], "
            f"expected every element to equal {expected}"
        )
    raise AssertionError(f"all-reduce correctness failed: {detail}")


def _build_report(
    *,
    rank_environments: list[dict[str, Any]],
    world_size: int,
    message_bytes: int,
    dtype_name: str,
    observed: float,
    expected: float,
    critical_path_samples: list[float],
    warmup: int,
    repeats: int,
    commit: str | None,
    dirty: bool | None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the versioned report independently of distributed execution."""

    return {
        "schema_version": 3,
        "benchmark_type": "allreduce",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": {
            "world_size": world_size,
            "ranks": rank_environments,
        },
        "protocol": {
            "timer": "CUDA events",
            "warmup_iterations": warmup,
            "measured_iterations": repeats,
            "rank_aggregation": "maximum latency for each measured iteration",
            "correctness_consensus": "all-rank integer minimum",
        },
        "result": {
            "message_bytes": message_bytes,
            "dtype": dtype_name,
            "correctness_observed": observed,
            "correctness_expected": expected,
            "critical_path_latency_ms": {
                "samples": critical_path_samples,
                "median": statistics.median(critical_path_samples),
                "observed_minimum": min(critical_path_samples),
                "observed_maximum": max(critical_path_samples),
            },
        },
    }


def run(
    *,
    message_bytes: int,
    dtype_name: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any] | None:
    """Run under ``torchrun`` and return a report on rank zero."""

    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    if dtype_name not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype: {dtype_name}")

    try:
        import torch
        import torch.distributed as distributed
    except ModuleNotFoundError as error:
        raise RuntimeError("install the 'torch' extra before benchmarking collectives") from error

    if not torch.cuda.is_available() or not distributed.is_nccl_available():
        raise RuntimeError("CUDA and the NCCL backend are required")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch with torchrun so LOCAL_RANK is defined")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend="nccl")
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    if world_size < 2:
        distributed.destroy_process_group()
        raise RuntimeError("the collective benchmark requires at least two ranks")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype_name
    ]
    element_size = torch.empty((), dtype=dtype).element_size()
    try:
        number_of_elements = element_count(message_bytes, element_size)
    except ValueError:
        distributed.destroy_process_group()
        raise

    tensor = torch.ones(number_of_elements, device="cuda", dtype=dtype)
    try:
        for _ in range(warmup):
            _allreduce_once(distributed, tensor)
        torch.cuda.synchronize()

        expected = float(world_size)
        _allreduce_once(distributed, tensor)
        torch.cuda.synchronize()
        observed_min, observed_max = _tensor_extrema(torch, tensor)
        _assert_collective_correctness(
            torch,
            distributed,
            observed_min=observed_min,
            observed_max=observed_max,
            expected=expected,
            device=tensor.device,
            rank=rank,
        )

        local_samples: list[float] = []
        for _ in range(repeats):
            tensor.fill_(1)
            torch.cuda.synchronize()
            distributed.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            distributed.all_reduce(tensor)
            end.record()
            end.synchronize()
            local_samples.append(float(start.elapsed_time(end)))

        local_payload = {
            "samples": local_samples,
            "environment": _rank_metadata(torch, rank, local_rank),
        }
        gathered: list[dict[str, Any] | None] | None = (
            [None for _ in range(world_size)] if rank == 0 else None
        )
        distributed.gather_object(local_payload, gathered, dst=0)
        if rank != 0:
            return None
        assert gathered is not None
        rank_payloads = [payload for payload in gathered if payload is not None]
        if len(rank_payloads) != world_size:
            raise RuntimeError("rank metadata gather returned an incomplete world")
        rank_samples = [payload["samples"] for payload in rank_payloads]
        rank_environments = sorted(
            (payload["environment"] for payload in rank_payloads),
            key=lambda environment: environment["rank"],
        )
        critical_path_samples = [max(values) for values in zip(*rank_samples, strict=True)]
        return _build_report(
            rank_environments=rank_environments,
            world_size=world_size,
            message_bytes=message_bytes,
            dtype_name=dtype_name,
            observed=observed_min,
            expected=expected,
            critical_path_samples=critical_path_samples,
            warmup=warmup,
            repeats=repeats,
            commit=git_commit(),
            dirty=git_dirty(),
        )
    finally:
        distributed.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message-bytes", type=positive_integer, default=128 * 1024 * 1024)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--warmup", type=positive_integer, default=10)
    parser.add_argument("--repeats", type=positive_integer, default=50)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(
            message_bytes=args.message_bytes,
            dtype_name=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        if report is None:
            return 0
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
