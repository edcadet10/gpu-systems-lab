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


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _metadata(torch: Any, world_size: int) -> dict[str, Any]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "world_size": world_size,
        "device_name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "nccl_version": list(torch.cuda.nccl.version()),
    }


def _allreduce_once(distributed: Any, tensor: Any) -> None:
    tensor.fill_(1)
    distributed.all_reduce(tensor)


def run(
    *,
    message_bytes: int,
    dtype_name: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any] | None:
    """Run under ``torchrun`` and return a report on rank zero."""

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
    element_count = message_bytes // element_size
    if element_count == 0 or element_count * element_size != message_bytes:
        distributed.destroy_process_group()
        raise ValueError("message bytes must be a positive multiple of the dtype size")

    tensor = torch.ones(element_count, device="cuda", dtype=dtype)
    try:
        for _ in range(warmup):
            _allreduce_once(distributed, tensor)
        torch.cuda.synchronize()

        expected = float(world_size)
        _allreduce_once(distributed, tensor)
        torch.cuda.synchronize()
        observed = float(tensor[0])
        if observed != expected:
            raise AssertionError(
                f"all-reduce correctness failed: observed {observed}, expected {expected}"
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

        gathered: list[list[float] | None] | None = (
            [None for _ in range(world_size)] if rank == 0 else None
        )
        distributed.gather_object(local_samples, gathered, dst=0)
        if rank != 0:
            return None
        assert gathered is not None
        rank_samples = [sample for sample in gathered if sample is not None]
        critical_path_samples = [max(values) for values in zip(*rank_samples, strict=True)]
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment": _metadata(torch, world_size),
            "protocol": {
                "timer": "CUDA events",
                "warmup_iterations": warmup,
                "measured_iterations": repeats,
                "rank_aggregation": "maximum latency for each measured iteration",
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
    finally:
        distributed.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message-bytes", type=_positive_integer, default=128 * 1024 * 1024)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--warmup", type=_positive_integer, default=10)
    parser.add_argument("--repeats", type=_positive_integer, default=50)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(
        message_bytes=args.message_bytes,
        dtype_name=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    if report is None:
        return 0
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
