from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

ROOT = Path(__file__).parents[1]


def _schema(name: str) -> dict[str, object]:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def _validate(schema_name: str, report: dict[str, object]) -> None:
    schema = _schema(schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(report)


def test_rmsnorm_schema_accepts_the_emitted_contract() -> None:
    report = {
        "schema_version": 2,
        "benchmark_type": "residual_rmsnorm",
        "generated_at": "2026-08-20T12:00:00+00:00",
        "git_commit": "a" * 40,
        "environment": {
            "device_name": "Example GPU",
            "compute_capability": [9, 0],
            "total_memory_bytes": 80_000_000_000,
            "pytorch_version": "2.13.0",
            "triton_version": "3.7.1",
            "cuda_runtime_version": "13.0",
            "cudnn_version": 99999,
        },
        "protocol": {
            "timer": "CUDA events",
            "warmup_iterations": 25,
            "measured_iterations": 100,
            "seed": 17,
            "epsilon": 1e-6,
        },
        "results": [
            {
                "shape": [128, 4096],
                "dtype": "float16",
                "provider": "triton",
                "correctness": {
                    "max_absolute_error": 0.001,
                    "absolute_tolerance": 0.002,
                },
                "latency_ms": {
                    "samples": [0.1, 0.11],
                    "median": 0.105,
                    "observed_minimum": 0.1,
                    "observed_maximum": 0.11,
                },
            }
        ],
    }

    _validate("rmsnorm.schema.json", report)


def test_allreduce_schema_accepts_the_emitted_contract() -> None:
    report = {
        "schema_version": 2,
        "benchmark_type": "allreduce",
        "generated_at": "2026-08-20T12:00:00+00:00",
        "git_commit": None,
        "environment": {
            "world_size": 8,
            "device_name": "Example GPU",
            "compute_capability": [9, 0],
            "pytorch_version": "2.13.0",
            "cuda_runtime_version": "13.0",
            "nccl_version": [2, 27, 7],
        },
        "protocol": {
            "timer": "CUDA events",
            "warmup_iterations": 10,
            "measured_iterations": 50,
            "rank_aggregation": "maximum latency for each measured iteration",
        },
        "result": {
            "message_bytes": 134217728,
            "dtype": "float16",
            "correctness_observed": 8.0,
            "correctness_expected": 8.0,
            "critical_path_latency_ms": {
                "samples": [1.0, 1.1],
                "median": 1.05,
                "observed_minimum": 1.0,
                "observed_maximum": 1.1,
            },
        },
    }

    _validate("allreduce.schema.json", report)


@pytest.mark.parametrize("schema_name", ["rmsnorm.schema.json", "allreduce.schema.json"])
def test_result_schemas_reject_incomplete_reports(schema_name: str) -> None:
    with pytest.raises(ValidationError):
        _validate(schema_name, {"schema_version": 2})
