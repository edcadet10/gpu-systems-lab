"""Validation of benchmark reports against package-shipped JSON Schemas."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

SCHEMA_FILES = {
    ("residual_rmsnorm", 3): "rmsnorm-v3.schema.json",
    ("allreduce", 3): "allreduce-v3.schema.json",
    ("residual_rmsnorm_suite", 1): "rmsnorm-suite-v1.schema.json",
}


class ReportValidationError(ValueError):
    """A benchmark report could not be parsed or did not match its schema."""


@dataclass(frozen=True)
class ValidationResult:
    benchmark_type: str
    schema_version: int
    schema_file: str
    validated_children: int = 0


def load_schema(schema_file: str) -> dict[str, Any]:
    """Load one schema from the installed package resources."""

    resource = files("gpu_systems_lab.schemas").joinpath(schema_file)
    return json.loads(resource.read_text(encoding="utf-8"))


def _error_path(parts: Any) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _reject_nonfinite_numbers(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReportValidationError(f"{path}: non-finite numbers are not allowed")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite_numbers(child, f"{path}[{index}]")


def _validate_latency_summary(
    latency: dict[str, Any],
    *,
    expected_samples: int,
    path: str,
) -> None:
    samples = latency["samples"]
    if len(samples) != expected_samples:
        raise ReportValidationError(
            f"{path}.samples: expected {expected_samples} samples, got {len(samples)}"
        )
    expected_summary = {
        "median": statistics.median(samples),
        "observed_minimum": min(samples),
        "observed_maximum": max(samples),
    }
    for field, expected in expected_summary.items():
        if not math.isclose(latency[field], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ReportValidationError(
                f"{path}.{field}: expected {expected!r} from raw samples, got {latency[field]!r}"
            )


def _validate_report_semantics(report: dict[str, Any]) -> None:
    benchmark_type = report["benchmark_type"]
    if benchmark_type == "residual_rmsnorm":
        provider_order = report["protocol"]["provider_order"]
        if len(provider_order) != len(set(provider_order)):
            raise ReportValidationError("$.protocol.provider_order: providers must be unique")
        seen_results: set[tuple[int, int, str, str]] = set()
        shapes_by_provider: dict[str, set[tuple[int, int]]] = {
            provider: set() for provider in provider_order
        }
        observed_dtypes: set[str] = set()
        for index, result in enumerate(report["results"]):
            identity = (*result["shape"], result["dtype"], result["provider"])
            if identity in seen_results:
                raise ReportValidationError(f"$.results[{index}]: duplicate result identity")
            seen_results.add(identity)
            if result["provider"] not in provider_order:
                raise ReportValidationError(
                    f"$.results[{index}].provider: missing from protocol provider_order"
                )
            shapes_by_provider[result["provider"]].add(tuple(result["shape"]))
            observed_dtypes.add(result["dtype"])
            _validate_latency_summary(
                result["latency_ms"],
                expected_samples=report["protocol"]["measured_iterations"],
                path=f"$.results[{index}].latency_ms",
            )
        if len(observed_dtypes) != 1:
            raise ReportValidationError("$.results: one child report must contain one dtype")
        shape_sets = list(shapes_by_provider.values())
        if not shape_sets[0] or any(shapes != shape_sets[0] for shapes in shape_sets[1:]):
            raise ReportValidationError(
                "$.results: every protocol provider must cover the same non-empty shape grid"
            )
    elif benchmark_type == "allreduce":
        environment = report["environment"]
        world_size = environment["world_size"]
        ranks = environment["ranks"]
        observed_ranks = [rank["rank"] for rank in ranks]
        if len(ranks) != world_size or sorted(observed_ranks) != list(range(world_size)):
            raise ReportValidationError(
                "$.environment.ranks: expected exactly one record for every global rank"
            )
        result = report["result"]
        if result["correctness_expected"] != float(world_size):
            raise ReportValidationError(
                "$.result.correctness_expected: expected the recorded world size"
            )
        if result["correctness_observed"] != result["correctness_expected"]:
            raise ReportValidationError(
                "$.result.correctness_observed: collective correctness did not pass"
            )
        _validate_latency_summary(
            result["critical_path_latency_ms"],
            expected_samples=report["protocol"]["measured_iterations"],
            path="$.result.critical_path_latency_ms",
        )


def validate_report(report: Any) -> ValidationResult:
    """Validate a decoded report and return the resolved contract identity."""

    _reject_nonfinite_numbers(report)
    if not isinstance(report, dict):
        raise ReportValidationError("$: expected a JSON object")
    benchmark_type = report.get("benchmark_type")
    schema_version = report.get("schema_version")
    if not isinstance(benchmark_type, str):
        raise ReportValidationError("$.benchmark_type: expected a string")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ReportValidationError("$.schema_version: expected an integer")
    schema_file = SCHEMA_FILES.get((benchmark_type, schema_version))
    if schema_file is None:
        supported = ", ".join(f"{name}@{version}" for name, version in sorted(SCHEMA_FILES))
        raise ReportValidationError(
            f"unsupported report contract {benchmark_type}@{schema_version}; supported: {supported}"
        )
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "report validation requires the 'reports' extra; install gpu-systems-lab[reports]"
        ) from error
    validator = Draft202012Validator(load_schema(schema_file), format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(report),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        first = errors[0]
        raise ReportValidationError(f"{_error_path(first.absolute_path)}: {first.message}")
    _validate_report_semantics(report)
    return ValidationResult(
        benchmark_type=benchmark_type,
        schema_version=schema_version,
        schema_file=schema_file,
    )


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value} is not valid report JSON")


def load_report_path(path: Path) -> Any:
    """Read strict JSON from disk, rejecting JavaScript-style non-finite numbers."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ReportValidationError(f"{path}: {error.strerror or error}") from error
    try:
        return json.loads(content, parse_constant=_reject_nonfinite_constant)
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, json.JSONDecodeError):
            location = f":{error.lineno}:{error.colno}"
            message = error.msg
        else:
            location = ""
            message = str(error)
        raise ReportValidationError(f"{path}{location}: invalid JSON: {message}") from error


def validate_report_path(path: Path) -> ValidationResult:
    """Read and validate one JSON report from disk."""

    return validate_report(load_report_path(path))


def _expected_suite_command(manifest: dict[str, Any], record: dict[str, Any]) -> list[str]:
    protocol = manifest["protocol"]
    return [
        "python",
        "-m",
        "gpu_systems_lab.benchmarks.residual_rmsnorm",
        "--rows",
        ",".join(str(value) for value in protocol["rows"]),
        "--hidden",
        ",".join(str(value) for value in protocol["hidden_sizes"]),
        "--dtype",
        record["dtype"],
        "--providers",
        record["provider"],
        "--warmup",
        str(protocol["warmup_iterations"]),
        "--repeats",
        str(protocol["measured_iterations"]),
        "--seed",
        str(protocol["base_seed"] + record["run_index"] - 1),
        "--epsilon",
        str(protocol["epsilon"]),
        "--output",
        record["relative_path"],
    ]


def _validate_suite_child(
    manifest: dict[str, Any],
    record: dict[str, Any],
    child: dict[str, Any],
) -> None:
    expected_identity = {
        "benchmark_type": "residual_rmsnorm",
        "schema_version": 3,
        "git_commit": manifest["git_commit"],
        "git_dirty": manifest["git_dirty"],
    }
    for field, expected in expected_identity.items():
        observed = child.get(field)
        if observed != expected:
            raise ReportValidationError(
                f"child report {field} mismatch: expected {expected!r}, got {observed!r}"
            )

    protocol = manifest["protocol"]
    expected_protocol = {
        "provider_order": [record["provider"]],
        "warmup_iterations": protocol["warmup_iterations"],
        "measured_iterations": protocol["measured_iterations"],
        "seed": protocol["base_seed"] + record["run_index"] - 1,
        "epsilon": protocol["epsilon"],
    }
    child_protocol = child["protocol"]
    for field, expected in expected_protocol.items():
        observed = child_protocol.get(field)
        if observed != expected:
            raise ReportValidationError(
                f"child protocol {field} mismatch: expected {expected!r}, got {observed!r}"
            )

    expected_shapes = {
        (row_count, hidden_size)
        for row_count in protocol["rows"]
        for hidden_size in protocol["hidden_sizes"]
    }
    observed_shapes: list[tuple[int, int]] = []
    for child_result in child["results"]:
        if (
            child_result["provider"] != record["provider"]
            or child_result["dtype"] != record["dtype"]
        ):
            raise ReportValidationError(
                f"child result identity mismatch for {record['dtype']}/{record['provider']}"
            )
        observed_shapes.append(tuple(child_result["shape"]))
    if len(observed_shapes) != len(expected_shapes) or set(observed_shapes) != expected_shapes:
        raise ReportValidationError(
            f"child result shape coverage mismatch: expected {sorted(expected_shapes)}, "
            f"got {sorted(observed_shapes)}"
        )


def validate_result_bundle_path(path: Path) -> ValidationResult:
    """Validate one report and all recursively referenced suite reports."""

    report = load_report_path(path)
    result = validate_report(report)
    if result.benchmark_type != "residual_rmsnorm_suite":
        return result
    assert isinstance(report, dict)

    protocol = report["protocol"]
    expected_records = {
        (run_index, dtype, provider)
        for run_index in range(1, protocol["process_runs"] + 1)
        for dtype in protocol["dtypes"]
        for provider in protocol["providers"]
    }
    observed_records = [
        (record["run_index"], record["dtype"], record["provider"]) for record in report["reports"]
    ]
    if len(observed_records) != len(expected_records) or set(observed_records) != expected_records:
        raise ReportValidationError(
            "$.reports: suite schedule is incomplete or contains duplicate identities"
        )

    suite_root = path.parent.resolve()
    seen_paths: set[Path] = set()
    for record in report["reports"]:
        if record["command"] != _expected_suite_command(report, record):
            raise ReportValidationError(
                f"$.reports: command mismatch for {record['relative_path']}"
            )
        child_path = (suite_root / record["relative_path"]).resolve()
        if child_path.parent != suite_root and suite_root not in child_path.parents:
            raise ReportValidationError(
                f"$.reports: child path escapes suite directory: {record['relative_path']}"
            )
        if child_path in seen_paths:
            raise ReportValidationError(
                f"$.reports: duplicate child path: {record['relative_path']}"
            )
        seen_paths.add(child_path)
        child = load_report_path(child_path)
        child_result = validate_report(child)
        if child_result.benchmark_type != "residual_rmsnorm" or child_result.schema_version != 3:
            raise ReportValidationError(
                f"child report has unsupported suite contract: {record['relative_path']}"
            )
        assert isinstance(child, dict)
        _validate_suite_child(report, record, child)

    return ValidationResult(
        benchmark_type=result.benchmark_type,
        schema_version=result.schema_version,
        schema_file=result.schema_file,
        validated_children=len(seen_paths),
    )
