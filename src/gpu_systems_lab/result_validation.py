"""Validation of benchmark reports against package-shipped JSON Schemas."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from gpu_systems_lab.correctness import CORRECTNESS_COMPARISON, CORRECTNESS_TOLERANCES

SCHEMA_FILES = {
    ("residual_rmsnorm", 3): "rmsnorm-v3.schema.json",
    ("residual_rmsnorm", 4): "rmsnorm-v4.schema.json",
    ("residual_rmsnorm", 5): "rmsnorm-v5.schema.json",
    ("allreduce", 3): "allreduce-v3.schema.json",
    ("residual_rmsnorm_suite", 1): "rmsnorm-suite-v1.schema.json",
    ("residual_rmsnorm_suite", 2): "rmsnorm-suite-v2.schema.json",
    ("residual_rmsnorm_suite", 3): "rmsnorm-suite-v3.schema.json",
}

SUITE_CHILD_SCHEMA_VERSIONS = {1: 3, 2: 4, 3: 5}

SUITE_ENVIRONMENT_IDENTITY_FIELDS = (
    "device_index",
    "device_name",
    "compute_capability",
    "total_memory_bytes",
    "python_version",
    "operating_system",
    "kernel_release",
    "machine",
    "pytorch_version",
    "triton_version",
    "cuda_runtime_version",
    "cudnn_version",
    "driver_version",
    "power_limit_watts",
    "cuda_compiler_version",
    "cuda_arch_list",
)


class ReportValidationError(ValueError):
    """A benchmark report could not be parsed or did not match its schema."""


@dataclass(frozen=True)
class ValidationResult:
    benchmark_type: str
    schema_version: int
    schema_file: str
    validated_children: int = 0


@dataclass(frozen=True)
class ValidatedDocument:
    """One report decoded from the exact bytes retained for bundle analysis."""

    relative_path: str
    content: bytes
    report: Any


@dataclass(frozen=True)
class ValidatedResultBundle:
    """A recursively validated, single-read snapshot of a result bundle."""

    validation: ValidationResult
    report: dict[str, Any]
    documents: tuple[ValidatedDocument, ...]


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
            if result["provider"] == "cuda_extension" and result["dtype"] != "float16":
                raise ReportValidationError(
                    f"$.results[{index}].dtype: the CUDA extension provider supports float16 only"
                )
            shapes_by_provider[result["provider"]].add(tuple(result["shape"]))
            observed_dtypes.add(result["dtype"])
            correctness = result["correctness"]
            if report["schema_version"] >= 5:
                relative_tolerance, absolute_tolerance = CORRECTNESS_TOLERANCES[result["dtype"]]
                expected_fields = {
                    "comparison": CORRECTNESS_COMPARISON,
                    "relative_tolerance": relative_tolerance,
                    "absolute_tolerance": absolute_tolerance,
                }
                for field, expected in expected_fields.items():
                    if correctness[field] != expected:
                        raise ReportValidationError(
                            f"$.results[{index}].correctness.{field}: "
                            f"expected {expected!r} for {result['dtype']}, got "
                            f"{correctness[field]!r}"
                        )
                max_absolute_allowance = (
                    absolute_tolerance
                    + relative_tolerance * correctness["max_absolute_error_reference_magnitude"]
                )
                if correctness["max_absolute_error"] > max_absolute_allowance:
                    raise ReportValidationError(
                        f"$.results[{index}].correctness.max_absolute_error: "
                        "maximum-error witness exceeded its scale-aware allowance"
                    )
                if (
                    correctness["max_error_ratio_absolute_error"]
                    > correctness["max_absolute_error"]
                ):
                    raise ReportValidationError(
                        f"$.results[{index}].correctness.max_error_ratio_absolute_error: "
                        "cannot exceed the recorded maximum absolute error"
                    )
                recomputed_ratio = correctness["max_error_ratio_absolute_error"] / (
                    absolute_tolerance
                    + relative_tolerance * correctness["max_error_ratio_reference_magnitude"]
                )
                if not math.isclose(
                    correctness["max_error_ratio"],
                    recomputed_ratio,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ReportValidationError(
                        f"$.results[{index}].correctness.max_error_ratio: "
                        f"expected {recomputed_ratio!r} from its witness, got "
                        f"{correctness['max_error_ratio']!r}"
                    )
                if correctness["max_error_ratio"] > 1:
                    raise ReportValidationError(
                        f"$.results[{index}].correctness.max_error_ratio: "
                        "recorded scale-aware correctness tolerance was exceeded"
                    )
            if (
                report["schema_version"] < 5
                and correctness["max_absolute_error"] > correctness["absolute_tolerance"]
            ):
                raise ReportValidationError(
                    f"$.results[{index}].correctness.max_absolute_error: "
                    "recorded correctness tolerance was exceeded"
                )
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
    elif benchmark_type == "residual_rmsnorm_suite":
        protocol = report["protocol"]
        if "cuda_extension" in protocol["providers"] and any(
            dtype != "float16" for dtype in protocol["dtypes"]
        ):
            raise ReportValidationError(
                "$.protocol.dtypes: suites using the CUDA extension provider support float16 only"
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


def _read_report_document(path: Path, *, relative_path: str) -> ValidatedDocument:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ReportValidationError(f"{path}: {error.strerror or error}") from error
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReportValidationError(f"{path}:{error.start}: report is not valid UTF-8") from error
    try:
        report = json.loads(text, parse_constant=_reject_nonfinite_constant)
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, json.JSONDecodeError):
            location = f":{error.lineno}:{error.colno}"
            message = error.msg
        else:
            location = ""
            message = str(error)
        raise ReportValidationError(f"{path}{location}: invalid JSON: {message}") from error
    return ValidatedDocument(relative_path=relative_path, content=content, report=report)


def load_report_path(path: Path) -> Any:
    """Read strict JSON from disk, rejecting JavaScript-style non-finite numbers."""

    return _read_report_document(path, relative_path=path.name).report


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
    child_schema_version = SUITE_CHILD_SCHEMA_VERSIONS[manifest["schema_version"]]
    expected_identity = {
        "benchmark_type": "residual_rmsnorm",
        "schema_version": child_schema_version,
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


def load_validated_result_bundle_path(path: Path) -> ValidatedResultBundle:
    """Read each document once, recursively validate it, and retain the exact bytes."""

    root_document = _read_report_document(path, relative_path=path.name)
    report = root_document.report
    result = validate_report(report)
    assert isinstance(report, dict)
    if result.benchmark_type != "residual_rmsnorm_suite":
        return ValidatedResultBundle(
            validation=result,
            report=report,
            documents=(root_document,),
        )

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
    child_documents: list[ValidatedDocument] = []
    reference_environment: dict[str, Any] | None = None
    reference_environment_path: str | None = None
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
        normalized_relative_path = child_path.relative_to(suite_root).as_posix()
        child_document = _read_report_document(
            child_path,
            relative_path=normalized_relative_path,
        )
        child = child_document.report
        child_result = validate_report(child)
        expected_child_version = SUITE_CHILD_SCHEMA_VERSIONS[result.schema_version]
        if (
            child_result.benchmark_type != "residual_rmsnorm"
            or child_result.schema_version != expected_child_version
        ):
            raise ReportValidationError(
                f"child report has unsupported suite contract: {record['relative_path']}"
            )
        assert isinstance(child, dict)
        _validate_suite_child(report, record, child)
        environment_identity = {
            field: child["environment"][field]
            for field in SUITE_ENVIRONMENT_IDENTITY_FIELDS
            if field in child["environment"]
        }
        if reference_environment is None:
            reference_environment = environment_identity
            reference_environment_path = normalized_relative_path
        else:
            for field, expected in reference_environment.items():
                observed = environment_identity[field]
                if observed != expected:
                    raise ReportValidationError(
                        f"child environment {field} mismatch: "
                        f"{normalized_relative_path} recorded {observed!r}; "
                        f"{reference_environment_path} recorded {expected!r}"
                    )
        child_documents.append(child_document)

    return ValidatedResultBundle(
        validation=ValidationResult(
            benchmark_type=result.benchmark_type,
            schema_version=result.schema_version,
            schema_file=result.schema_file,
            validated_children=len(seen_paths),
        ),
        report=report,
        documents=(root_document, *child_documents),
    )


def validate_result_bundle_path(path: Path) -> ValidationResult:
    """Validate one report and all recursively referenced suite reports."""

    return load_validated_result_bundle_path(path).validation
