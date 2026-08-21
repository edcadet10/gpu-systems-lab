from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from gpu_systems_lab.benchmarks.residual_rmsnorm import _build_report as build_rmsnorm_report
from gpu_systems_lab.distributed.benchmark_allreduce import _build_report as build_allreduce_report
from gpu_systems_lab.result_validation import (
    SCHEMA_FILES,
    ReportValidationError,
    load_schema,
    validate_report,
    validate_report_path,
    validate_result_bundle_path,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _write_bundle(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.json"
    child_path = tmp_path / "run-01" / "float16" / "triton.json"
    child_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        (FIXTURES / "valid-rmsnorm-suite-v1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    child_path.write_text(
        (FIXTURES / "valid-rmsnorm-v3.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return manifest_path


def _write_current_bundle(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.json"
    child_path = tmp_path / "run-01" / "float16" / "cuda_extension.json"
    child_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        (FIXTURES / "valid-rmsnorm-suite-v3.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    child_path.write_text(
        (FIXTURES / "valid-rmsnorm-v5.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return manifest_path


@pytest.mark.parametrize(
    ("fixture_name", "benchmark_type", "schema_version"),
    [
        ("valid-rmsnorm-v3.json", "residual_rmsnorm", 3),
        ("valid-rmsnorm-v4.json", "residual_rmsnorm", 4),
        ("valid-rmsnorm-v5.json", "residual_rmsnorm", 5),
        ("valid-allreduce-v3.json", "allreduce", 3),
        ("valid-rmsnorm-suite-v1.json", "residual_rmsnorm_suite", 1),
        ("valid-rmsnorm-suite-v2.json", "residual_rmsnorm_suite", 2),
        ("valid-rmsnorm-suite-v3.json", "residual_rmsnorm_suite", 3),
    ],
)
def test_packaged_schema_accepts_complete_report(
    fixture_name: str,
    benchmark_type: str,
    schema_version: int,
) -> None:
    result = validate_report(_fixture(fixture_name))
    assert result.benchmark_type == benchmark_type
    assert result.schema_version == schema_version


def test_current_bundle_contracts_validate_together(tmp_path: Path) -> None:
    validation = validate_result_bundle_path(_write_current_bundle(tmp_path))

    assert validation.schema_version == 3
    assert validation.validated_children == 1


def test_rmsnorm_producer_emits_current_contract() -> None:
    fixture = _fixture("valid-rmsnorm-v5.json")
    report = build_rmsnorm_report(
        environment=fixture["environment"],  # type: ignore[arg-type]
        results=fixture["results"],  # type: ignore[arg-type]
        providers=["cuda_extension"],
        warmup=25,
        repeats=2,
        seed=17,
        epsilon=1e-6,
        commit="a" * 40,
        dirty=False,
        generated_at="2026-08-20T12:00:00+00:00",
    )

    assert validate_report(report).schema_file == "rmsnorm-v5.schema.json"


def test_allreduce_producer_emits_current_contract() -> None:
    fixture = _fixture("valid-allreduce-v3.json")
    result = fixture["result"]
    assert isinstance(result, dict)
    latency = result["critical_path_latency_ms"]
    assert isinstance(latency, dict)
    environment = fixture["environment"]
    assert isinstance(environment, dict)
    report = build_allreduce_report(
        rank_environments=environment["ranks"],  # type: ignore[arg-type]
        world_size=2,
        message_bytes=134217728,
        dtype_name="float16",
        observed=2.0,
        expected=2.0,
        critical_path_samples=latency["samples"],  # type: ignore[arg-type]
        warmup=10,
        repeats=2,
        commit="a" * 40,
        dirty=False,
        generated_at="2026-08-20T12:00:00+00:00",
    )

    assert validate_report(report).schema_file == "allreduce-v3.schema.json"


@pytest.mark.parametrize("schema_file", SCHEMA_FILES.values())
def test_packaged_schemas_are_valid_draft_2020_12(schema_file: str) -> None:
    Draft202012Validator.check_schema(load_schema(schema_file))


def test_validation_reports_precise_field_path() -> None:
    report = _fixture("valid-rmsnorm-v3.json")
    del report["environment"]["driver_version"]  # type: ignore[index]

    with pytest.raises(ReportValidationError, match=r"\$\.environment:.*driver_version"):
        validate_report(report)


def test_validation_recomputes_latency_summary() -> None:
    report = _fixture("valid-rmsnorm-v3.json")
    report["results"][0]["latency_ms"]["median"] = 99.0  # type: ignore[index]

    with pytest.raises(ReportValidationError, match=r"expected 0\.105"):
        validate_report(report)


def test_validation_rejects_legacy_recorded_correctness_failure() -> None:
    report = _fixture("valid-rmsnorm-v3.json")
    report["results"][0]["correctness"]["max_absolute_error"] = 0.0021  # type: ignore[index]

    with pytest.raises(ReportValidationError, match="correctness tolerance was exceeded"):
        validate_report(report)


def test_validation_rejects_scale_aware_correctness_failure() -> None:
    report = _fixture("valid-rmsnorm-v5.json")
    correctness = report["results"][0]["correctness"]  # type: ignore[index]
    correctness["max_absolute_error"] = 0.002010201  # type: ignore[index]
    correctness["max_absolute_error_reference_magnitude"] = 1000.0  # type: ignore[index]
    correctness["max_error_ratio"] = 1.0001  # type: ignore[index]
    correctness["max_error_ratio_absolute_error"] = 0.002010201  # type: ignore[index]

    with pytest.raises(ReportValidationError, match="scale-aware correctness tolerance"):
        validate_report(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [("relative_tolerance", 1.0), ("absolute_tolerance", 1.0)],
)
def test_validation_rejects_modified_scale_aware_tolerance(field: str, value: float) -> None:
    report = _fixture("valid-rmsnorm-v5.json")
    report["results"][0]["correctness"][field] = value  # type: ignore[index]

    with pytest.raises(ReportValidationError, match=f"correctness.{field}"):
        validate_report(report)


def test_validation_rejects_inconsistent_scale_aware_ratio_witness() -> None:
    report = _fixture("valid-rmsnorm-v5.json")
    report["results"][0]["correctness"]["max_error_ratio"] = 0.25  # type: ignore[index]

    with pytest.raises(ReportValidationError, match=r"expected .* from its witness"):
        validate_report(report)


def test_validation_rejects_failing_maximum_absolute_error_witness() -> None:
    report = _fixture("valid-rmsnorm-v5.json")
    correctness = report["results"][0]["correctness"]  # type: ignore[index]
    correctness["max_absolute_error"] = 0.1  # type: ignore[index]
    correctness["max_absolute_error_reference_magnitude"] = 0.0  # type: ignore[index]

    with pytest.raises(ReportValidationError, match="maximum-error witness exceeded"):
        validate_report(report)


def test_validation_rejects_non_fp16_cuda_extension_report() -> None:
    report = _fixture("valid-rmsnorm-v5.json")
    report["results"][0]["dtype"] = "bfloat16"  # type: ignore[index]

    with pytest.raises(ReportValidationError, match="CUDA extension provider supports float16"):
        validate_report(report)


def test_validation_rejects_non_fp16_cuda_extension_suite() -> None:
    report = _fixture("valid-rmsnorm-suite-v3.json")
    report["protocol"]["dtypes"] = ["bfloat16"]  # type: ignore[index]
    report["reports"][0]["dtype"] = "bfloat16"  # type: ignore[index]
    report["reports"][0]["command"][8] = "bfloat16"  # type: ignore[index]

    with pytest.raises(ReportValidationError, match="CUDA extension provider support"):
        validate_report(report)


def test_validation_rejects_wrong_sample_count() -> None:
    report = _fixture("valid-rmsnorm-v3.json")
    report["protocol"]["measured_iterations"] = 3  # type: ignore[index]

    with pytest.raises(ReportValidationError, match="expected 3 samples, got 2"):
        validate_report(report)


def test_validation_rejects_nonfinite_in_memory_number() -> None:
    report = _fixture("valid-rmsnorm-v3.json")
    report["results"][0]["first_call_latency_ms"] = float("nan")  # type: ignore[index]

    with pytest.raises(ReportValidationError, match="non-finite numbers"):
        validate_report(report)


def test_validation_requires_one_environment_per_collective_rank() -> None:
    report = _fixture("valid-allreduce-v3.json")
    report["environment"]["world_size"] = 3  # type: ignore[index]
    report["result"]["correctness_expected"] = 3.0  # type: ignore[index]
    report["result"]["correctness_observed"] = 3.0  # type: ignore[index]

    with pytest.raises(ReportValidationError, match="one record for every global rank"):
        validate_report(report)


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ([], "expected a JSON object"),
        ({"schema_version": 3}, "benchmark_type"),
        ({"benchmark_type": "allreduce", "schema_version": True}, "expected an integer"),
        (
            {"benchmark_type": "allreduce", "schema_version": 99},
            "unsupported report contract",
        ),
    ],
)
def test_validation_rejects_unknown_contracts(report: object, message: str) -> None:
    with pytest.raises(ReportValidationError, match=message):
        validate_report(report)


def test_validate_report_path_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ReportValidationError, match=r"broken\.json:1:2: invalid JSON"):
        validate_report_path(path)


def test_validate_report_path_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReportValidationError, match=r"missing\.json"):
        validate_report_path(tmp_path / "missing.json")


def test_validate_report_path_rejects_nonfinite_json_number(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text('{"value": NaN}\n', encoding="utf-8")

    with pytest.raises(ReportValidationError, match="non-finite number NaN"):
        validate_report_path(path)


def test_validate_report_path_rejects_non_utf8_input(tmp_path: Path) -> None:
    path = tmp_path / "binary.json"
    path.write_bytes(b"\xff")

    with pytest.raises(ReportValidationError, match="not valid UTF-8"):
        validate_report_path(path)


def test_bundle_validation_rejects_child_path_escape(tmp_path: Path) -> None:
    manifest = _fixture("valid-rmsnorm-suite-v1.json")
    manifest["reports"][0]["relative_path"] = "../outside.json"  # type: ignore[index]
    manifest["reports"][0]["command"][-1] = "../outside.json"  # type: ignore[index]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ReportValidationError, match="escapes suite directory"):
        validate_result_bundle_path(path)


def test_bundle_validation_rejects_incomplete_schedule(tmp_path: Path) -> None:
    manifest = _fixture("valid-rmsnorm-suite-v1.json")
    manifest["protocol"]["providers"].append("torch_eager")  # type: ignore[index]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ReportValidationError, match="suite schedule is incomplete"):
        validate_result_bundle_path(path)


def test_bundle_validation_rejects_recorded_command_drift(tmp_path: Path) -> None:
    manifest = _fixture("valid-rmsnorm-suite-v1.json")
    manifest["reports"][0]["command"][4] = "999"  # type: ignore[index]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ReportValidationError, match="command mismatch"):
        validate_result_bundle_path(path)


def test_bundle_validation_rejects_child_commit_drift(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    child_path = tmp_path / "run-01" / "float16" / "triton.json"
    child = json.loads(child_path.read_text(encoding="utf-8"))
    child["git_commit"] = "b" * 40
    child_path.write_text(json.dumps(child), encoding="utf-8")

    with pytest.raises(ReportValidationError, match="git_commit mismatch"):
        validate_result_bundle_path(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("device_name", "Different GPU"), ("cuda_compiler_version", "Different compiler")],
)
def test_bundle_validation_rejects_child_environment_drift(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    manifest = _fixture("valid-rmsnorm-suite-v3.json")
    protocol = manifest["protocol"]
    protocol["providers"] = ["torch_eager", "cuda_extension"]  # type: ignore[index]
    second_record = copy.deepcopy(manifest["reports"][0])  # type: ignore[index]
    second_record["provider"] = "torch_eager"
    second_record["relative_path"] = "run-01/float16/torch_eager.json"
    second_record["command"][10] = "torch_eager"
    second_record["command"][-1] = second_record["relative_path"]
    manifest["reports"].append(second_record)  # type: ignore[union-attr]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    source = _fixture("valid-rmsnorm-v5.json")
    for provider in ("cuda_extension", "torch_eager"):
        child = copy.deepcopy(source)
        child["protocol"]["provider_order"] = [provider]  # type: ignore[index]
        child["results"][0]["provider"] = provider  # type: ignore[index]
        if provider == "torch_eager":
            child["environment"][field] = value  # type: ignore[index]
        child_path = tmp_path / "run-01" / "float16" / f"{provider}.json"
        child_path.parent.mkdir(parents=True, exist_ok=True)
        child_path.write_text(json.dumps(child), encoding="utf-8")

    with pytest.raises(ReportValidationError, match=rf"child environment {field} mismatch"):
        validate_result_bundle_path(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warmup_iterations", 26),
        ("measured_iterations", 3),
        ("seed", 18),
        ("epsilon", 2e-6),
    ],
)
def test_bundle_validation_rejects_child_protocol_drift(
    tmp_path: Path,
    field: str,
    value: int | float,
) -> None:
    path = _write_bundle(tmp_path)
    child_path = tmp_path / "run-01" / "float16" / "triton.json"
    child = json.loads(child_path.read_text(encoding="utf-8"))
    child["protocol"][field] = value
    if field == "measured_iterations":
        child["results"][0]["latency_ms"] = {
            "samples": [0.1, 0.11, 0.12],
            "median": 0.11,
            "observed_minimum": 0.1,
            "observed_maximum": 0.12,
        }
    child_path.write_text(json.dumps(child), encoding="utf-8")

    with pytest.raises(ReportValidationError, match=rf"child protocol {field} mismatch"):
        validate_result_bundle_path(path)


def test_bundle_validation_rejects_child_dtype_drift(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    child_path = tmp_path / "run-01" / "float16" / "triton.json"
    child = json.loads(child_path.read_text(encoding="utf-8"))
    child["results"][0]["dtype"] = "bfloat16"
    child_path.write_text(json.dumps(child), encoding="utf-8")

    with pytest.raises(ReportValidationError, match="child result identity mismatch"):
        validate_result_bundle_path(path)


def test_bundle_validation_rejects_child_shape_drift(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    child_path = tmp_path / "run-01" / "float16" / "triton.json"
    child = json.loads(child_path.read_text(encoding="utf-8"))
    child["results"][0]["shape"] = [128, 2048]
    child_path.write_text(json.dumps(child), encoding="utf-8")

    with pytest.raises(ReportValidationError, match="shape coverage mismatch"):
        validate_result_bundle_path(path)
