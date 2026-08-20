from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_systems_lab.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("arguments", "expected_key"),
    [
        (["traffic", "--rows", "2", "--hidden", "4"], "fused_bytes"),
        (
            [
                "roofline",
                "--flops",
                "100",
                "--bytes",
                "10",
                "--peak-tflops",
                "20",
                "--bandwidth-gbytes-s",
                "1000",
            ],
            "attainable_tflops",
        ),
        (
            [
                "ring",
                "--ranks",
                "4",
                "--message-bytes",
                "1024",
                "--bandwidth-gbytes-s",
                "100",
                "--latency-us",
                "1",
            ],
            "predicted_seconds",
        ),
    ],
)
def test_cli_emits_json(
    arguments: list[str], expected_key: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert expected_key in payload


def test_cli_validates_report(capsys: pytest.CaptureFixture[str]) -> None:
    path = FIXTURES / "valid-rmsnorm-v3.json"

    assert main(["validate-result", str(path)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "benchmark_type": "residual_rmsnorm",
        "schema_file": "rmsnorm-v3.schema.json",
        "schema_version": 3,
        "valid": True,
        "validated_reports": 1,
    }


def test_cli_recursively_validates_suite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = (FIXTURES / "valid-rmsnorm-suite-v1.json").read_text(encoding="utf-8")
    child = (FIXTURES / "valid-rmsnorm-v3.json").read_text(encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    child_path = tmp_path / "run-01" / "float16" / "triton.json"
    child_path.parent.mkdir(parents=True)
    manifest_path.write_text(manifest, encoding="utf-8")
    child_path.write_text(child, encoding="utf-8")

    assert main(["validate-result", str(manifest_path)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["validated_reports"] == 2


def test_cli_returns_nonzero_with_precise_validation_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"benchmark_type": "allreduce", "schema_version": 3}\n', encoding="utf-8")

    assert main(["validate-result", str(path)]) == 2

    assert "invalid report: $:" in capsys.readouterr().err
