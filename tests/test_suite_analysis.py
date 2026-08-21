from __future__ import annotations

import copy
import json
import shutil
import subprocess
from decimal import localcontext
from pathlib import Path

import pytest

from gpu_systems_lab.cli import main
from gpu_systems_lab.suite_analysis import SuiteAnalysisError, compare_suite

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _command(provider: str, relative_path: str, *, seed: int) -> list[str]:
    return [
        "python",
        "-m",
        "gpu_systems_lab.benchmarks.residual_rmsnorm",
        "--rows",
        "128",
        "--hidden",
        "4096",
        "--dtype",
        "float16",
        "--providers",
        provider,
        "--warmup",
        "25",
        "--repeats",
        "2",
        "--seed",
        str(seed),
        "--epsilon",
        "1e-06",
        "--output",
        relative_path,
    ]


def _write_comparison_bundle(
    tmp_path: Path,
    *,
    candidate_median: float,
    baseline_median: float = 1.0,
    process_runs: int = 5,
    dirty: bool = False,
) -> Path:
    manifest = _load_fixture("valid-rmsnorm-suite-v1.json")
    protocol = manifest["protocol"]
    assert isinstance(protocol, dict)
    protocol["providers"] = ["torch_eager", "triton"]
    protocol["process_runs"] = process_runs
    manifest["git_dirty"] = dirty
    reports: list[dict[str, object]] = []
    base_child = _load_fixture("valid-rmsnorm-v3.json")
    for run_index in range(1, process_runs + 1):
        seed = 16 + run_index
        for provider, median in (
            ("torch_eager", baseline_median),
            ("triton", candidate_median),
        ):
            relative_path = f"run-{run_index:02d}/float16/{provider}.json"
            reports.append(
                {
                    "command": _command(provider, relative_path, seed=seed),
                    "dtype": "float16",
                    "provider": provider,
                    "relative_path": relative_path,
                    "run_index": run_index,
                }
            )
            child = copy.deepcopy(base_child)
            child["git_dirty"] = dirty
            child_protocol = child["protocol"]
            assert isinstance(child_protocol, dict)
            child_protocol["provider_order"] = [provider]
            child_protocol["seed"] = seed
            results = child["results"]
            assert isinstance(results, list)
            result = results[0]
            assert isinstance(result, dict)
            result["provider"] = provider
            result["latency_ms"] = {
                "samples": [median, median],
                "median": median,
                "observed_minimum": median,
                "observed_maximum": median,
            }
            child_path = tmp_path / relative_path
            child_path.parent.mkdir(parents=True, exist_ok=True)
            child_path.write_text(json.dumps(child), encoding="utf-8")
    manifest["reports"] = reports
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_compare_suite_holds_only_when_every_observation_meets_boundary(tmp_path: Path) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.95)

    result = compare_suite(path, candidate="triton", baseline="torch_eager")

    assert result["broad_claim_status"] == "held"
    assert result["source_git_commit"] == "a" * 40
    assert result["source_git_dirty"] is False
    assert len(result["source_bundle_sha256"]) == 64
    assert result["process_runs"] == 5
    assert result["minimum_reduction_percent"] == "5.0"
    assert result["observation_count"] == 5
    assert result["dtypes"] == ["float16"]
    assert result["shape_grid"] == [[128, 4096]]
    assert result["environment_identity"]["device_name"] == "Example GPU"
    assert result["observations"][0] == {
        "run_index": 1,
        "dtype": "float16",
        "shape": [128, 4096],
        "baseline_median_ms": "1.0",
        "candidate_median_ms": "0.95",
        "maximum_candidate_latency_ms": "0.950",
        "criterion_met": True,
    }


def test_compare_suite_retracts_broad_claim_on_one_failure(tmp_path: Path) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.951)

    result = compare_suite(path, candidate="triton", baseline="torch_eager")

    assert result["broad_claim_status"] == "retracted"
    assert result["observations"][0]["criterion_met"] is False


def test_compare_suite_holds_at_exact_nonunit_boundary(tmp_path: Path) -> None:
    path = _write_comparison_bundle(
        tmp_path,
        baseline_median=3.0,
        candidate_median=2.85,
    )

    result = compare_suite(path, candidate="triton", baseline="torch_eager")

    assert result["broad_claim_status"] == "held"
    assert result["observations"][0]["maximum_candidate_latency_ms"] == "2.850"


def test_compare_suite_pins_decimal_precision(tmp_path: Path) -> None:
    path = _write_comparison_bundle(
        tmp_path,
        baseline_median=3.0,
        candidate_median=2.85,
    )

    with localcontext() as context:
        context.prec = 2
        result = compare_suite(path, candidate="triton", baseline="torch_eager")

    assert result["broad_claim_status"] == "held"
    assert result["observations"][0]["maximum_candidate_latency_ms"] == "2.850"


def test_compare_suite_uses_exact_json_decimal_at_boundary(tmp_path: Path) -> None:
    path = _write_comparison_bundle(
        tmp_path,
        baseline_median=3.0,
        candidate_median=2.85,
    )
    for child_path in tmp_path.glob("run-*/float16/triton.json"):
        text = child_path.read_text(encoding="utf-8")
        child_path.write_text(text.replace("2.85", "2.8500000000000001"), encoding="utf-8")

    result = compare_suite(path, candidate="triton", baseline="torch_eager")

    assert result["broad_claim_status"] == "retracted"
    assert result["observations"][0]["candidate_median_ms"] == "2.8500000000000001"
    assert result["observations"][0]["maximum_candidate_latency_ms"] == "2.850"


def test_compare_suite_rejects_zero_duration_percentage(tmp_path: Path) -> None:
    path = _write_comparison_bundle(
        tmp_path,
        baseline_median=0.0,
        candidate_median=0.0,
    )

    with pytest.raises(SuiteAnalysisError, match="positive median latencies"):
        compare_suite(path, candidate="triton", baseline="torch_eager")


def test_compare_suite_rejects_oversized_float_literal(tmp_path: Path) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.9)
    child_path = tmp_path / "run-01" / "float16" / "triton.json"
    text = child_path.read_text(encoding="utf-8")
    oversized_literal = "0." + ("9" * 129)
    child_path.write_text(text.replace("0.9", oversized_literal), encoding="utf-8")

    with pytest.raises(SuiteAnalysisError, match="numeric literal exceeds 128 characters"):
        compare_suite(path, candidate="triton", baseline="torch_eager")


def test_compare_suite_rejects_dirty_bundle(tmp_path: Path) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.9, dirty=True)

    with pytest.raises(SuiteAnalysisError, match="clean, versioned suite"):
        compare_suite(path, candidate="triton", baseline="torch_eager")


def test_compare_suite_rejects_fewer_than_five_runs(tmp_path: Path) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.9, process_runs=1)

    with pytest.raises(SuiteAnalysisError, match="at least five fresh process runs"):
        compare_suite(path, candidate="triton", baseline="torch_eager")


@pytest.mark.parametrize("minimum", [0.0, -1.0, 100.0])
def test_compare_suite_rejects_invalid_reduction_boundary(tmp_path: Path, minimum: float) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.9)

    with pytest.raises(SuiteAnalysisError, match="between 0 and 100"):
        compare_suite(
            path,
            candidate="triton",
            baseline="torch_eager",
            minimum_reduction_percent=minimum,
        )


def test_compare_suite_rejects_missing_provider(tmp_path: Path) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.9)

    with pytest.raises(SuiteAnalysisError, match="absent from the suite"):
        compare_suite(path, candidate="unknown", baseline="torch_eager")


def test_compare_suite_rejects_same_provider(tmp_path: Path) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.9)

    with pytest.raises(SuiteAnalysisError, match="must be different"):
        compare_suite(path, candidate="triton", baseline="triton")


def test_compare_suite_rejects_non_suite_report() -> None:
    path = FIXTURES / "valid-rmsnorm-v3.json"

    with pytest.raises(SuiteAnalysisError, match="requires a residual RMSNorm suite"):
        compare_suite(path, candidate="triton", baseline="torch_eager")


def test_compare_suite_digest_changes_with_valid_byte_change(tmp_path: Path) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.9)
    before = compare_suite(path, candidate="triton", baseline="torch_eager")
    child_path = tmp_path / "run-01" / "float16" / "triton.json"
    child_path.write_text(child_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    after = compare_suite(path, candidate="triton", baseline="torch_eager")

    assert before["source_bundle_sha256"] != after["source_bundle_sha256"]
    assert before["observations"] == after["observations"]


def test_compare_suite_reads_each_document_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.9)
    original_read_bytes = Path.read_bytes
    reads: dict[Path, int] = {}

    def counted_read_bytes(document_path: Path) -> bytes:
        if document_path.is_relative_to(tmp_path):
            reads[document_path] = reads.get(document_path, 0) + 1
        return original_read_bytes(document_path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    compare_suite(path, candidate="triton", baseline="torch_eager")

    assert len(reads) == 11
    assert set(reads.values()) == {1}


def test_compare_suite_cli_emits_auditable_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.94)

    assert (
        main(
            [
                "compare-suite",
                str(path),
                "--candidate",
                "triton",
                "--baseline",
                "torch_eager",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["broad_claim_status"] == "held"
    assert payload["observations"][0]["criterion_met"] is True


def test_compare_suite_console_script_runs_full_comparison(tmp_path: Path) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.94)
    entry_point = shutil.which("gpu-systems-lab")
    assert entry_point is not None

    completed = subprocess.run(
        [
            entry_point,
            "compare-suite",
            str(path),
            "--candidate",
            "triton",
            "--baseline",
            "torch_eager",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["broad_claim_status"] == "held"
    assert payload["observation_count"] == 5


def test_compare_suite_cli_can_fail_on_valid_retraction(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_comparison_bundle(tmp_path / "bundle", candidate_median=0.951)
    output = tmp_path / "comparison.json"

    assert (
        main(
            [
                "compare-suite",
                str(path),
                "--candidate",
                "triton",
                "--baseline",
                "torch_eager",
                "--output",
                str(output),
                "--fail-on-retract",
            ]
        )
        == 1
    )

    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text(encoding="utf-8"))["broad_claim_status"] == "retracted"


def test_compare_suite_cli_reports_invalid_comparison(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_comparison_bundle(tmp_path, candidate_median=0.9)

    assert (
        main(
            [
                "compare-suite",
                str(path),
                "--candidate",
                "triton",
                "--baseline",
                "triton",
            ]
        )
        == 2
    )

    assert "invalid comparison:" in capsys.readouterr().err


def test_compare_suite_cli_reports_output_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_comparison_bundle(tmp_path / "bundle", candidate_median=0.9)

    assert (
        main(
            [
                "compare-suite",
                str(path),
                "--candidate",
                "triton",
                "--baseline",
                "torch_eager",
                "--output",
                str(tmp_path),
            ]
        )
        == 2
    )

    assert "could not write comparison:" in capsys.readouterr().err
