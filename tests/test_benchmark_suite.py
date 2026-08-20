from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_systems_lab.benchmark_suite import (
    SuiteRunError,
    build_schedule,
    main,
    run_suite,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_schedule_is_deterministic_complete_and_order_varies() -> None:
    arguments = {
        "providers": ["torch_eager", "triton", "torch_compile"],
        "dtypes": ["float16", "float32"],
        "process_runs": 3,
        "schedule_seed": 2026,
    }
    first = build_schedule(**arguments)
    second = build_schedule(**arguments)

    assert first == second
    assert len(first) == 18
    expected = {
        (dtype, provider) for dtype in arguments["dtypes"] for provider in arguments["providers"]
    }
    for run_index in range(1, 4):
        run = [spec for spec in first if spec.run_index == run_index]
        assert {(spec.dtype, spec.provider) for spec in run} == expected
    assert first[:6] != first[6:12]


@pytest.mark.parametrize(
    "arguments",
    [
        {"providers": [], "dtypes": ["float16"], "process_runs": 1},
        {"providers": ["triton", "triton"], "dtypes": ["float16"], "process_runs": 1},
        {"providers": ["unknown"], "dtypes": ["float16"], "process_runs": 1},
        {"providers": ["triton"], "dtypes": ["float16"], "process_runs": 0},
    ],
)
def test_schedule_rejects_invalid_contract(arguments: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        build_schedule(schedule_seed=1, **arguments)


def test_suite_runs_each_provider_in_a_fresh_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr("gpu_systems_lab.benchmark_suite.git_commit", lambda: "a" * 40)
    monkeypatch.setattr("gpu_systems_lab.benchmark_suite.git_dirty", lambda: False)

    def fake_runner(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        output = Path(_argument(command, "--output"))
        report = json.loads((FIXTURES / "valid-rmsnorm-v3.json").read_text(encoding="utf-8"))
        report["protocol"].update(
            {
                "provider_order": [_argument(command, "--providers")],
                "warmup_iterations": int(_argument(command, "--warmup")),
                "measured_iterations": int(_argument(command, "--repeats")),
                "seed": int(_argument(command, "--seed")),
                "epsilon": float(_argument(command, "--epsilon")),
            }
        )
        report["results"][0].update(
            {
                "shape": [2, 17],
                "dtype": _argument(command, "--dtype"),
                "provider": _argument(command, "--providers"),
            }
        )
        report["results"][0]["latency_ms"] = {
            "samples": [0.1, 0.11, 0.12],
            "median": 0.11,
            "observed_minimum": 0.1,
            "observed_maximum": 0.12,
        }
        output.write_text(json.dumps(report) + "\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    output_dir = tmp_path / "suite"
    manifest_path = run_suite(
        rows=[2],
        hidden_sizes=[17],
        dtypes=["float16"],
        providers=["torch_eager", "triton"],
        process_runs=2,
        warmup=2,
        repeats=3,
        base_seed=17,
        epsilon=1e-6,
        schedule_seed=5,
        output_dir=output_dir,
        allow_unversioned=True,
        runner=fake_runner,
    )

    assert len(commands) == 4
    assert all(
        command[1:3] == ["-m", "gpu_systems_lab.benchmarks.residual_rmsnorm"]
        for command in commands
    )
    assert len({command[command.index("--output") + 1] for command in commands}) == 4
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["reports"]) == 4
    assert manifest["protocol"]["provider_isolation"] == "one provider per fresh Python process"
    assert {report["run_index"] for report in manifest["reports"]} == {1, 2}


def test_suite_rejects_valid_but_unrelated_child_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gpu_systems_lab.benchmark_suite.git_commit", lambda: "a" * 40)
    monkeypatch.setattr("gpu_systems_lab.benchmark_suite.git_dirty", lambda: False)

    def fake_runner(command: list[str], **kwargs: Any) -> SimpleNamespace:
        output = Path(_argument(command, "--output"))
        output.write_text(
            (FIXTURES / "valid-rmsnorm-v3.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(SuiteRunError, match="provider_order mismatch"):
        run_suite(
            rows=[128],
            hidden_sizes=[4096],
            dtypes=["float16"],
            providers=["torch_eager"],
            process_runs=1,
            warmup=25,
            repeats=100,
            base_seed=17,
            epsilon=1e-6,
            schedule_seed=1,
            output_dir=tmp_path / "suite",
            runner=fake_runner,
        )


def test_suite_refuses_unversioned_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("gpu_systems_lab.benchmark_suite.git_commit", lambda: "a" * 40)
    monkeypatch.setattr("gpu_systems_lab.benchmark_suite.git_dirty", lambda: True)

    with pytest.raises(SuiteRunError, match="clean Git commit"):
        run_suite(
            rows=[2],
            hidden_sizes=[17],
            dtypes=["float16"],
            providers=["triton"],
            process_runs=1,
            warmup=1,
            repeats=1,
            base_seed=17,
            epsilon=1e-6,
            schedule_seed=1,
            output_dir=tmp_path / "suite",
        )


def test_suite_surfaces_child_failure(tmp_path: Path) -> None:
    failed = SimpleNamespace(returncode=1, stdout="", stderr="kernel failed")

    with pytest.raises(SuiteRunError, match="kernel failed"):
        run_suite(
            rows=[2],
            hidden_sizes=[17],
            dtypes=["float16"],
            providers=["triton"],
            process_runs=1,
            warmup=1,
            repeats=1,
            base_seed=17,
            epsilon=1e-6,
            schedule_seed=1,
            output_dir=tmp_path / "suite",
            allow_unversioned=True,
            runner=lambda *args, **kwargs: failed,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"rows": []},
        {"hidden_sizes": [17, 17]},
        {"warmup": 0},
        {"repeats": 0},
        {"epsilon": 0.0},
    ],
)
def test_suite_rejects_invalid_measurement_inputs(
    tmp_path: Path,
    overrides: dict[str, Any],
) -> None:
    arguments: dict[str, Any] = {
        "rows": [2],
        "hidden_sizes": [17],
        "dtypes": ["float16"],
        "providers": ["triton"],
        "process_runs": 1,
        "warmup": 1,
        "repeats": 1,
        "base_seed": 17,
        "epsilon": 1e-6,
        "schedule_seed": 1,
        "output_dir": tmp_path / "suite",
        "allow_unversioned": True,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError):
        run_suite(**arguments)


def test_suite_refuses_existing_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "suite"
    output_dir.mkdir()

    with pytest.raises(SuiteRunError, match="already exists"):
        run_suite(
            rows=[2],
            hidden_sizes=[17],
            dtypes=["float16"],
            providers=["triton"],
            process_runs=1,
            warmup=1,
            repeats=1,
            base_seed=17,
            epsilon=1e-6,
            schedule_seed=1,
            output_dir=output_dir,
            allow_unversioned=True,
        )


def test_suite_cli_reports_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    captured: dict[str, Any] = {}

    def fake_run_suite(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return manifest

    monkeypatch.setattr("gpu_systems_lab.benchmark_suite.run_suite", fake_run_suite)
    monkeypatch.setattr(
        "gpu_systems_lab.benchmark_suite._default_output_dir", lambda: tmp_path / "default"
    )

    assert main(["--allow-unversioned"]) == 0
    assert captured["output_dir"] == tmp_path / "default"
    assert captured["allow_unversioned"] is True
    assert json.loads(capsys.readouterr().out) == {
        "manifest": str(manifest),
        "status": "complete",
    }


def test_suite_cli_surfaces_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**kwargs: Any) -> Path:
        raise SuiteRunError("measurement failed")

    monkeypatch.setattr("gpu_systems_lab.benchmark_suite.run_suite", fail)

    with pytest.raises(SystemExit, match="measurement failed"):
        main(["--output-dir", "unused", "--allow-unversioned"])
