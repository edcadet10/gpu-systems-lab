from __future__ import annotations

import argparse
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_systems_lab.benchmark_inputs import (
    choice_list,
    element_count,
    positive_float,
    positive_integer,
    positive_integer_list,
)
from gpu_systems_lab.benchmarks.residual_rmsnorm import (
    _require_correctness,
    _time_first_call,
)
from gpu_systems_lab.benchmarks.residual_rmsnorm import (
    main as rmsnorm_main,
)
from gpu_systems_lab.distributed.benchmark_allreduce import (
    _assert_collective_correctness,
)
from gpu_systems_lab.distributed.benchmark_allreduce import (
    main as allreduce_main,
)
from gpu_systems_lab.reporting import (
    git_commit,
    git_dirty,
    nvidia_smi_metadata,
    runtime_metadata,
)


@pytest.mark.parametrize(("value", "expected"), [("1", 1), (" 17 ", 17)])
def test_positive_integer_accepts_valid_values(value: str, expected: int) -> None:
    assert positive_integer(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_positive_integer_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        positive_integer(value)


def test_positive_integer_list_parses_whitespace() -> None:
    assert positive_integer_list("1, 17,513") == [1, 17, 513]


@pytest.mark.parametrize("value", ["", "1,,2", "1,0", "1,-2", "1,two", "1,1"])
def test_positive_integer_list_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated"):
        positive_integer_list(value)


@pytest.mark.parametrize(("value", "expected"), [("0.1", 0.1), (" 1e-6 ", 1e-6)])
def test_positive_float_accepts_valid_values(value: str, expected: float) -> None:
    assert positive_float(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_positive_float_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="finite positive"):
        positive_float(value)


def test_element_count_requires_an_exact_positive_multiple() -> None:
    assert element_count(16, 2) == 8
    for message_bytes, element_size in ((0, 2), (15, 2), (16, 0)):
        with pytest.raises(ValueError, match="positive multiple"):
            element_count(message_bytes, element_size)


def test_choice_list_accepts_unique_known_values() -> None:
    assert choice_list("eager, compiled", allowed={"eager", "compiled"}, label="providers") == [
        "eager",
        "compiled",
    ]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "comma-separated"),
        ("eager,other", "unsupported provider"),
        ("eager,eager", "duplicate"),
    ],
)
def test_choice_list_rejects_invalid_values(value: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        choice_list(value, allowed={"eager", "compiled"}, label="providers")


def test_git_commit_returns_resolved_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = SimpleNamespace(returncode=0, stdout="abc123\n")
    monkeypatch.setattr(
        "gpu_systems_lab.reporting.subprocess.run", lambda *args, **kwargs: completed
    )

    assert git_commit() == "abc123"


def test_git_commit_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = SimpleNamespace(returncode=1, stdout="")
    monkeypatch.setattr(
        "gpu_systems_lab.reporting.subprocess.run", lambda *args, **kwargs: completed
    )
    assert git_commit() is None


def test_git_commit_returns_none_when_git_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    def time_out(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired("git", 2)

    monkeypatch.setattr("gpu_systems_lab.reporting.subprocess.run", time_out)
    assert git_commit() is None


@pytest.mark.parametrize(("status_output", "expected"), [("", False), (" M README.md\n", True)])
def test_git_dirty_reports_repository_state(
    monkeypatch: pytest.MonkeyPatch,
    status_output: str,
    expected: bool,
) -> None:
    completed = SimpleNamespace(returncode=0, stdout=status_output)
    monkeypatch.setattr(
        "gpu_systems_lab.reporting.subprocess.run", lambda *args, **kwargs: completed
    )
    assert git_dirty() is expected


def test_runtime_metadata_excludes_host_identity() -> None:
    metadata = runtime_metadata()
    assert set(metadata) == {"python_version", "operating_system", "kernel_release", "machine"}


def test_nvidia_smi_metadata_parses_state() -> None:
    completed = SimpleNamespace(returncode=0, stdout="600.1, P0, 1800, 3000, 700.0\n")
    metadata = nvidia_smi_metadata(0, runner=lambda *args, **kwargs: completed)

    assert metadata == {
        "nvidia_smi_query_status": "ok",
        "driver_version": "600.1",
        "performance_state": "P0",
        "sm_clock_mhz": 1800.0,
        "memory_clock_mhz": 3000.0,
        "power_limit_watts": 700.0,
    }


def test_nvidia_smi_metadata_handles_query_failure() -> None:
    completed = SimpleNamespace(returncode=1, stdout="")
    metadata = nvidia_smi_metadata(0, runner=lambda *args, **kwargs: completed)
    assert metadata["nvidia_smi_query_status"] == "error"
    assert metadata["driver_version"] is None


def test_nvidia_smi_metadata_handles_missing_binary() -> None:
    def missing(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError

    metadata = nvidia_smi_metadata(0, runner=missing)
    assert metadata["nvidia_smi_query_status"] == "unavailable"


class _FakeCuda:
    def __init__(self) -> None:
        self.synchronizations = 0

    def synchronize(self) -> None:
        self.synchronizations += 1


def test_first_call_timer_synchronizes_around_host_clock() -> None:
    cuda = _FakeCuda()
    framework = SimpleNamespace(cuda=cuda)
    clock_values = iter((10.0, 10.125))

    result, latency_ms = _time_first_call(
        framework,
        lambda: "output",
        clock=lambda: next(clock_values),
    )

    assert result == "output"
    assert latency_ms == 125.0
    assert cuda.synchronizations == 2


@pytest.mark.parametrize("error", [float("nan"), float("inf"), 0.003])
def test_correctness_gate_rejects_nonfinite_or_large_error(error: float) -> None:
    with pytest.raises(AssertionError, match="failed correctness"):
        _require_correctness("triton", (2, 17), error, 0.002)


def test_correctness_gate_accepts_finite_error_at_tolerance() -> None:
    _require_correctness("triton", (2, 17), 0.002, 0.002)


def test_rmsnorm_cli_surfaces_runtime_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("GPU unavailable")

    monkeypatch.setattr(
        "gpu_systems_lab.benchmarks.residual_rmsnorm.run_benchmark",
        fail,
    )
    with pytest.raises(SystemExit, match="GPU unavailable"):
        rmsnorm_main([])


def test_allreduce_cli_surfaces_runtime_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("NCCL unavailable")

    monkeypatch.setattr(
        "gpu_systems_lab.distributed.benchmark_allreduce.run",
        fail,
    )
    with pytest.raises(SystemExit, match="NCCL unavailable"):
        allreduce_main([])


class _StatusTensor:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _FakeTorch:
    int32 = "int32"

    @staticmethod
    def tensor(value: int, *, dtype: str, device: str) -> _StatusTensor:
        assert dtype == "int32"
        assert device == "cuda:0"
        return _StatusTensor(value)


class _FakeDistributed:
    class ReduceOp:
        MIN = "min"

    def __init__(self, peer_status: int) -> None:
        self.peer_status = peer_status
        self.calls = 0

    def all_reduce(self, tensor: _StatusTensor, *, op: str) -> None:
        assert op == self.ReduceOp.MIN
        self.calls += 1
        tensor.value = min(tensor.value, self.peer_status)


def test_collective_correctness_consensus_passes_on_every_rank() -> None:
    distributed = _FakeDistributed(peer_status=1)
    _assert_collective_correctness(
        _FakeTorch,
        distributed,
        observed=2.0,
        expected=2.0,
        device="cuda:0",
        rank=0,
    )
    assert distributed.calls == 1


@pytest.mark.parametrize(
    ("observed", "peer_status", "message"),
    [(1.0, 1, "rank 0 observed"), (2.0, 0, "peer rank failed")],
)
def test_collective_correctness_consensus_fails_every_rank(
    observed: float,
    peer_status: int,
    message: str,
) -> None:
    distributed = _FakeDistributed(peer_status=peer_status)
    with pytest.raises(AssertionError, match=message):
        _assert_collective_correctness(
            _FakeTorch,
            distributed,
            observed=observed,
            expected=2.0,
            device="cuda:0",
            rank=0,
        )
    assert distributed.calls == 1
