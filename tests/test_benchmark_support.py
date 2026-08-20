from __future__ import annotations

import argparse
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_systems_lab.benchmark_inputs import (
    element_count,
    positive_integer,
    positive_integer_list,
)
from gpu_systems_lab.distributed.benchmark_allreduce import _assert_collective_correctness
from gpu_systems_lab.reporting import git_commit


@pytest.mark.parametrize(("value", "expected"), [("1", 1), (" 17 ", 17)])
def test_positive_integer_accepts_valid_values(value: str, expected: int) -> None:
    assert positive_integer(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_positive_integer_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        positive_integer(value)


def test_positive_integer_list_parses_whitespace() -> None:
    assert positive_integer_list("1, 17,513") == [1, 17, 513]


@pytest.mark.parametrize("value", ["", "1,,2", "1,0", "1,-2", "1,two"])
def test_positive_integer_list_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated"):
        positive_integer_list(value)


def test_element_count_requires_an_exact_positive_multiple() -> None:
    assert element_count(16, 2) == 8
    for message_bytes, element_size in ((0, 2), (15, 2), (16, 0)):
        with pytest.raises(ValueError, match="positive multiple"):
            element_count(message_bytes, element_size)


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
