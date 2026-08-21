from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from threading import Event
from time import sleep
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
    _correctness_metrics,
    _cuda_compiler_metadata,
    _require_correctness,
    _time_first_call,
    run_benchmark,
)
from gpu_systems_lab.benchmarks.residual_rmsnorm import (
    main as rmsnorm_main,
)
from gpu_systems_lab.distributed.benchmark_allreduce import (
    _assert_collective_correctness,
    _tensor_extrema,
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

cuda_extension_module = import_module("gpu_systems_lab.kernels.residual_rmsnorm_cuda")


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


class _FakeVisibleCuda:
    capabilities = ((7, 5), (6, 0), (7, 5))

    def device_count(self) -> int:
        return len(self.capabilities)

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return self.capabilities[index]


def test_cuda_compiler_metadata_records_unique_visible_architectures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr("gpu_systems_lab.benchmarks.residual_rmsnorm.shutil.which", lambda _: None)
    framework = SimpleNamespace(cuda=_FakeVisibleCuda())

    metadata = _cuda_compiler_metadata(framework)

    assert metadata["cuda_arch_list"] == "6.0;7.5"


def test_cuda_compiler_metadata_preserves_architecture_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.5+PTX")
    monkeypatch.setattr("gpu_systems_lab.benchmarks.residual_rmsnorm.shutil.which", lambda _: None)
    framework = SimpleNamespace(cuda=_FakeVisibleCuda())

    metadata = _cuda_compiler_metadata(framework)

    assert metadata["cuda_arch_list"] == "7.5+PTX"


def test_cuda_extension_first_build_is_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    extension = object()
    start = Event()

    def compile_extension(_compute_capability: tuple[int, int]) -> object:
        nonlocal calls
        calls += 1
        sleep(0.05)
        return extension

    monkeypatch.setattr(cuda_extension_module, "_EXTENSION_MODULES", {})
    monkeypatch.setattr(cuda_extension_module, "_compile_cuda_extension", compile_extension)

    def load_extension() -> object:
        start.wait()
        return cuda_extension_module._load_cuda_extension((7, 5))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(load_extension) for _ in range(4)]
        start.set()
        loaded = [future.result() for future in futures]

    assert loaded == [extension] * 4
    assert calls == 1


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


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), 1.0001])
def test_correctness_gate_rejects_nonfinite_or_large_error_ratio(ratio: float) -> None:
    with pytest.raises(AssertionError, match="failed correctness"):
        _require_correctness("triton", (2, 17), 0.001, ratio)


@pytest.mark.parametrize("absolute_error", [float("nan"), float("inf")])
def test_correctness_gate_rejects_nonfinite_absolute_error(absolute_error: float) -> None:
    with pytest.raises(AssertionError, match="failed correctness"):
        _require_correctness("triton", (2, 17), absolute_error, 0.5)


def test_correctness_gate_accepts_finite_error_at_ratio_boundary() -> None:
    _require_correctness("triton", (2, 17), 0.00390625, 1.0)


def test_scale_aware_metrics_allow_one_fp16_step_at_large_magnitude() -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([5.0], dtype=torch.float16)
    candidate = torch.nextafter(reference, torch.tensor([float("inf")], dtype=torch.float16))

    metrics = _correctness_metrics(candidate, reference, "float16")

    assert metrics["max_absolute_error"] == 0.00390625
    assert float(metrics["max_error_ratio"]) < 1
    assert metrics["max_absolute_error_reference_magnitude"] == 5.0
    assert metrics["max_error_ratio_absolute_error"] == 0.00390625
    assert metrics["max_error_ratio_reference_magnitude"] == 5.0


def test_scale_aware_metrics_reject_material_near_zero_error() -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([0.0], dtype=torch.float16)
    candidate = torch.tensor([1e-4], dtype=torch.float16)

    metrics = _correctness_metrics(candidate, reference, "float16")

    assert float(metrics["max_error_ratio"]) > 1


def test_scale_aware_metrics_retain_distinct_maximum_witnesses() -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([100.0, 0.0], dtype=torch.float32)
    candidate = torch.tensor([100.0001, 9e-6], dtype=torch.float32)

    metrics = _correctness_metrics(candidate, reference, "float32")

    assert float(metrics["max_absolute_error"]) > 9e-5
    assert metrics["max_absolute_error_reference_magnitude"] == 100.0
    assert float(metrics["max_error_ratio_absolute_error"]) < 1e-5
    assert metrics["max_error_ratio_reference_magnitude"] == 0.0


def test_cuda_extension_benchmark_rejects_non_fp16_before_device_setup() -> None:
    with pytest.raises(ValueError, match="CUDA extension provider supports float16"):
        run_benchmark(
            rows=[2],
            hidden_sizes=[17],
            dtype_name="bfloat16",
            providers=["cuda_extension"],
            warmup=1,
            repeats=1,
            seed=17,
            epsilon=1e-6,
        )


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
        observed_min=2.0,
        observed_max=2.0,
        expected=2.0,
        device="cuda:0",
        rank=0,
    )
    assert distributed.calls == 1


@pytest.mark.parametrize(
    ("observed_min", "observed_max", "peer_status", "message"),
    [
        (1.0, 2.0, 1, "rank 0 observed range"),
        (2.0, 3.0, 1, "rank 0 observed range"),
        (2.0, 2.0, 0, "peer rank failed"),
    ],
)
def test_collective_correctness_consensus_fails_every_rank(
    observed_min: float,
    observed_max: float,
    peer_status: int,
    message: str,
) -> None:
    distributed = _FakeDistributed(peer_status=peer_status)
    with pytest.raises(AssertionError, match=message):
        _assert_collective_correctness(
            _FakeTorch,
            distributed,
            observed_min=observed_min,
            observed_max=observed_max,
            expected=2.0,
            device="cuda:0",
            rank=0,
        )
    assert distributed.calls == 1


class _FakeExtremaTorch:
    @staticmethod
    def aminmax(tensor: list[float]) -> tuple[float, float]:
        return min(tensor), max(tensor)


def test_collective_correctness_extrema_expose_nonfirst_mismatch() -> None:
    assert _tensor_extrema(_FakeExtremaTorch, [2.0, 1.0, 2.0]) == (1.0, 2.0)
