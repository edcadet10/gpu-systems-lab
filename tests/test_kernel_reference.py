from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gpu_systems_lab.kernels.residual_rmsnorm import (  # noqa: E402
    _triton_eligible,
    _triton_hardware_supported,
    residual_rmsnorm,
    residual_rmsnorm_reference,
    residual_rmsnorm_triton,
)
from gpu_systems_lab.kernels.residual_rmsnorm_cuda import (  # noqa: E402
    residual_rmsnorm_cuda,
)

cuda_extension_module = import_module("gpu_systems_lab.kernels.residual_rmsnorm_cuda")


def test_reference_matches_hand_calculation() -> None:
    input_tensor = torch.tensor([[3.0, 4.0]])
    residual = torch.zeros_like(input_tensor)
    weight = torch.ones(2)

    result = residual_rmsnorm_reference(input_tensor, residual, weight, epsilon=1e-12)

    expected = torch.tensor([[3.0, 4.0]]) / (12.5**0.5)
    torch.testing.assert_close(result, expected)


def test_auto_dispatch_preserves_gradients_on_cpu() -> None:
    input_tensor = torch.randn(2, 7, requires_grad=True)
    residual = torch.randn(2, 7, requires_grad=True)
    weight = torch.randn(7, requires_grad=True)

    residual_rmsnorm(input_tensor, residual, weight).sum().backward()

    assert input_tensor.grad is not None
    assert residual.grad is not None
    assert weight.grad is not None


@pytest.mark.parametrize(
    ("input_shape", "residual_shape", "weight_shape", "message"),
    [((2, 3), (1, 3), (3,), "identical"), ((2, 3), (2, 3), (2,), "match")],
)
def test_reference_validates_shapes(
    input_shape: tuple[int, ...],
    residual_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        residual_rmsnorm_reference(
            torch.ones(input_shape),
            torch.ones(residual_shape),
            torch.ones(weight_shape),
        )


def test_explicit_triton_path_rejects_cpu_without_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    with pytest.raises((RuntimeError, ValueError)):
        residual_rmsnorm_triton(torch.ones(1, 4), torch.ones(1, 4), torch.ones(4))


def test_explicit_cuda_extension_rejects_cpu() -> None:
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        residual_rmsnorm_cuda(
            torch.ones(1, 4, dtype=torch.float16),
            torch.ones(1, 4, dtype=torch.float16),
            torch.ones(4, dtype=torch.float16),
        )


def test_triton_hardware_support_rejects_pre_ampere_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCudaTensor:
        is_cuda = True
        device = torch.device("cuda", 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (7, 5))

    assert not _triton_hardware_supported(FakeCudaTensor())


def test_triton_auto_eligibility_rejects_over_64_kib_final_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    input_tensor = torch.ones((1, 16_385), dtype=torch.float32)

    assert not _triton_eligible(input_tensor, input_tensor, torch.ones(16_385))


def test_packaged_cuda_source_is_present() -> None:
    module_path = Path(cuda_extension_module.__file__).resolve()

    assert (module_path.parent.parent / "csrc" / "residual_rmsnorm_cuda.cu").is_file()


@pytest.mark.gpu
def test_triton_matches_reference_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("the installed Triton release requires compute capability 8.0 or newer")
    torch.manual_seed(17)
    input_tensor = torch.randn((3, 513), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(input_tensor)
    weight = torch.randn((513,), device="cuda", dtype=torch.float16)

    expected = residual_rmsnorm_reference(input_tensor, residual, weight)
    actual = residual_rmsnorm_triton(input_tensor, residual, weight)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-3)


@pytest.mark.gpu
def test_cuda_extension_matches_reference_on_awkward_cuda_shape() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(29)
    shape = (3, 513)
    input_tensor = torch.randn(shape, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(input_tensor)
    weight = torch.randn((shape[-1],), device="cuda", dtype=torch.float16)

    expected = residual_rmsnorm_reference(input_tensor, residual, weight)
    actual = residual_rmsnorm_cuda(input_tensor, residual, weight)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-3)


@pytest.mark.gpu
def test_cuda_extension_matches_reference_on_registered_grid() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(17)
    for shape in (
        (128, 1024),
        (128, 4096),
        (128, 8192),
        (1024, 1024),
        (1024, 4096),
        (1024, 8192),
    ):
        input_tensor = torch.randn(shape, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(input_tensor)
        weight = torch.randn((shape[-1],), device="cuda", dtype=torch.float16)

        expected = residual_rmsnorm_reference(input_tensor, residual, weight)
        actual = residual_rmsnorm_cuda(input_tensor, residual, weight)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-3)


@pytest.mark.triton_interpreter
def test_triton_matches_reference_in_interpreter() -> None:
    import os

    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("set TRITON_INTERPRET=1 to enable this smoke test")
    pytest.importorskip("triton")
    torch.manual_seed(17)
    for shape in ((1, 7), (3, 17), (2, 513)):
        input_tensor = torch.randn(shape, dtype=torch.float32)
        residual = torch.randn_like(input_tensor)
        weight = torch.randn((shape[-1],), dtype=torch.float32)
        expected = residual_rmsnorm_reference(input_tensor, residual, weight)
        actual = residual_rmsnorm_triton(input_tensor, residual, weight)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1.3e-6)
