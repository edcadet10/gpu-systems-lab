from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gpu_systems_lab.kernels.residual_rmsnorm import (  # noqa: E402
    residual_rmsnorm,
    residual_rmsnorm_reference,
    residual_rmsnorm_triton,
)


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


@pytest.mark.gpu
def test_triton_matches_reference_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    torch.manual_seed(17)
    input_tensor = torch.randn((3, 513), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(input_tensor)
    weight = torch.randn((513,), device="cuda", dtype=torch.float16)

    expected = residual_rmsnorm_reference(input_tensor, residual, weight)
    actual = residual_rmsnorm_triton(input_tensor, residual, weight)

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=0)


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
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
