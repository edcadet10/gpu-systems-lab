"""Forward-only fused residual addition and RMSNorm.

The accelerated path computes one row per Triton program. Input, residual, and
weight values are promoted to FP32 for arithmetic, then cast to the input dtype at
the output. The public dispatcher preserves a PyTorch fallback for unsupported
devices and layouts.
"""

from __future__ import annotations

import math
import os
from typing import Any, Literal

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal installs
    torch = None  # type: ignore[assignment]

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - exercised in CPU-only CI
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if triton is not None:

    @triton.jit
    def _residual_rmsnorm_kernel(
        input_pointer,
        residual_pointer,
        weight_pointer,
        output_pointer,
        row_stride,
        hidden_size,
        epsilon,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < hidden_size
        row_offsets = row * row_stride + columns

        values = tl.load(input_pointer + row_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_pointer + row_offsets, mask=mask, other=0.0).to(tl.float32)
        summed = values + residual
        summed = tl.where(mask, summed, 0.0)
        mean_square = tl.sum(summed * summed, axis=0) / hidden_size
        inverse_rms = tl.rsqrt(mean_square + epsilon)
        weight = tl.load(weight_pointer + columns, mask=mask, other=0.0).to(tl.float32)
        output = summed * inverse_rms * weight
        tl.store(output_pointer + row_offsets, output, mask=mask)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required; install the 'torch' or 'gpu' extra")
    return torch


def _validate_common(input_tensor: Any, residual: Any, weight: Any, epsilon: float) -> None:
    framework = _require_torch()
    if not all(isinstance(tensor, framework.Tensor) for tensor in (input_tensor, residual, weight)):
        raise TypeError("input_tensor, residual, and weight must be PyTorch tensors")
    if input_tensor.shape != residual.shape:
        raise ValueError("input_tensor and residual must have identical shapes")
    if input_tensor.ndim < 1 or input_tensor.shape[-1] == 0:
        raise ValueError("input_tensor must have a non-empty final dimension")
    if weight.ndim != 1 or weight.shape[0] != input_tensor.shape[-1]:
        raise ValueError("weight must be one-dimensional and match the final input dimension")
    if not all(tensor.device == input_tensor.device for tensor in (residual, weight)):
        raise ValueError("all tensors must be on the same device")
    if not all(tensor.dtype == input_tensor.dtype for tensor in (residual, weight)):
        raise ValueError("all tensors must have the same dtype")
    if not input_tensor.is_floating_point():
        raise ValueError("only floating-point tensors are supported")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be a positive finite number")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number")


def residual_rmsnorm_reference(
    input_tensor: Any,
    residual: Any,
    weight: Any,
    epsilon: float = 1e-6,
) -> Any:
    """PyTorch reference with FP32 residual addition and normalization."""

    _validate_common(input_tensor, residual, weight, epsilon)
    summed = input_tensor.float() + residual.float()
    inverse_rms = _require_torch().rsqrt(summed.square().mean(dim=-1, keepdim=True) + epsilon)
    return (summed * inverse_rms * weight.float()).to(dtype=input_tensor.dtype)


def _interpreter_enabled() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def _triton_eligible(input_tensor: Any, residual: Any, weight: Any) -> bool:
    framework = _require_torch()
    supported_dtype = input_tensor.dtype in (
        framework.float16,
        framework.bfloat16,
        framework.float32,
    )
    supported_device = input_tensor.is_cuda or _interpreter_enabled()
    return bool(
        triton is not None
        and supported_dtype
        and supported_device
        and input_tensor.numel() > 0
        and input_tensor.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
    )


def residual_rmsnorm_triton(
    input_tensor: Any,
    residual: Any,
    weight: Any,
    epsilon: float = 1e-6,
) -> Any:
    """Launch the forward Triton kernel or fail with an explicit contract error."""

    _validate_common(input_tensor, residual, weight, epsilon)
    framework = _require_torch()
    if triton is None:
        raise RuntimeError("Triton is required; install the 'gpu' extra")
    if framework.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (input_tensor, residual, weight)
    ):
        raise RuntimeError("the Triton path is forward-only; use the reference path for autograd")
    if not _triton_eligible(input_tensor, residual, weight):
        raise ValueError(
            "the Triton path requires a supported device, dtype, and contiguous tensors"
        )

    hidden_size = input_tensor.shape[-1]
    max_fused_size = 65_536 // input_tensor.element_size()
    block_size = min(max_fused_size, triton.next_power_of_2(hidden_size))
    if hidden_size > block_size:
        raise ValueError("the final dimension must occupy no more than 64 KiB")

    input_2d = input_tensor.reshape(-1, hidden_size)
    residual_2d = residual.reshape(-1, hidden_size)
    output = framework.empty_like(input_2d)
    number_of_warps = min(max(block_size // 256, 1), 8)
    _residual_rmsnorm_kernel[(input_2d.shape[0],)](
        input_2d,
        residual_2d,
        weight,
        output,
        input_2d.stride(0),
        hidden_size,
        epsilon,
        BLOCK_SIZE=block_size,
        num_warps=number_of_warps,
    )
    return output.reshape_as(input_tensor)


def residual_rmsnorm(
    input_tensor: Any,
    residual: Any,
    weight: Any,
    epsilon: float = 1e-6,
    *,
    implementation: Literal["auto", "reference", "triton"] = "auto",
) -> Any:
    """Dispatch to Triton when eligible, otherwise preserve reference semantics."""

    _validate_common(input_tensor, residual, weight, epsilon)
    if implementation == "reference":
        return residual_rmsnorm_reference(input_tensor, residual, weight, epsilon)
    if implementation == "triton":
        return residual_rmsnorm_triton(input_tensor, residual, weight, epsilon)
    if implementation != "auto":
        raise ValueError("implementation must be 'auto', 'reference', or 'triton'")
    if _triton_eligible(input_tensor, residual, weight):
        no_grad_required = not (
            _require_torch().is_grad_enabled()
            and any(tensor.requires_grad for tensor in (input_tensor, residual, weight))
        )
        if no_grad_required:
            return residual_rmsnorm_triton(input_tensor, residual, weight, epsilon)
    return residual_rmsnorm_reference(input_tensor, residual, weight, epsilon)
