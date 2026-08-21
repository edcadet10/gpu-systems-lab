"""Lazy PyTorch binding for the forward-only FP16 CUDA extension."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from gpu_systems_lab.kernels.residual_rmsnorm import _require_torch, _validate_common

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


_EXTENSION_MODULES: dict[tuple[int, int], Any] = {}
_EXTENSION_BUILD_LOCK = Lock()


def _compile_cuda_extension(compute_capability: tuple[int, int]) -> Any:
    _require_torch()
    try:
        from torch.utils.cpp_extension import load
    except ModuleNotFoundError as error:
        raise RuntimeError("the CUDA extension requires PyTorch build tooling") from error

    source = Path(__file__).resolve().parent.parent / "csrc" / "residual_rmsnorm_cuda.cu"
    if not source.is_file():
        raise RuntimeError(f"the packaged CUDA source is missing: {source}")
    major, minor = compute_capability
    if major < 1 or minor < 0:
        raise RuntimeError(f"invalid CUDA compute capability: {major}.{minor}")
    return load(
        name=f"gpu_systems_lab_residual_rmsnorm_cuda_sm{major}{minor}",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=os.environ.get("GPU_LAB_EXTENSION_VERBOSE", "0") == "1",
    )


def _load_cuda_extension(compute_capability: tuple[int, int]) -> Any:
    """Build at most once per architecture, including under concurrent first use."""

    with _EXTENSION_BUILD_LOCK:
        extension = _EXTENSION_MODULES.get(compute_capability)
        if extension is None:
            extension = _compile_cuda_extension(compute_capability)
            _EXTENSION_MODULES[compute_capability] = extension
        return extension


def residual_rmsnorm_cuda(
    input_tensor: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Compile on first use and launch the forward-only FP16 CUDA kernel."""

    _validate_common(input_tensor, residual, weight, epsilon)
    framework = _require_torch()
    if framework.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (input_tensor, residual, weight)
    ):
        raise RuntimeError(
            "the CUDA extension is forward-only; use the reference path for autograd"
        )
    if not input_tensor.is_cuda:
        raise ValueError("the CUDA extension requires CUDA tensors")
    if input_tensor.dtype != framework.float16:
        raise ValueError("the CUDA extension currently supports float16 only")
    if input_tensor.numel() == 0:
        raise ValueError("the CUDA extension requires non-empty tensors")
    if not all(tensor.is_contiguous() for tensor in (input_tensor, residual, weight)):
        raise ValueError("the CUDA extension requires contiguous tensors")
    compute_capability = tuple(framework.cuda.get_device_capability(input_tensor.device))
    extension = _load_cuda_extension(compute_capability)
    return extension.forward(input_tensor, residual, weight, float(epsilon))
