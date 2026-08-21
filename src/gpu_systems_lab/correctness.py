"""Shared numerical-correctness contracts for benchmark producers and validators."""

from __future__ import annotations

from typing import Final

CORRECTNESS_COMPARISON: Final = "abs(actual - expected) <= atol + rtol * abs(expected)"
CORRECTNESS_TOLERANCES: Final[dict[str, tuple[float, float]]] = {
    "float16": (1e-3, 1e-5),
    "bfloat16": (1.6e-2, 1e-5),
    "float32": (1.3e-6, 1e-5),
}
