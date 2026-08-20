"""Logical tensor-traffic models.

These counts describe reads and writes implied by an operator decomposition. They
are not measurements of DRAM transactions: caches, compiler fusion, and spills can
all change physical traffic. Use a hardware profiler to test that separate claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrafficEstimate:
    """Logical bytes for the registered residual-plus-RMSNorm decompositions."""

    rows: int
    hidden_size: int
    element_size_bytes: int
    fused_bytes: int
    unfused_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def estimate_residual_rmsnorm_traffic(
    rows: int,
    hidden_size: int,
    element_size_bytes: int = 2,
) -> TrafficEstimate:
    """Count logical bytes for fused and explicitly materialized decompositions.

    The same-width unfused decomposition materializes add, square, normalize, and
    affine outputs. Per element it performs eleven transfers: three for add, two
    for square, one for the reduction input, two for normalization, and three for
    the affine transform. The fused form reads input, residual, and weight once and
    writes output once, for four transfers. Scalar reduction traffic is ignored.
    """

    _positive_integer("rows", rows)
    _positive_integer("hidden_size", hidden_size)
    _positive_integer("element_size_bytes", element_size_bytes)

    tensor_bytes = rows * hidden_size * element_size_bytes
    return TrafficEstimate(
        rows=rows,
        hidden_size=hidden_size,
        element_size_bytes=element_size_bytes,
        fused_bytes=4 * tensor_bytes,
        unfused_bytes=11 * tensor_bytes,
    )
