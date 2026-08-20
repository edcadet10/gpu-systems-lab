"""Minimal roofline calculations with explicit units."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class RooflineEstimate:
    arithmetic_intensity_flops_per_byte: float
    memory_ceiling_tflops: float
    attainable_tflops: float
    limiting_resource: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def estimate_roofline(
    *,
    work_flops: float,
    bytes_moved: float,
    peak_tflops: float,
    memory_bandwidth_gbytes_per_second: float,
) -> RooflineEstimate:
    """Return the idealized roofline ceiling for one workload and one device."""

    for name, value in (
        ("work_flops", work_flops),
        ("bytes_moved", bytes_moved),
        ("peak_tflops", peak_tflops),
        ("memory_bandwidth_gbytes_per_second", memory_bandwidth_gbytes_per_second),
    ):
        _positive_finite(name, value)

    intensity = work_flops / bytes_moved
    memory_ceiling = intensity * memory_bandwidth_gbytes_per_second / 1_000
    attainable = min(peak_tflops, memory_ceiling)
    limiter = "memory" if memory_ceiling < peak_tflops else "compute"
    return RooflineEstimate(
        arithmetic_intensity_flops_per_byte=intensity,
        memory_ceiling_tflops=memory_ceiling,
        attainable_tflops=attainable,
        limiting_resource=limiter,
    )
