from __future__ import annotations

import math

import pytest

from gpu_systems_lab.models.roofline import estimate_roofline


def test_memory_limited_roofline() -> None:
    estimate = estimate_roofline(
        work_flops=1e9,
        bytes_moved=1e9,
        peak_tflops=10,
        memory_bandwidth_gbytes_per_second=1_000,
    )

    assert estimate.arithmetic_intensity_flops_per_byte == 1
    assert estimate.memory_ceiling_tflops == 1
    assert estimate.attainable_tflops == 1
    assert estimate.limiting_resource == "memory"


def test_compute_limited_roofline() -> None:
    estimate = estimate_roofline(
        work_flops=20e9,
        bytes_moved=1e9,
        peak_tflops=10,
        memory_bandwidth_gbytes_per_second=1_000,
    )

    assert estimate.attainable_tflops == 10
    assert estimate.limiting_resource == "compute"
    assert estimate.to_dict()["memory_ceiling_tflops"] == 20


@pytest.mark.parametrize("invalid", [0, -1, math.inf, math.nan, True, "1"])
def test_roofline_rejects_invalid_inputs(invalid: object) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        estimate_roofline(
            work_flops=invalid,  # type: ignore[arg-type]
            bytes_moved=1,
            peak_tflops=1,
            memory_bandwidth_gbytes_per_second=1,
        )
