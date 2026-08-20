from __future__ import annotations

import pytest

from gpu_systems_lab.models.ring_allreduce import estimate_ring_allreduce


def test_ring_allreduce_counts_steps_and_wire_bytes() -> None:
    estimate = estimate_ring_allreduce(
        ranks=8,
        message_bytes=800,
        link_bandwidth_gbytes_per_second=100,
        step_latency_us=1,
    )

    assert estimate.algorithm_steps == 14
    assert estimate.wire_bytes_per_rank == 1_400
    assert estimate.latency_seconds == pytest.approx(14e-6)
    assert estimate.transfer_seconds == pytest.approx(1_400 / 100e9)
    assert estimate.predicted_seconds == pytest.approx(14e-6 + 1_400 / 100e9)
    assert estimate.to_dict()["ranks"] == 8


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "ranks": 1,
            "message_bytes": 8,
            "link_bandwidth_gbytes_per_second": 1,
            "step_latency_us": 1,
        },
        {
            "ranks": 2,
            "message_bytes": 0,
            "link_bandwidth_gbytes_per_second": 1,
            "step_latency_us": 1,
        },
        {
            "ranks": 2,
            "message_bytes": 8,
            "link_bandwidth_gbytes_per_second": 0,
            "step_latency_us": 1,
        },
        {
            "ranks": 2,
            "message_bytes": 8,
            "link_bandwidth_gbytes_per_second": 1,
            "step_latency_us": 0,
        },
    ],
)
def test_ring_allreduce_rejects_invalid_inputs(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        estimate_ring_allreduce(**arguments)
