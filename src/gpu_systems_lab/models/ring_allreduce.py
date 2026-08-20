"""A transparent alpha-beta model for a ring all-reduce."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class RingAllReduceEstimate:
    ranks: int
    message_bytes: int
    algorithm_steps: int
    wire_bytes_per_rank: float
    latency_seconds: float
    transfer_seconds: float
    predicted_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def estimate_ring_allreduce(
    *,
    ranks: int,
    message_bytes: int,
    link_bandwidth_gbytes_per_second: float,
    step_latency_us: float,
) -> RingAllReduceEstimate:
    """Estimate ring all-reduce time, excluding reduction-compute cost.

    Reduce-scatter and all-gather each take ``ranks - 1`` steps. Every rank sends
    one ``message_bytes / ranks`` chunk per step. The model assumes a homogeneous,
    full-duplex ring with no contention or topology changes.
    """

    if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 2:
        raise ValueError("ranks must be an integer of at least 2")
    if isinstance(message_bytes, bool) or not isinstance(message_bytes, int) or message_bytes <= 0:
        raise ValueError("message_bytes must be a positive integer")
    _positive_finite("link_bandwidth_gbytes_per_second", link_bandwidth_gbytes_per_second)
    _positive_finite("step_latency_us", step_latency_us)

    steps = 2 * (ranks - 1)
    wire_bytes = 2 * (ranks - 1) * message_bytes / ranks
    latency_seconds = steps * step_latency_us * 1e-6
    transfer_seconds = wire_bytes / (link_bandwidth_gbytes_per_second * 1e9)
    return RingAllReduceEstimate(
        ranks=ranks,
        message_bytes=message_bytes,
        algorithm_steps=steps,
        wire_bytes_per_rank=wire_bytes,
        latency_seconds=latency_seconds,
        transfer_seconds=transfer_seconds,
        predicted_seconds=latency_seconds + transfer_seconds,
    )
