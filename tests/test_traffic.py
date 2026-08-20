from __future__ import annotations

import pytest

from gpu_systems_lab.models.traffic import estimate_residual_rmsnorm_traffic


def test_registered_traffic_counts() -> None:
    estimate = estimate_residual_rmsnorm_traffic(rows=2, hidden_size=4, element_size_bytes=2)

    assert estimate.fused_bytes == 64
    assert estimate.unfused_bytes == 176
    assert estimate.to_dict()["hidden_size"] == 4


@pytest.mark.parametrize(
    ("rows", "hidden", "element_size"),
    [(0, 4, 2), (1, -1, 2), (1, 4, 0), (True, 4, 2)],
)
def test_traffic_rejects_non_positive_inputs(rows: int, hidden: int, element_size: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        estimate_residual_rmsnorm_traffic(rows, hidden, element_size)
