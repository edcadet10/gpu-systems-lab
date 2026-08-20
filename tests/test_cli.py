from __future__ import annotations

import json

import pytest

from gpu_systems_lab.cli import main


@pytest.mark.parametrize(
    ("arguments", "expected_key"),
    [
        (["traffic", "--rows", "2", "--hidden", "4"], "fused_bytes"),
        (
            [
                "roofline",
                "--flops",
                "100",
                "--bytes",
                "10",
                "--peak-tflops",
                "20",
                "--bandwidth-gbytes-s",
                "1000",
            ],
            "attainable_tflops",
        ),
        (
            [
                "ring",
                "--ranks",
                "4",
                "--message-bytes",
                "1024",
                "--bandwidth-gbytes-s",
                "100",
                "--latency-us",
                "1",
            ],
            "predicted_seconds",
        ),
    ],
)
def test_cli_emits_json(
    arguments: list[str], expected_key: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert expected_key in payload
