from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script_name", "executable", "expected_arguments"),
    [
        (
            "profile_ncu.sh",
            "ncu",
            ("--export", "artifacts/ncu/residual-rmsnorm-cuda_extension"),
        ),
        (
            "profile_nsys.sh",
            "nsys",
            ("--output=artifacts/nsys/residual-rmsnorm-cuda_extension",),
        ),
    ],
)
def test_profiler_script_builds_provider_specific_command(
    tmp_path: Path,
    script_name: str,
    executable: str,
    expected_arguments: tuple[str, ...],
) -> None:
    capture = tmp_path / "arguments.txt"
    fake_profiler = tmp_path / executable
    fake_profiler.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$GPU_LAB_CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_profiler.chmod(0o755)
    environment = os.environ | {
        "GPU_LAB_CAPTURE_PATH": str(capture),
        "GPU_LAB_PROFILE_PROVIDER": "cuda_extension",
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }

    completed = subprocess.run(
        ["bash", str(REPOSITORY_ROOT / "scripts" / script_name)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    for expected in expected_arguments:
        assert expected in arguments
    if executable == "ncu":
        assert "--output" not in arguments
