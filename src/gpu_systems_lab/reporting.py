"""Metadata helpers shared by benchmark reports."""

from __future__ import annotations

import csv
import platform
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

CommandRunner = Callable[..., Any]


def _run_command(
    command: Sequence[str], *, timeout: int = 2
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_commit() -> str | None:
    """Return the checked-out Git commit when one can be resolved quickly."""

    result = _run_command(["git", "rev-parse", "HEAD"])
    if result is None:
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def git_dirty() -> bool | None:
    """Return whether tracked or untracked files differ from the checked-out commit."""

    result = _run_command(["git", "status", "--porcelain=v1", "--untracked-files=normal"])
    if result is None or result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def runtime_metadata() -> dict[str, str]:
    """Capture portable runtime identity without hostnames or other machine identifiers."""

    return {
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
    }


def _optional_float(value: str) -> float | None:
    normalized = value.strip()
    if not normalized or normalized.upper() in {"N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def nvidia_smi_metadata(
    device_index: int,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, str | float | None]:
    """Capture stable, non-identifying GPU state from ``nvidia-smi`` when available."""

    command_runner = runner or subprocess.run
    empty: dict[str, str | float | None] = {
        "nvidia_smi_query_status": "unavailable",
        "driver_version": None,
        "performance_state": None,
        "sm_clock_mhz": None,
        "memory_clock_mhz": None,
        "power_limit_watts": None,
    }
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=driver_version,pstate,clocks.current.sm,clocks.current.memory,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = command_runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return empty
    if result.returncode != 0:
        return {**empty, "nvidia_smi_query_status": "error"}
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 5:
        return {**empty, "nvidia_smi_query_status": "error"}
    driver, state, sm_clock, memory_clock, power_limit = (item.strip() for item in rows[0])
    return {
        "nvidia_smi_query_status": "ok",
        "driver_version": driver or None,
        "performance_state": state or None,
        "sm_clock_mhz": _optional_float(sm_clock),
        "memory_clock_mhz": _optional_float(memory_clock),
        "power_limit_watts": _optional_float(power_limit),
    }
