"""Metadata helpers shared by benchmark reports."""

from __future__ import annotations

import subprocess


def git_commit() -> str | None:
    """Return the checked-out Git commit when one can be resolved quickly."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None
