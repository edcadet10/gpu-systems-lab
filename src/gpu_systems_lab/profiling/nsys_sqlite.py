"""Read-only analysis of CUDA kernels launched inside one Nsight Systems NVTX range."""

from __future__ import annotations

import hashlib
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

# Systems serializes HW, VM, PID, and TID as 8/8/24/24-bit fields. Clearing the
# low TID field turns a globalTid into the matching globalPid.
_IDENTIFIER_COMPONENT_MODULUS = 1 << 24


class NsightAnalysisError(RuntimeError):
    """Raised when a trace export cannot support the requested analysis."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier; values still come from the fixed kernel-table allowlist."""

    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def analyze_nsys_sqlite(path: Path, *, nvtx_range: str) -> dict[str, Any]:
    """Summarize kernels correlated to launch APIs inside an NVTX push/pop range."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise NsightAnalysisError(f"SQLite export does not exist: {path}")
    if not nvtx_range:
        raise NsightAnalysisError("NVTX range must not be empty")

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only = ON")
        tables = _table_names(connection)
        required = {
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "NVTX_EVENTS",
            "StringIds",
        }
        missing = sorted(required - tables)
        if missing:
            raise NsightAnalysisError(f"trace export is missing tables: {', '.join(missing)}")
        kernel_tables = sorted(
            tables
            & {
                "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
                "CUPTI_ACTIVITY_KIND_KERNEL",
            }
        )
        if not kernel_tables:
            raise NsightAnalysisError("trace export has no CUDA kernel activity table")

        ranges = connection.execute(
            """
            SELECT event.start, event.end, event.globalTid
            FROM NVTX_EVENTS AS event
            LEFT JOIN StringIds AS registered_text
              ON registered_text.id = event.textId
            WHERE COALESCE(event.text, registered_text.value) = ?
            ORDER BY event.start
            """,
            (nvtx_range,),
        ).fetchall()
        if len(ranges) != 1:
            raise NsightAnalysisError(
                f"expected exactly one NVTX range named {nvtx_range!r}, found {len(ranges)}"
            )
        range_start, range_end, global_tid = ranges[0]
        if range_end is None or global_tid is None or range_end < range_start:
            raise NsightAnalysisError("NVTX range has invalid timing or thread identity")

        launches: list[tuple[int, int, str]] = []
        for kernel_table in kernel_tables:
            quoted_kernel_table = _quote_identifier(kernel_table)
            launches.extend(
                connection.execute(
                    f"""
                    SELECT kernel.start, kernel.end,
                           COALESCE(demangled.value, short.value, '<unknown>')
                    FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
                    JOIN {quoted_kernel_table} AS kernel
                      ON kernel.correlationId = runtime.correlationId
                     AND kernel.globalPid =
                         runtime.globalTid - (runtime.globalTid % ?)
                    LEFT JOIN StringIds AS demangled
                      ON demangled.id = kernel.demangledName
                    LEFT JOIN StringIds AS short
                      ON short.id = kernel.shortName
                    WHERE runtime.globalTid = ?
                      AND runtime.start >= ?
                      AND runtime.end <= ?
                    ORDER BY runtime.start, kernel.start
                    """,
                    (_IDENTIFIER_COMPONENT_MODULUS, global_tid, range_start, range_end),
                ).fetchall()
            )
    except sqlite3.Error as error:
        raise NsightAnalysisError(f"could not analyze SQLite export: {error}") from error
    finally:
        if connection is not None:
            connection.close()

    by_name: dict[str, list[int]] = defaultdict(list)
    for kernel_start, kernel_end, name in launches:
        if kernel_end < kernel_start:
            raise NsightAnalysisError("kernel activity has a negative duration")
        by_name[name].append(kernel_end - kernel_start)
    kernel_summary = [
        {
            "instances": len(durations),
            "name": name,
            "total_time_ns": sum(durations),
        }
        for name, durations in sorted(by_name.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    return {
        "analysis_type": "nsys_nvtx_kernel_launches",
        "analysis_version": 1,
        "kernel_launches": len(launches),
        "kernels": kernel_summary,
        "nvtx_range": nvtx_range,
        "nvtx_range_duration_ns": range_end - range_start,
        "sqlite_sha256": _file_sha256(resolved),
    }
