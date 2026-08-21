from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from gpu_systems_lab.cli import main
from gpu_systems_lab.profiling.nsys_sqlite import (
    NsightAnalysisError,
    _quote_identifier,
    analyze_nsys_sqlite,
)


def _write_trace_database(path: Path, *, duplicate_range: bool = False) -> None:
    global_pid = 42 << 24
    global_tid = global_pid | 7
    other_global_pid = 43 << 24
    other_global_tid = other_global_pid | 8
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE NVTX_EVENTS (
                start INTEGER NOT NULL,
                end INTEGER,
                text TEXT,
                globalTid INTEGER,
                textId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                globalTid INTEGER NOT NULL,
                correlationId INTEGER NOT NULL
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                correlationId INTEGER NOT NULL,
                globalPid INTEGER NOT NULL,
                demangledName INTEGER,
                shortName INTEGER
            );
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO StringIds VALUES (1, 'kernel_a'), (2, 'kernel_b');
            """
        )
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (100, 200, 'measured', ?, NULL)", (global_tid,)
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
            (
                (110, 111, global_tid, 10),
                (120, 121, global_tid, 11),
                (220, 221, global_tid, 12),
                (130, 131, other_global_tid, 13),
                (190, 210, global_tid, 14),
            ),
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
            (
                (115, 120, 10, global_pid, 1, 1),
                (116, 121, 10, other_global_pid, 2, 2),
                (125, 132, 11, global_pid, 2, 2),
                (225, 230, 12, global_pid, 1, 1),
                (135, 140, 13, other_global_pid, 1, 1),
                (211, 216, 14, global_pid, 1, 1),
            ),
        )
        if duplicate_range:
            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES (300, 400, 'measured', ?, NULL)", (global_tid,)
            )
        connection.commit()
    finally:
        connection.close()


def test_quote_identifier_escapes_sql_delimiters() -> None:
    assert _quote_identifier('kernel"; DROP TABLE StringIds; --') == (
        '"kernel""; DROP TABLE StringIds; --"'
    )


def test_analyze_nsys_sqlite_correlates_only_launches_inside_range(tmp_path: Path) -> None:
    database = tmp_path / "trace.sqlite"
    _write_trace_database(database)

    analysis = analyze_nsys_sqlite(database, nvtx_range="measured")

    assert analysis["analysis_type"] == "nsys_nvtx_kernel_launches"
    assert analysis["analysis_version"] == 1
    assert analysis["kernel_launches"] == 2
    assert analysis["nvtx_range_duration_ns"] == 100
    assert analysis["kernels"] == [
        {"instances": 1, "name": "kernel_a", "total_time_ns": 5},
        {"instances": 1, "name": "kernel_b", "total_time_ns": 7},
    ]
    assert len(analysis["sqlite_sha256"]) == 64


def test_analyze_nsys_cli_writes_replayable_json(tmp_path: Path) -> None:
    database = tmp_path / "trace.sqlite"
    output = tmp_path / "analysis.json"
    _write_trace_database(database)

    assert (
        main(
            [
                "analyze-nsys",
                str(database),
                "--nvtx-range",
                "measured",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == analyze_nsys_sqlite(database, nvtx_range="measured")


def test_analyze_nsys_rejects_missing_or_ambiguous_ranges(tmp_path: Path) -> None:
    with pytest.raises(NsightAnalysisError, match="does not exist"):
        analyze_nsys_sqlite(tmp_path / "missing.sqlite", nvtx_range="measured")

    database = tmp_path / "trace.sqlite"
    _write_trace_database(database, duplicate_range=True)
    with pytest.raises(NsightAnalysisError, match="exactly one"):
        analyze_nsys_sqlite(database, nvtx_range="measured")
