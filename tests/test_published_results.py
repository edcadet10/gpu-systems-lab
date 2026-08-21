from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import median

from gpu_systems_lab.profiling.nsys_sqlite import analyze_nsys_sqlite
from gpu_systems_lab.result_validation import validate_result_bundle_path
from gpu_systems_lab.suite_analysis import compare_suite

PUBLISHED_ROOT = Path(__file__).resolve().parents[1] / "results" / "published"
P100_RAW = PUBLISHED_ROOT / "p100-a48afa5" / "raw"
P100_PROFILER = P100_RAW.parent / "profiler"
T4_ATTEMPTS = PUBLISHED_ROOT / "t4-a48afa5" / "attempts"
T4_RAW = PUBLISHED_ROOT / "t4-a48afa5" / "raw"
RANGE_SEPARATOR = "\N{EN DASH}"


def _assert_checksum_manifest(root: Path) -> None:
    declared: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative_path = line.split("  ", maxsplit=1)
        path = (root / relative_path).resolve()
        path.relative_to(root.resolve())
        declared[relative_path] = digest

    observed_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(declared) == observed_files
    for relative_path, expected_digest in declared.items():
        observed_digest = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        assert observed_digest == expected_digest, relative_path


def test_p100_raw_import_matches_checksum_manifest() -> None:
    _assert_checksum_manifest(P100_RAW)


def _assert_profiler_text_import(
    attempt: Path, *, expected_archived_paths: set[str]
) -> dict[str, str]:
    declared: dict[str, str] = {}
    for line in (attempt / "runner-SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative_path = line.split("  ", maxsplit=1)
        declared[relative_path] = digest
    observed = {
        path.relative_to(attempt).as_posix()
        for path in attempt.rglob("*")
        if path.is_file() and path.name != "runner-SHA256SUMS" and "correlated" not in path.parts
    }
    assert set(declared) - observed == expected_archived_paths
    assert observed - set(declared) == set()
    for relative_path in observed:
        assert (
            hashlib.sha256((attempt / relative_path).read_bytes()).hexdigest()
            == declared[relative_path]
        )
    return declared


def test_p100_profiler_attempts_bind_text_and_omitted_binaries() -> None:
    v1_omitted = {
        "profiles/residual-rmsnorm-cuda_extension.nsys-rep",
        "profiles/residual-rmsnorm-torch.nsys-rep",
    }
    v2_omitted = v1_omitted | {
        "analysis/residual-rmsnorm-cuda_extension.sqlite",
        "analysis/residual-rmsnorm-torch.sqlite",
    }
    v1_declared = _assert_profiler_text_import(
        P100_PROFILER / "v1", expected_archived_paths=v1_omitted
    )
    v2_declared = _assert_profiler_text_import(
        P100_PROFILER / "v2", expected_archived_paths=v2_omitted
    )
    for version, declared, omitted in (
        ("v1", v1_declared, v1_omitted),
        ("v2", v2_declared, v2_omitted),
    ):
        records = json.loads(
            (P100_PROFILER / version / "profiles.json").read_text(encoding="utf-8")
        )
        recorded = {record["relative_path"]: record["sha256"] for record in records}
        assert recorded == {
            path: declared[path] for path in omitted if path.startswith("profiles/")
        }

    v2_records = json.loads((P100_PROFILER / "v2" / "profiles.json").read_text(encoding="utf-8"))
    public_manifest = json.loads((P100_PROFILER / "public-assets.json").read_text(encoding="utf-8"))
    public_assets = {asset["provider"]: asset for asset in public_manifest["assets"]}
    source_records = {record["provider"]: record for record in v2_records}
    for provider, expected_launches, expected_unique_names in (
        ("torch", 220, 8),
        ("cuda_extension", 20, 1),
    ):
        asset = public_assets[provider]
        source = source_records[provider]
        assert asset["original_report_sha256"] == source["sha256"]
        assert asset["original_report_size_bytes"] == source["size_bytes"]
        assert asset["original_full_sqlite_sha256"] == source["sqlite_sha256"]
        assert asset["original_full_sqlite_size_bytes"] == source["sqlite_size_bytes"]
        analysis = json.loads(
            (P100_PROFILER / "v2" / "correlated" / f"{provider}.json").read_text(encoding="utf-8")
        )
        assert analysis["analysis_type"] == "nsys_nvtx_kernel_launches"
        assert analysis["analysis_version"] == 1
        assert analysis["kernel_launches"] == expected_launches
        assert sum(kernel["instances"] for kernel in analysis["kernels"]) == expected_launches
        assert len(analysis["kernels"]) == expected_unique_names
        assert analysis["sqlite_sha256"] == asset["public_sha256"]

    checksum_lines = set(
        (P100_PROFILER / "public-SHA256SUMS").read_text(encoding="utf-8").splitlines()
    )
    assert checksum_lines == {
        f"{asset['public_sha256']}  {asset['filename']}" for asset in public_assets.values()
    }
    assert public_manifest["transformation"]["exported_tables"] == [
        "NVTX_EVENTS",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "StringIds",
    ]

    assert "SKIPPED:" in (P100_PROFILER / "v1" / "logs" / "stats-nsys-torch.log").read_text(
        encoding="utf-8"
    )
    assert "SKIPPED:" not in (P100_PROFILER / "v2" / "logs" / "stats-nsys-torch.log").read_text(
        encoding="utf-8"
    )


def test_profiler_public_export_filter_removes_unreferenced_strings(tmp_path: Path) -> None:
    database = tmp_path / "trace.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE NVTX_EVENTS (
                start INTEGER NOT NULL,
                end INTEGER,
                text TEXT,
                globalTid INTEGER,
                textId INTEGER,
                jsonTextId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                globalTid INTEGER,
                correlationId INTEGER,
                nameId INTEGER NOT NULL
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                correlationId INTEGER,
                globalPid INTEGER,
                demangledName INTEGER NOT NULL,
                shortName INTEGER NOT NULL,
                mangledName INTEGER
            );
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO NVTX_EVENTS VALUES (100, 200, 'measured', 7, NULL, NULL);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (110, 111, 7, 10, 2);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (115, 120, 10, 0, 1, 1, NULL);
            INSERT INTO StringIds VALUES
                (1, 'kernel_a'),
                (2, 'cudaLaunchKernel'),
                (3, 'UNREFERENCED_SENSITIVE_VALUE');
            """
        )
        filter_sql = (P100_PROFILER / "sanitize-public-export.sql").read_text(encoding="utf-8")
        connection.executescript(filter_sql)
        remaining = dict(connection.execute("SELECT id, value FROM StringIds"))
    finally:
        connection.close()

    assert remaining == {1: "kernel_a", 2: "cudaLaunchKernel"}
    assert analyze_nsys_sqlite(database, nvtx_range="measured")["kernel_launches"] == 1


def test_t4_negative_attempts_are_complete_and_fail_closed() -> None:
    for attempt in ("v1", "v2", "v3"):
        root = T4_ATTEMPTS / attempt
        _assert_checksum_manifest(root)
        status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
        assert status["success"] is False

    for attempt in ("v2", "v3"):
        root = T4_ATTEMPTS / attempt
        suite_validation = validate_result_bundle_path(root / "rmsnorm-suite" / "manifest.json")
        allreduce_validation = validate_result_bundle_path(root / "allreduce-project.json")
        assert suite_validation.validated_children == 10
        assert allreduce_validation.validated_children == 0
        assert (
            json.loads(
                (root / "rmsnorm-suite" / "cuda-extension-vs-eager.json").read_text(
                    encoding="utf-8"
                )
            )["broad_claim_status"]
            == "held"
        )

    v3_upstream = json.loads(
        (T4_ATTEMPTS / "v3" / "allreduce-nccl-tests.json").read_text(encoding="utf-8")
    )
    assert v3_upstream["config"] == {
        "aggregated_iterations": 1,
        "blocking_collectives": "false",
        "graph": 0,
        "iterations": 50,
        "maximum_bytes": 134217728,
        "minimum_bytes": 134217728,
        "ngpus": 2,
        "nthreads": 1,
        "parallel_init": "false",
        "per_iter_skip": 0,
        "per_iter_timing": "true",
        "step_factor": 2,
        "validation": 1,
        "warmup_iters": 10,
    }
    assert "results" not in v3_upstream


def test_t4_success_bundle_recomputes_exactly() -> None:
    _assert_checksum_manifest(T4_RAW)
    manifest = T4_RAW / "rmsnorm-suite" / "manifest.json"
    suite_validation = validate_result_bundle_path(manifest)
    allreduce_validation = validate_result_bundle_path(T4_RAW / "allreduce-project.json")
    recomputed = compare_suite(
        manifest,
        candidate="cuda_extension",
        baseline="torch_eager",
        minimum_reduction_percent=5.0,
    )
    stored = json.loads(
        (T4_RAW / "rmsnorm-suite" / "cuda-extension-vs-eager.json").read_text(encoding="utf-8")
    )
    status = json.loads((T4_RAW / "run_status.json").read_text(encoding="utf-8"))

    assert suite_validation.validated_children == 10
    assert allreduce_validation.validated_children == 0
    assert recomputed == stored
    assert recomputed["broad_claim_status"] == "held"
    assert recomputed["observation_count"] == 30
    assert status["success"] is True
    assert status["cross_tool_comparable"] is False


def test_t4_collective_summary_is_derived_from_raw_samples() -> None:
    project = json.loads((T4_RAW / "allreduce-project.json").read_text(encoding="utf-8"))
    project_samples = project["result"]["critical_path_latency_ms"]["samples"]
    project_median_float = median(project_samples)
    project_median = Decimal(str(project_median_float))
    assert len(project_samples) == 50
    assert project_median_float == project["result"]["critical_path_latency_ms"]["median"]
    assert project["result"]["correctness_expected"] == 2.0
    assert project["result"]["correctness_observed"] == 2.0

    upstream = json.loads((T4_RAW / "allreduce-nccl-tests.json").read_text(encoding="utf-8"))
    assert upstream["config"]["ngpus"] == 2
    assert upstream["config"]["warmup_iters"] == 10
    assert upstream["config"]["iterations"] == 50
    assert upstream["out_of_bounds"] == {"count": 0, "okay": "true"}
    upstream_medians: dict[str, Decimal] = {}
    for mode in ("out_of_place", "in_place"):
        samples = sorted(
            Decimal(str(value)) for value in upstream["results"][0][f"{mode}_per_iter"]["times_us"]
        )
        assert len(samples) == 50
        assert upstream["results"][0][mode]["nwrong"] == 0.0
        upstream_medians[mode] = (samples[24] + samples[25]) / Decimal(2000)

    readme = (T4_RAW.parent / "README.md").read_text(encoding="utf-8")
    assert (
        "| Project benchmark | Two processes, one GPU per process; maximum rank latency "
        f"for each sample | {project_median:.4f} ms |" in readme
    )
    assert (
        "| Pinned upstream, out of place | One process, one thread, two GPUs | "
        f"{upstream_medians['out_of_place']:.4f} ms |" in readme
    )
    assert (
        "| Pinned upstream, in place | One process, one thread, two GPUs | "
        f"{upstream_medians['in_place']:.4f} ms |" in readme
    )


def test_p100_suite_and_stored_comparison_recompute_exactly() -> None:
    suite = P100_RAW / "rmsnorm-suite"
    manifest = suite / "manifest.json"

    validation = validate_result_bundle_path(manifest)
    recomputed = compare_suite(
        manifest,
        candidate="cuda_extension",
        baseline="torch_eager",
        minimum_reduction_percent=5.0,
    )
    stored = json.loads((suite / "cuda-extension-vs-eager.json").read_text(encoding="utf-8"))

    assert validation.schema_version == 3
    assert validation.validated_children == 10
    assert recomputed == stored
    assert recomputed["broad_claim_status"] == "held"
    assert recomputed["observation_count"] == 30


def _assert_readme_table_is_derived_from_raw_reports(raw: Path) -> None:
    suite = raw / "rmsnorm-suite"
    comparison = json.loads((suite / "cuda-extension-vs-eager.json").read_text(encoding="utf-8"))
    by_shape: dict[tuple[int, int], dict[str, list[Decimal]]] = defaultdict(
        lambda: {"baseline": [], "candidate": [], "reduction": [], "ratio": []}
    )
    for observation in comparison["observations"]:
        shape = tuple(observation["shape"])
        baseline = Decimal(observation["baseline_median_ms"])
        candidate = Decimal(observation["candidate_median_ms"])
        by_shape[shape]["baseline"].append(baseline)
        by_shape[shape]["candidate"].append(candidate)
        by_shape[shape]["reduction"].append((baseline - candidate) * 100 / baseline)
    for report_path in suite.glob("run-*/*/cuda_extension.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for result in report["results"]:
            by_shape[tuple(result["shape"])]["ratio"].append(
                Decimal(str(result["correctness"]["max_error_ratio"]))
            )

    readme = (raw.parent / "README.md").read_text(encoding="utf-8")
    for (rows, hidden), values in sorted(by_shape.items()):
        expected_line = (
            f"| `{rows} x {hidden}` | "
            f"{min(values['baseline']):.4f}{RANGE_SEPARATOR}"
            f"{max(values['baseline']):.4f} | "
            f"{min(values['candidate']):.4f}{RANGE_SEPARATOR}"
            f"{max(values['candidate']):.4f} | "
            f"{min(values['reduction']):.1f}%{RANGE_SEPARATOR}"
            f"{max(values['reduction']):.1f}% | "
            f"{max(values['ratio']):.4f} |"
        )
        assert expected_line in readme


def test_p100_readme_table_is_derived_from_raw_reports() -> None:
    _assert_readme_table_is_derived_from_raw_reports(P100_RAW)


def test_t4_readme_table_is_derived_from_raw_reports() -> None:
    _assert_readme_table_is_derived_from_raw_reports(T4_RAW)
