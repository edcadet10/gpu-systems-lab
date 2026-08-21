"""Command-line access to the repository's analytical models."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from gpu_systems_lab.models import (
    estimate_residual_rmsnorm_traffic,
    estimate_ring_allreduce,
    estimate_roofline,
)
from gpu_systems_lab.profiling.nsys_sqlite import NsightAnalysisError, analyze_nsys_sqlite
from gpu_systems_lab.result_validation import ReportValidationError, validate_result_bundle_path
from gpu_systems_lab.suite_analysis import SuiteAnalysisError, compare_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu-systems-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    traffic = subparsers.add_parser("traffic", help="model logical residual-RMSNorm traffic")
    traffic.add_argument("--rows", type=int, required=True)
    traffic.add_argument("--hidden", type=int, required=True)
    traffic.add_argument("--element-bytes", type=int, default=2)

    roofline = subparsers.add_parser("roofline", help="calculate an idealized roofline ceiling")
    roofline.add_argument("--flops", type=float, required=True)
    roofline.add_argument("--bytes", type=float, required=True)
    roofline.add_argument("--peak-tflops", type=float, required=True)
    roofline.add_argument(
        "--bandwidth-gbytes-s",
        type=float,
        required=True,
        help="sustained decimal gigabytes per second (GB/s)",
    )

    ring = subparsers.add_parser("ring", help="model a ring all-reduce")
    ring.add_argument("--ranks", type=int, required=True)
    ring.add_argument("--message-bytes", type=int, required=True)
    ring.add_argument(
        "--bandwidth-gbytes-s",
        type=float,
        required=True,
        help="one-way decimal gigabytes per second (GB/s)",
    )
    ring.add_argument("--latency-us", type=float, required=True)

    validate = subparsers.add_parser(
        "validate-result",
        help="validate a benchmark report against its versioned schema",
    )
    validate.add_argument("path", type=Path)

    compare = subparsers.add_parser(
        "compare-suite",
        help="apply a pre-registered speed-claim criterion to a valid suite",
    )
    compare.add_argument("path", type=Path)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--minimum-reduction-percent", type=float, default=5.0)
    compare.add_argument("--output", type=Path)
    compare.add_argument(
        "--fail-on-retract",
        action="store_true",
        help="return exit status 1 when the broad claim is retracted",
    )

    nsys = subparsers.add_parser(
        "analyze-nsys",
        help="summarize CUDA kernel launches inside one NVTX range",
    )
    nsys.add_argument("path", type=Path, help="SQLite export produced by nsys")
    nsys.add_argument("--nvtx-range", required=True)
    nsys.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-result":
        try:
            validation = validate_result_bundle_path(args.path)
        except (ReportValidationError, RuntimeError) as error:
            print(f"invalid report: {error}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "benchmark_type": validation.benchmark_type,
                    "schema_file": validation.schema_file,
                    "schema_version": validation.schema_version,
                    "valid": True,
                    "validated_reports": validation.validated_children + 1,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "compare-suite":
        try:
            comparison = compare_suite(
                args.path,
                candidate=args.candidate,
                baseline=args.baseline,
                minimum_reduction_percent=args.minimum_reduction_percent,
            )
        except (ReportValidationError, SuiteAnalysisError, RuntimeError) as error:
            print(f"invalid comparison: {error}", file=sys.stderr)
            return 2
        serialized = json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False)
        if args.output:
            try:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized + "\n", encoding="utf-8")
            except OSError as error:
                print(f"could not write comparison: {error}", file=sys.stderr)
                return 2
        else:
            print(serialized)
        return 1 if args.fail_on_retract and comparison["broad_claim_status"] == "retracted" else 0
    if args.command == "analyze-nsys":
        try:
            analysis = analyze_nsys_sqlite(args.path, nvtx_range=args.nvtx_range)
        except (NsightAnalysisError, OSError) as error:
            print(f"invalid trace export: {error}", file=sys.stderr)
            return 2
        serialized = json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False)
        if args.output:
            try:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized + "\n", encoding="utf-8")
            except OSError as error:
                print(f"could not write trace analysis: {error}", file=sys.stderr)
                return 2
        else:
            print(serialized)
        return 0
    if args.command == "traffic":
        result = estimate_residual_rmsnorm_traffic(args.rows, args.hidden, args.element_bytes)
    elif args.command == "roofline":
        result = estimate_roofline(
            work_flops=args.flops,
            bytes_moved=args.bytes,
            peak_tflops=args.peak_tflops,
            memory_bandwidth_gbytes_per_second=args.bandwidth_gbytes_s,
        )
    else:
        result = estimate_ring_allreduce(
            ranks=args.ranks,
            message_bytes=args.message_bytes,
            link_bandwidth_gbytes_per_second=args.bandwidth_gbytes_s,
            step_latency_us=args.latency_us,
        )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
