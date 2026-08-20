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
from gpu_systems_lab.result_validation import ReportValidationError, validate_result_bundle_path


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
