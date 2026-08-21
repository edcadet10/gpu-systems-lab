"""Isolated-process orchestration for residual plus RMSNorm measurements."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from gpu_systems_lab.benchmark_inputs import (
    choice_list,
    positive_float,
    positive_integer,
    positive_integer_list,
)
from gpu_systems_lab.reporting import git_commit, git_dirty
from gpu_systems_lab.result_validation import (
    ReportValidationError,
    validate_report,
    validate_result_bundle_path,
)

PROVIDERS = ("torch_eager", "cuda_extension", "triton", "torch_compile")
DEFAULT_PROVIDERS = ("torch_eager", "triton", "torch_compile")
DTYPES = ("float16", "bfloat16", "float32")
CommandRunner = Callable[..., Any]


class SuiteRunError(RuntimeError):
    """An isolated child benchmark or suite precondition failed."""


@dataclass(frozen=True)
class RunSpec:
    run_index: int
    dtype: str
    provider: str


def _validate_unique_choices(values: Sequence[str], allowed: Sequence[str], label: str) -> None:
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be a non-empty unique sequence")
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"unsupported {label.rstrip('s')}: {unknown[0]}")


def _validate_dimensions(values: Sequence[int], label: str) -> None:
    if not values or len(values) != len(set(values)) or any(value < 1 for value in values):
        raise ValueError(f"{label} must be a non-empty unique sequence of positive integers")


def build_schedule(
    *,
    providers: Sequence[str],
    dtypes: Sequence[str],
    process_runs: int,
    schedule_seed: int,
) -> list[RunSpec]:
    """Build a deterministic schedule with provider order shuffled within each run."""

    _validate_unique_choices(providers, PROVIDERS, "providers")
    _validate_unique_choices(dtypes, DTYPES, "dtypes")
    if process_runs < 1:
        raise ValueError("process_runs must be positive")
    schedule: list[RunSpec] = []
    combinations = [(dtype, provider) for dtype in dtypes for provider in providers]
    for run_index in range(1, process_runs + 1):
        ordered = list(combinations)
        random.Random(schedule_seed + run_index - 1).shuffle(ordered)
        schedule.extend(
            RunSpec(run_index=run_index, dtype=dtype, provider=provider)
            for dtype, provider in ordered
        )
    return schedule


def _child_arguments(
    spec: RunSpec,
    *,
    rows: Sequence[int],
    hidden_sizes: Sequence[int],
    warmup: int,
    repeats: int,
    base_seed: int,
    epsilon: float,
    output: str,
) -> list[str]:
    return [
        "-m",
        "gpu_systems_lab.benchmarks.residual_rmsnorm",
        "--rows",
        ",".join(str(value) for value in rows),
        "--hidden",
        ",".join(str(value) for value in hidden_sizes),
        "--dtype",
        spec.dtype,
        "--providers",
        spec.provider,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--seed",
        str(base_seed + spec.run_index - 1),
        "--epsilon",
        str(epsilon),
        "--output",
        output,
    ]


def run_suite(
    *,
    rows: Sequence[int],
    hidden_sizes: Sequence[int],
    dtypes: Sequence[str],
    providers: Sequence[str],
    process_runs: int,
    warmup: int,
    repeats: int,
    base_seed: int,
    epsilon: float,
    schedule_seed: int,
    output_dir: Path,
    allow_unversioned: bool = False,
    runner: CommandRunner = subprocess.run,
) -> Path:
    """Run every provider in a fresh process and write a validated suite manifest."""

    commit = git_commit()
    dirty = git_dirty()
    if not allow_unversioned and (commit is None or dirty is not False):
        raise SuiteRunError(
            "publishable suites require a clean Git commit; use --allow-unversioned for local work"
        )
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    _validate_dimensions(rows, "rows")
    _validate_dimensions(hidden_sizes, "hidden_sizes")
    if "cuda_extension" in providers and any(dtype != "float16" for dtype in dtypes):
        raise ValueError("the CUDA extension provider supports float16 suites only")
    schedule = build_schedule(
        providers=providers,
        dtypes=dtypes,
        process_runs=process_runs,
        schedule_seed=schedule_seed,
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise SuiteRunError(f"output directory already exists: {output_dir}") from error

    reports: list[dict[str, Any]] = []
    for spec in schedule:
        relative_path = Path(f"run-{spec.run_index:02d}") / spec.dtype / f"{spec.provider}.json"
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        actual_arguments = _child_arguments(
            spec,
            rows=rows,
            hidden_sizes=hidden_sizes,
            warmup=warmup,
            repeats=repeats,
            base_seed=base_seed,
            epsilon=epsilon,
            output=str(output_path),
        )
        result = runner(
            [sys.executable, *actual_arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no child output"
            raise SuiteRunError(
                f"run {spec.run_index} {spec.dtype}/{spec.provider} failed: {detail}"
            )
        display_arguments = _child_arguments(
            spec,
            rows=rows,
            hidden_sizes=hidden_sizes,
            warmup=warmup,
            repeats=repeats,
            base_seed=base_seed,
            epsilon=epsilon,
            output=relative_path.as_posix(),
        )
        reports.append(
            {
                "run_index": spec.run_index,
                "dtype": spec.dtype,
                "provider": spec.provider,
                "relative_path": relative_path.as_posix(),
                "command": ["python", *display_arguments],
            }
        )

    manifest = {
        "schema_version": 3,
        "benchmark_type": "residual_rmsnorm_suite",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "protocol": {
            "process_runs": process_runs,
            "providers": list(providers),
            "dtypes": list(dtypes),
            "rows": list(rows),
            "hidden_sizes": list(hidden_sizes),
            "warmup_iterations": warmup,
            "measured_iterations": repeats,
            "base_seed": base_seed,
            "epsilon": epsilon,
            "schedule_seed": schedule_seed,
            "provider_isolation": "one provider per fresh Python process",
        },
        "reports": reports,
    }
    validate_report(manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        validate_result_bundle_path(manifest_path)
    except ReportValidationError as error:
        raise SuiteRunError(f"suite validation failed: {error}") from error
    return manifest_path


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / "local" / f"rmsnorm-suite-{timestamp}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive_integer_list, default=[128, 1024])
    parser.add_argument("--hidden", type=positive_integer_list, default=[1024, 4096, 8192])
    parser.add_argument(
        "--dtypes",
        type=partial(choice_list, allowed=DTYPES, label="dtypes"),
        default=["float16"],
    )
    parser.add_argument(
        "--providers",
        type=partial(choice_list, allowed=PROVIDERS, label="providers"),
        default=list(DEFAULT_PROVIDERS),
    )
    parser.add_argument("--process-runs", type=positive_integer, default=5)
    parser.add_argument("--warmup", type=positive_integer, default=25)
    parser.add_argument("--repeats", type=positive_integer, default=100)
    parser.add_argument("--base-seed", type=int, default=17)
    parser.add_argument("--schedule-seed", type=int, default=2026)
    parser.add_argument("--epsilon", type=positive_float, default=1e-6)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-unversioned",
        action="store_true",
        help="allow a missing or dirty Git commit for local, non-publishable runs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = run_suite(
            rows=args.rows,
            hidden_sizes=args.hidden,
            dtypes=args.dtypes,
            providers=args.providers,
            process_runs=args.process_runs,
            warmup=args.warmup,
            repeats=args.repeats,
            base_seed=args.base_seed,
            epsilon=args.epsilon,
            schedule_seed=args.schedule_seed,
            output_dir=args.output_dir or _default_output_dir(),
            allow_unversioned=args.allow_unversioned,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"manifest": str(manifest), "status": "complete"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
