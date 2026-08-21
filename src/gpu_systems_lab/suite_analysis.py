"""Evaluate a pre-registered candidate-versus-baseline suite claim."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from gpu_systems_lab.result_validation import (
    SUITE_ENVIRONMENT_IDENTITY_FIELDS,
    ValidatedResultBundle,
    load_validated_result_bundle_path,
)


class SuiteAnalysisError(ValueError):
    """A suite cannot support the requested comparison."""


MAX_EXACT_FLOAT_LITERAL_CHARACTERS = 128


def _exact_float_literal(value: str) -> str:
    if len(value) > MAX_EXACT_FLOAT_LITERAL_CHARACTERS:
        raise SuiteAnalysisError(
            "numeric literal exceeds "
            f"{MAX_EXACT_FLOAT_LITERAL_CHARACTERS} characters in comparison input"
        )
    return value


def _bundle_sha256(bundle: ValidatedResultBundle) -> str:
    digest = hashlib.sha256()
    for document in bundle.documents:
        relative_path = document.relative_path.encode("utf-8")
        content = document.content
        digest.update(len(relative_path).to_bytes(8, byteorder="big"))
        digest.update(relative_path)
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
    return digest.hexdigest()


def _exact_documents(bundle: ValidatedResultBundle) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(document.content, parse_float=_exact_float_literal)
        for document in bundle.documents[1:]
    )


def _provider_results(
    documents: tuple[dict[str, Any], ...],
    manifest: dict[str, Any],
    provider: str,
) -> dict[tuple[int, str, int, int], dict[str, Any]]:
    indexed: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    for record, child in zip(manifest["reports"], documents, strict=True):
        if record["provider"] != provider:
            continue
        for result in child["results"]:
            rows, hidden = result["shape"]
            indexed[(record["run_index"], record["dtype"], rows, hidden)] = result
    return indexed


def _environment_identity(bundle: ValidatedResultBundle) -> dict[str, Any]:
    environment = bundle.documents[1].report["environment"]
    return {
        field: environment[field]
        for field in SUITE_ENVIRONMENT_IDENTITY_FIELDS
        if field in environment
    }


def compare_suite(
    manifest_path: Path,
    *,
    candidate: str,
    baseline: str,
    minimum_reduction_percent: float = 5.0,
) -> dict[str, Any]:
    """Apply the repository's broad speed-claim kill criterion to a valid suite."""

    if candidate == baseline:
        raise SuiteAnalysisError("candidate and baseline must be different providers")
    if not 0 < minimum_reduction_percent < 100:
        raise SuiteAnalysisError("minimum reduction percent must be between 0 and 100")

    bundle = load_validated_result_bundle_path(manifest_path)
    if bundle.validation.benchmark_type != "residual_rmsnorm_suite":
        raise SuiteAnalysisError("comparison requires a residual RMSNorm suite manifest")
    manifest = bundle.report
    if manifest["git_dirty"] is not False or not manifest["git_commit"]:
        raise SuiteAnalysisError("comparison requires a clean, versioned suite")
    if manifest["protocol"]["process_runs"] < 5:
        raise SuiteAnalysisError("comparison requires at least five fresh process runs")
    providers = manifest["protocol"]["providers"]
    for label, provider in (("candidate", candidate), ("baseline", baseline)):
        if provider not in providers:
            raise SuiteAnalysisError(f"{label} provider is absent from the suite: {provider}")

    documents = _exact_documents(bundle)
    candidate_results = _provider_results(documents, manifest, candidate)
    baseline_results = _provider_results(documents, manifest, baseline)

    minimum_reduction = Decimal(str(minimum_reduction_percent))
    with localcontext() as context:
        context.prec = max(34, len(minimum_reduction.as_tuple().digits) + 4)
        maximum_fraction = (Decimal(100) - minimum_reduction) / Decimal(100)
    observations: list[dict[str, Any]] = []
    for identity in sorted(candidate_results):
        run_index, dtype, rows, hidden = identity
        candidate_median = candidate_results[identity]["latency_ms"]["median"]
        baseline_median = baseline_results[identity]["latency_ms"]["median"]
        candidate_decimal = Decimal(str(candidate_median))
        baseline_decimal = Decimal(str(baseline_median))
        if candidate_decimal <= 0 or baseline_decimal <= 0:
            raise SuiteAnalysisError(
                "percentage comparison requires positive median latencies; "
                f"run {run_index} {dtype} [{rows}, {hidden}] recorded "
                f"baseline={baseline_decimal} and candidate={candidate_decimal} milliseconds"
            )
        with localcontext() as context:
            context.prec = max(
                34,
                len(baseline_decimal.as_tuple().digits)
                + len(maximum_fraction.as_tuple().digits)
                + 4,
            )
            maximum_candidate_latency = baseline_decimal * maximum_fraction
        criterion_met = candidate_decimal <= maximum_candidate_latency
        observations.append(
            {
                "run_index": run_index,
                "dtype": dtype,
                "shape": [rows, hidden],
                "baseline_median_ms": str(baseline_decimal),
                "candidate_median_ms": str(candidate_decimal),
                "maximum_candidate_latency_ms": str(maximum_candidate_latency),
                "criterion_met": criterion_met,
            }
        )

    broad_claim_holds = all(observation["criterion_met"] for observation in observations)
    return {
        "analysis_type": "rmsnorm_suite_comparison",
        "analysis_version": 1,
        "source_bundle_sha256": _bundle_sha256(bundle),
        "source_git_commit": manifest["git_commit"],
        "source_git_dirty": manifest["git_dirty"],
        "environment_identity": _environment_identity(bundle),
        "candidate": candidate,
        "baseline": baseline,
        "minimum_reduction_percent": str(minimum_reduction),
        "process_runs": manifest["protocol"]["process_runs"],
        "dtypes": manifest["protocol"]["dtypes"],
        "shape_grid": [
            [rows, hidden]
            for rows in manifest["protocol"]["rows"]
            for hidden in manifest["protocol"]["hidden_sizes"]
        ],
        "observation_count": len(observations),
        "broad_claim_status": "held" if broad_claim_holds else "retracted",
        "observations": observations,
    }
