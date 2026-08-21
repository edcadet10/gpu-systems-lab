# Changelog

All notable changes are documented here. The project follows semantic versioning for
its Python interfaces; benchmark observations remain scoped to their recorded commit,
environment, shapes, and protocol.

## [Unreleased]

### Added

- Add `compare-suite`, an auditable evaluator for the pre-registered per-run,
  per-dtype, per-shape speed-claim boundary with a complete-bundle digest.
- Add an explicit, lazily compiled FP16 CUDA extension provider with a one-block-per-row
  reduction and selectable Nsight profiling targets.
- Add RMSNorm reports v4/v5 and suite manifests v2/v3 for the new provider and
  scale-aware correctness evidence while retaining validation support for v3/v4 and
  v1/v2 history.
- Publish checksum-protected P100 and T4 candidate bundles with every raw sample,
  recomputable 30-decision comparisons, hardware/software boundaries, and retained
  failed attempts.
- Add a project element-zero witness and pinned-upstream `nwrong=0` dual-T4
  diagnostics while explicitly withholding a cross-tool claim because their
  execution models differ.
- Add a read-only Systems SQLite analyzer that correlates CUDA runtime launches inside
  an exact NVTX range and binds its result to the input digest.
- Publish a compatibility-pinned P100 trace showing one CUDA-extension launch versus
  eleven eager launches per operation, while retaining the newer tool's zero-CUDA
  negative result and withholding unsupported counter claims.

### Fixed

- Reject RMSNorm reports whose recorded maximum error exceeds their recorded
  tolerance, even when the JSON shape remains schema-valid.
- Reject cross-environment suites and percentage comparisons with non-positive median
  durations; evaluate comparison boundaries from bounded, exact JSON decimal literals.
- Serialize first-use extension builds, isolate cache names by compute capability,
  gate unsupported Triton targets, and preserve provider-specific profiler outputs.
- Use Nsight Compute's supported `--export` option and test both profiler command
  builders with provider-specific report paths.
- Validate collective correctness across the full tensor via its extrema instead of
  sampling only element zero before the all-rank consensus.
- Replace the current absolute-only numerical gate with the framework-documented
  absolute-relative relation, recording a maximum error-to-allowance ratio; historical
  report versions keep their original decisions.
- Add compensated FP32 square summation after a P100 wide-row correctness failure; a
  second absolute-gate failure remains recorded rather than being retroactively passed.

## [0.2.0] - 2026-08-20

### Added

- Add a deterministic suite runner that randomizes the registered provider/dtype
  schedule and executes one provider per fresh Python process.
- Record synchronized first-use timing, Git dirty state, runtime identity, and
  best-effort driver, performance-state, clock, and power metadata.
- Capture device and runtime provenance for every collective rank rather than only the
  reporting rank.
- Ship versioned RMSNorm, all-reduce, and suite schemas inside the Python wheel.
- Add a `validate-result` command that recursively checks suite schedule coverage,
  path confinement, commands, provenance, experiment identity, and child schemas.
- Add an explicit local-versus-published result registry and contribution workflow.

### Fixed

- Reject non-finite correctness errors instead of allowing IEEE comparison semantics
  to bypass the numerical gate.
- Reject non-standard `NaN` and infinity tokens when reading report JSON.
- Require clean, versioned state for publishable suites and refuse to reuse an existing
  output directory.

### Changed

- Advance current report contracts to schema v3 while preserving the historical v2
  contract URLs.
- Separate first-use diagnostics from correctness-gated steady-state CUDA-event
  samples and make current evidence boundaries prominent in the README.

## [0.1.1] - 2026-08-20

### Fixed

- Install the framework and Triton interpreter dependencies in the required quality
  check so reference, autograd, validation, and interpreter tests cannot be silently
  omitted.
- Coordinate the collective correctness decision across every rank before raising,
  preventing a locally detected mismatch from stranding peers in the next collective.

### Changed

- Split benchmark output validation into strict, tested schema-v2 single-GPU and
  all-reduce contracts, adding benchmark identity plus commit metadata to both report
  families.
- Add covered benchmark-input and report-metadata helpers, type-checkable tensor
  annotations, supported Python classifiers, and transparent protected-branch
  operations.
- Redesign the README around evidence status, fast navigation, and separate model,
  CPU-reference, interpreter, GPU, and multi-GPU paths.

## [0.1.0] - 2026-08-20

### Added

- Forward-only fused residual-plus-RMSNorm Triton kernel and PyTorch reference.
- Safe dispatch with explicit dtype, device, layout, size, and autograd boundaries.
- Correctness-gated single-GPU benchmark with raw CUDA-event samples.
- Correctness-gated NCCL all-reduce benchmark with critical-rank aggregation.
- Logical traffic, roofline, and ring all-reduce models.
- Nsight Compute and Nsight Systems profiling targets.
- CPU CI, contribution forms, security policy, governance, and research notes.

[Unreleased]: https://github.com/edcadet10/gpu-systems-lab/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/edcadet10/gpu-systems-lab/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/edcadet10/gpu-systems-lab/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/edcadet10/gpu-systems-lab/releases/tag/v0.1.0
