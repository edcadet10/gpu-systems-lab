# Changelog

All notable changes are documented here. The project follows semantic versioning for
its Python interfaces; benchmark observations remain scoped to their recorded commit,
environment, shapes, and protocol.

## [Unreleased]

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
