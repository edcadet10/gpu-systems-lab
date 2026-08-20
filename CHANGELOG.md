# Changelog

All notable changes are documented here. The project follows semantic versioning for
its Python interfaces; benchmark observations remain scoped to their recorded commit,
environment, shapes, and protocol.

## [Unreleased]

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

[Unreleased]: https://github.com/edcadet10/gpu-systems-lab/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/edcadet10/gpu-systems-lab/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/edcadet10/gpu-systems-lab/releases/tag/v0.1.0
