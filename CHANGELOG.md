# Changelog

All notable changes are documented here. The project follows semantic versioning for
its Python interfaces; benchmark observations remain scoped to their recorded commit,
environment, shapes, and protocol.

## [Unreleased]

## [0.1.0] - 2026-08-20

### Added

- Forward-only fused residual-plus-RMSNorm Triton kernel and PyTorch reference.
- Safe dispatch with explicit dtype, device, layout, size, and autograd boundaries.
- Correctness-gated single-GPU benchmark with raw CUDA-event samples.
- Correctness-gated NCCL all-reduce benchmark with critical-rank aggregation.
- Logical traffic, roofline, and ring all-reduce models.
- Nsight Compute and Nsight Systems profiling targets.
- CPU CI, contribution forms, security policy, governance, and research notes.

[Unreleased]: https://github.com/edcadet10/gpu-systems-lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/edcadet10/gpu-systems-lab/releases/tag/v0.1.0
