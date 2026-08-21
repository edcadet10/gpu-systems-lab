# Roadmap

The roadmap favors complete, measurable vertical slices. An item is complete only
when it includes a numerical reference, adversarial correctness cases, a benchmark
protocol, profiler instructions, and documented hardware boundaries.

## v0.1 — foundation

- [x] Fused residual-plus-RMSNorm forward kernel in Triton.
- [x] Explicit FP16 CUDA extension with warp/block reduction and current-stream launch.
- [x] PyTorch reference and safe dispatch contract.
- [x] Logical traffic, roofline, and ring all-reduce models.
- [x] Single-GPU and NCCL correctness-gated benchmark entry points.
- [x] Nsight Compute and Nsight Systems targets.
- [x] CPU CI and public contribution templates.

## v0.2 — measurement readiness

- [x] Ship versioned report contracts as installed package resources.
- [x] Add strict single-report and recursive suite validation.
- [x] Isolate each provider in a fresh process and randomize the registered schedule.
- [x] Record first-use timing, Git state, runtime identity, and driver/clock/power state.
- [x] Separate ignored local runs from a reviewed published-evidence registry.
- [x] Automate the pre-registered candidate-versus-baseline claim decision.
- [x] Check in isolated-GPU measurements for at least two accelerator generations
  ([tracking issue](https://github.com/edcadet10/gpu-systems-lab/issues/3)).
- [ ] Compare the custom collective path with `nccl-tests` under matched execution
  semantics; the first dual-T4 diagnostics are retained but not comparable.

## v0.3 — kernel depth

- [ ] Add backward gradients with an independent autograd and finite-difference gate.
- [ ] Compare direct Triton, structured framework operator, compiled composition, and
  a native CUDA or CUTLASS implementation.
- [ ] Add multi-CTA support for rows beyond the current 64 KiB boundary.

## v0.4 — low precision

- [ ] Implement an FP8 or microscaled quantization experiment with explicit format,
  scaling granularity, saturation accounting, and held-out accuracy checks.
- [ ] Separate pre-quantized kernel throughput from end-to-end quantization cost.
- [ ] Add shape-aware tensor-core alignment and capability checks.

## v0.5 — communication and resilience

- [ ] Add all-gather and reduce-scatter benchmarks.
- [ ] Measure compute/communication overlap on separate streams.
- [ ] Capture and model intra-node versus inter-node topology.
- [ ] Add timeout, asynchronous-error, and restart drills without claiming transparent
  recovery where the backend cannot provide it.

## Longer-term experiments

- Fused gated-MLP epilogues and grouped GEMM.
- Paged or block-sparse attention experiments grounded in IO complexity.
- Automated kernel configuration search with held-out shapes.
- End-to-end transformer-block latency and memory accounting.
- Versioned hardware capability records and regression dashboards.
