# GPU Systems Lab

[![CI](https://github.com/edcadet10/gpu-systems-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/edcadet10/gpu-systems-lab/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

Reproducible experiments in GPU kernel fusion, performance modeling, profiling, and
collective communication. This is a
small laboratory: every optimization starts with a numerical contract, every result
captures its environment, and every performance statement must be falsifiable.

> **Current release:** v0.1 implements a forward-only fused residual-plus-RMSNorm
> Triton kernel, an independent PyTorch reference, logical traffic and roofline
> models, a ring all-reduce model, a correctness-gated NCCL benchmark, and Nsight
> profiling targets. No GPU speedup is claimed because the initial development host
> has no CUDA device.

## What this demonstrates

- **Kernel engineering:** masked row reductions, FP32 accumulation, launch heuristics,
  layout checks, and an explicit 64 KiB feature-size boundary in Triton.
- **Framework integration:** a differentiable PyTorch reference, safe automatic
  dispatch, a forward-only accelerated contract, and an optional `torch.compile`
  comparator.
- **Performance engineering:** logical byte accounting, a roofline calculator, raw
  timing samples, warmups, correctness gates, environment capture, and stable NVTX
  ranges for Nsight Compute and Nsight Systems.
- **Distributed systems:** an inspectable alpha-beta ring model and a `torchrun`/NCCL
  benchmark that reports the slowest rank for each measured iteration.
- **Precision discipline:** FP16, BF16, and FP32 input paths with FP32 reduction math.
  FP8 work remains on the roadmap until it can be tested on suitable hardware.
- **Production habits:** typed interfaces, package builds, CPU CI, security guidance,
  benchmark schemas, issue forms, review rules, and a documented claims policy.

The analytical models are intentionally small. They expose assumptions; they do not
simulate a vendor library, a full topology, or a large training fleet.
The exact pre-publication checks are preserved in
[docs/validation.md](docs/validation.md).

## Quick start

The modeling tools have no runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest

gpu-systems-lab traffic --rows 1024 --hidden 4096 --element-bytes 2
gpu-systems-lab roofline \
  --flops 41943040 --bytes 33554432 \
  --peak-tflops 100 --bandwidth-gbytes-s 2000
gpu-systems-lab ring \
  --ranks 8 --message-bytes 134217728 \
  --bandwidth-gbytes-s 50 --latency-us 3
```

On a Linux host with a CUDA-capable GPU:

```bash
python -m pip install -e '.[dev,gpu]'
gpu-lab-rmsnorm \
  --rows 128,1024 \
  --hidden 1024,4096,8192 \
  --dtype float16 \
  --providers torch_eager,triton,torch_compile \
  --output results/local-rmsnorm.json
```

The benchmark refuses to time a provider until its output is within the registered
tolerance of the reference for that shape and dtype. JSON output includes raw
samples, software versions, device identity, seed, warmup count, and commit.

## Profile the kernel

```bash
./scripts/profile_ncu.sh
./scripts/profile_nsys.sh
```

The first command collects roofline, memory-workload, occupancy, and launch sections.
The second captures the CUDA/NVTX timeline. Reports are written below `artifacts/`,
which is intentionally ignored by Git.

## Exercise the collective path

Run one process per GPU:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m gpu_systems_lab.distributed.benchmark_allreduce \
  --message-bytes 134217728 \
  --dtype float16 \
  --output results/local-allreduce.json
```

The benchmark checks the reduction value before timing, synchronizes the ranks outside
the measured range, and reports the per-iteration maximum across ranks. Compare those
measurements with the deliberately simplified model:

```bash
gpu-systems-lab ring \
  --ranks 8 --message-bytes 134217728 \
  --bandwidth-gbytes-s 50 --latency-us 3
```

Disagreement is useful: it shows where topology, protocol selection, contention,
reduction work, or software overhead invalidates the model's assumptions.

## Repository map

```text
src/gpu_systems_lab/
├── kernels/       # PyTorch contract and Triton implementation
├── benchmarks/    # correctness-gated single-GPU timing
├── distributed/   # NCCL benchmark
├── models/        # traffic, roofline, and ring calculations
└── profiling/     # stable profiler targets and NVTX ranges
docs/              # architecture, methodology, research, and roadmap
scripts/           # Nsight command-line recipes
tests/             # CPU, optional framework, and GPU-gated tests
results/           # result schema; local measurements stay untracked by default
```

## Result policy

Performance claims in pull requests must include the raw JSON, the exact command,
five fresh process-level runs for both candidate and baseline, a correctness result,
and profiler evidence for the proposed mechanism. A result from one GPU model is a
result from that GPU model—not a statement about all devices or future runs. See
[the benchmark and claims policy](docs/benchmarking.md).

## Contributing

Issues, benchmark submissions, design critiques, and pull requests are welcome. Start
with [CONTRIBUTING.md](CONTRIBUTING.md), browse the
[roadmap](docs/roadmap.md), or open a
[GitHub Discussion](https://github.com/edcadet10/gpu-systems-lab/discussions).
Please report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Technical basis

The design follows the official Triton LayerNorm reduction pattern, PyTorch's current
benchmark and custom-kernel guidance, NVIDIA's roofline and collective documentation,
and the IO-awareness principle formalized by FlashAttention. The decisions and the
evidence that could overturn them are collected in [docs/research.md](docs/research.md).

Licensed under [Apache 2.0](LICENSE).
