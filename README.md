# GPU Systems Lab

[![CI](https://github.com/edcadet10/gpu-systems-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/edcadet10/gpu-systems-lab/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/edcadet10/gpu-systems-lab)](https://github.com/edcadet10/gpu-systems-lab/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

**An evidence-first GPU performance study: one transformer primitive followed
from numerical contract through kernel fusion, framework integration, profiling,
performance models, and collective communication.**

This public laboratory is built for inspection and extension. Every optimization
starts with a reference, every timed path passes correctness first, every result
records its environment, and every performance statement needs a test that could
prove it wrong.

[Quick start](#quick-start) · [Kernel](#the-vertical-slice) ·
[Evidence](#evidence-status) · [Profiling](#profile-the-kernel) ·
[Distributed](#exercise-the-collective-path) · [Research](docs/research.md) ·
[Roadmap](docs/roadmap.md) · [Contribute](CONTRIBUTING.md)

## Evidence status

| Continuously verified in hosted CI | Hardware-ready; measurement still required |
| --- | --- |
| Python 3.10–3.14 model and schema checks | Target-GPU code generation and numerical behavior |
| PyTorch reference, shape validation, and autograd | CUDA-event latency and speedup against baselines |
| Triton interpreter agreement on awkward FP32 shapes | Nsight counters, occupancy, and physical memory traffic |
| Benchmark parsers, report contracts, and rank-consensus logic | NCCL latency, topology effects, and multi-node behavior |

No GPU speedup is claimed: the development host has no CUDA device or driver. Triton
interpreter results support the operation-level contract, but the interpreter bypasses
target compilation and cannot provide hardware or performance evidence.

## The vertical slice

The current package implements forward-only fused residual addition plus RMSNorm and
connects it to the surrounding systems work:

| Layer | Shipped artifact | Explicit boundary |
| --- | --- | --- |
| Numerical contract | Differentiable PyTorch reference with FP32 reduction math | Accelerated path rejects autograd |
| Kernel | Triton row reduction with masking, launch heuristics, and FP16/BF16/FP32 storage | Contiguous tensors; final dimension at most 64 KiB |
| Framework | Safe automatic dispatch and optional `torch.compile` comparator | Direct launch, not yet a structured custom operator |
| Measurement | Correctness gate, warmups, raw CUDA-event samples, environment and commit capture | Results apply only to the recorded device and protocol |
| Modeling | Logical traffic, roofline, and alpha-beta ring calculators | Models expose assumptions; they are not hardware counters or NCCL simulators |
| Distributed | `torchrun`/NCCL benchmark with cross-rank correctness consensus and critical-rank timing | Requires at least two CUDA devices |

See the [architecture note](docs/architecture.md) for the contract and data flow, and
the [validation record](docs/validation.md) for exact observations and limits.

## Quick start

### Explore the dependency-free models

```bash
git clone https://github.com/edcadet10/gpu-systems-lab.git
cd gpu-systems-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

gpu-systems-lab traffic --rows 1024 --hidden 4096 --element-bytes 2
gpu-systems-lab roofline \
  --flops 41943040 --bytes 33554432 \
  --peak-tflops 100 --bandwidth-gbytes-s 2000
gpu-systems-lab ring \
  --ranks 8 --message-bytes 134217728 \
  --bandwidth-gbytes-s 50 --latency-us 3
```

The commands print structured JSON with units and the assumptions needed to interpret
it. They do not require PyTorch, Triton, CUDA, or a GPU.

### Reproduce the full CPU and interpreter checks

On Linux x86-64:

```bash
python -m pip install 'torch>=2.10,<2.14' \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,gpu]'
python -m pip check
ruff check .
ruff format --check .
pytest --cov=gpu_systems_lab --cov-report=term-missing
TRITON_INTERPRET=1 pytest -m triton_interpreter -vv
python -m build
```

Installing the CPU framework wheel first keeps this path independent of an NVIDIA
driver. The required quality check runs the same reference and interpreter scope; the
version matrix separately exercises the dependency-free core on Python 3.10–3.14.

## Benchmark on one GPU

On a Linux x86-64 GPU host, install the framework wheel appropriate for the CUDA
stack, then:

```bash
python -m pip install -e '.[dev,gpu]'
pytest -m gpu
gpu-lab-rmsnorm \
  --rows 128,1024 \
  --hidden 1024,4096,8192 \
  --dtype float16 \
  --providers torch_eager,triton,torch_compile \
  --output results/local-rmsnorm.json
```

The benchmark refuses to time a provider until its output is within the registered
tolerance of the reference for that shape and dtype. Output contains raw samples,
software and device identity, seed, warmup count, and Git commit. Its machine-readable
contract is [results/rmsnorm.schema.json](results/rmsnorm.schema.json).

## Profile the kernel

```bash
./scripts/profile_ncu.sh
./scripts/profile_nsys.sh
```

The first recipe collects roofline, memory-workload, occupancy, and launch sections.
The second captures the CUDA/NVTX timeline. Reports go below ignored `artifacts/`;
profiler collection is kept separate from latency measurement because it perturbs the
run.

## Exercise the collective path

Run one process per GPU:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m gpu_systems_lab.distributed.benchmark_allreduce \
  --message-bytes 134217728 \
  --dtype float16 \
  --output results/local-allreduce.json
```

Every rank first joins a correctness-status reduction. Only after all ranks agree does
timing begin; each measured iteration is reported at the slowest rank. The output
contract is [results/allreduce.schema.json](results/allreduce.schema.json).

Compare the measurement with the deliberately simplified model:

```bash
gpu-systems-lab ring \
  --ranks 8 --message-bytes 134217728 \
  --bandwidth-gbytes-s 50 --latency-us 3
```

Disagreement is evidence: it identifies where topology, protocol selection,
contention, reduction work, channelization, or software overhead breaks the model's
assumptions.

## Repository map

```text
src/gpu_systems_lab/
├── kernels/           # PyTorch contract and Triton implementation
├── benchmarks/        # correctness-gated single-GPU timing
├── distributed/       # NCCL benchmark and rank consensus
├── models/            # traffic, roofline, and ring calculations
└── profiling/         # stable profiler targets and NVTX ranges
docs/                  # architecture, methodology, research, roadmap, validation
results/               # strict report schemas; local measurements remain ignored
scripts/               # Nsight command-line recipes
tests/                 # core, framework, interpreter, schema, and GPU-gated tests
```

## Evidence and contribution policy

Performance claims in pull requests must include raw JSON, the exact command, five
fresh process-level runs for both candidate and baseline, a correctness result, and
profiler evidence for the proposed mechanism. A result from one GPU model is a result
from that GPU model—not a statement about all devices or future runs. Read the
[benchmark and claims policy](docs/benchmarking.md) before submitting measurements.

Issues, negative results, benchmark submissions, design critiques, and pull requests
are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), browse the
[roadmap](docs/roadmap.md), or open a
[GitHub Discussion](https://github.com/edcadet10/gpu-systems-lab/discussions).
Report vulnerabilities privately through [SECURITY.md](SECURITY.md).

## Technical basis

The design follows primary sources: Triton's official reduction and interpreter
guidance, PyTorch's custom-kernel, benchmark, and distributed semantics, NVIDIA's
profiling and collective documentation, the FlashAttention IO-awareness result, and
the RMSNorm definition. The decisions, counter-evidence, and conditions that would
change them are collected in [docs/research.md](docs/research.md).

Licensed under [Apache 2.0](LICENSE).
