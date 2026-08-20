# GPU Systems Lab

[![CI](https://github.com/edcadet10/gpu-systems-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/edcadet10/gpu-systems-lab/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/edcadet10/gpu-systems-lab)](https://github.com/edcadet10/gpu-systems-lab/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

**An evidence-first GPU performance study: one transformer primitive traced from
its numerical contract through kernel fusion, compiler comparison, reproducible
measurement, profiling, performance models, and collective communication.**

This public laboratory is built for inspection and extension. Every optimization
starts with a reference, every timed path passes correctness first, every result
records its environment, and every performance statement needs a test that could
prove it wrong.

[Evidence](#evidence-status) · [Quick start](#quick-start) ·
[GPU experiment](#run-the-gpu-experiment) · [Profiling](#profile-the-kernel) ·
[Distributed](#exercise-the-collective-path) · [Results](results/README.md) ·
[Research](docs/research.md) · [Contribute](CONTRIBUTING.md)

## Evidence status

| Evidence level | What is present | What it establishes |
| --- | --- | --- |
| Hosted CI | Dependency-free core on Python 3.10–3.14; PyTorch reference/autograd and Triton interpreter on Python 3.12; package and schema checks | CPU logic, operation semantics, packaging, and report-contract behavior |
| Measurement controls | Fresh-process provider isolation, randomized run schedule, first-use and steady-state timing, environment capture, recursive bundle validation | A reproducible protocol ready to collect reviewable GPU evidence |
| Hardware evidence | **None published yet** | No target-GPU correctness, latency, profiler, NCCL, or multi-node claim |

No GPU speedup is claimed. The development host has no CUDA device or driver. Triton
interpreter agreement supports the operation-level contract, but bypasses target
compilation and cannot provide hardware or performance evidence. The empty
[published-result registry](results/published/README.md) makes that boundary auditable.

## The vertical slice

The current package implements forward-only fused residual addition plus RMSNorm and
connects it to the surrounding systems work:

| Layer | Shipped artifact | Explicit boundary |
| --- | --- | --- |
| Numerical contract | Differentiable PyTorch reference with FP32 reduction math | Accelerated path rejects autograd |
| Kernel | Triton row reduction with masking, launch heuristics, and FP16/BF16/FP32 storage | Contiguous tensors; final dimension at most 64 KiB |
| Framework | Safe automatic dispatch and optional `torch.compile` comparator | Direct launch, not yet a structured custom operator |
| Measurement | Correctness gate, isolated process runs, raw CUDA-event samples, first-use timing, environment/driver state, commit capture, recursive validation | Results apply only to the recorded device and protocol |
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

## Run the GPU experiment

On a Linux x86-64 GPU host, install the framework wheel appropriate for the CUDA
stack, start from a clean commit, then:

```bash
python -m pip install -e '.[gpu]'
pytest -m gpu
gpu-lab-rmsnorm-suite \
  --rows 128,1024 \
  --hidden 1024,4096,8192 \
  --dtypes float16,bfloat16 \
  --providers torch_eager,triton,torch_compile \
  --process-runs 5 \
  --output-dir results/local/rmsnorm-example
gpu-systems-lab validate-result results/local/rmsnorm-example/manifest.json
```

The orchestrator shuffles provider/dtype order within each process-level run and
launches exactly one provider per fresh Python process. Each child must pass its
finite-error correctness gate before steady-state timing. Reports retain raw samples,
the first provider invocation, software and device identity, driver/power/clock state,
seeds, canonical replay commands, protocol, and Git state.

The validator checks the manifest plus every child file, not just their individual
schemas. It rejects missing or duplicate schedule entries, path traversal, command or
commit drift, mismatched seeds/providers/dtypes/shapes, non-finite JSON numbers, and
contract violations. Publishable suites require a clean Git commit. Use
`--allow-unversioned` only for local exploration.

The first-use number is diagnostic, not part of the speed acceptance rule. It is the
first invocation of the named provider after tensor setup; non-eager providers run the
eager reference first for correctness. Steady-state comparisons use the CUDA-event
samples. See the current
[RMSNorm report](src/gpu_systems_lab/schemas/rmsnorm-v3.schema.json) and
[suite manifest](src/gpu_systems_lab/schemas/rmsnorm-suite-v1.schema.json) contracts.

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
  --output results/local/allreduce.json
```

Every rank first joins a correctness-status reduction. Only after all ranks agree does
timing begin; each measured iteration is reported at the slowest rank. The current
output contract is
[allreduce-v3.schema.json](src/gpu_systems_lab/schemas/allreduce-v3.schema.json).

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
├── kernels/            # PyTorch contract and Triton implementation
├── benchmarks/         # correctness-gated single-GPU timing
├── benchmark_suite.py  # isolated, randomized process-level orchestration
├── result_validation.py # schema and cross-file bundle validation
├── schemas/            # package-shipped, versioned report contracts
├── distributed/        # NCCL benchmark and rank consensus
├── models/             # traffic, roofline, and ring calculations
└── profiling/          # stable profiler targets and NVTX ranges
docs/                   # architecture, methodology, research, roadmap, validation
results/                # ignored local runs and reviewed published evidence
scripts/                # Nsight command-line recipes
tests/                  # core, framework, interpreter, schema, and GPU-gated tests
```

## Evidence and contribution policy

Performance claims in pull requests must include a recursively valid suite with five
fresh process-level runs for every candidate and baseline, plus profiler evidence for
the proposed mechanism. A result from one GPU model is a result from that GPU
model—not a statement about all devices or future runs. Read the
[benchmark and claims policy](docs/benchmarking.md) and
[result registry](results/README.md) before submitting measurements.

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
