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
| P100 hardware | [Complete FP16 candidate bundle](results/published/p100-a48afa5/) from five fresh processes, 30 paired shape/run decisions, raw samples, correctness witnesses, and environment logs | The registered CUDA-extension timing claim held on this exact grid and stack; reduction ranges were 58.8%–76.6% |
| T4 hardware | [Complete FP16 candidate bundle](results/published/t4-a48afa5/) with 30 paired decisions, dual-device topology, a project collective, a pinned upstream diagnostic, and three retained failed attempts | The RMSNorm claim held at 72.6%–83.1%; the project element-zero witness and upstream `nwrong=0` checks held, but differing process models prohibit a speed/agreement claim |
| P100 trace | [Compatibility-pinned Systems evidence](results/published/p100-a48afa5/profiler/) correlates 20 candidate and 220 eager launches inside the exact 20-operation NVTX ranges | Supports one-versus-eleven launch fusion on the traced shape; no bandwidth, roofline, occupancy, or tensor-core claim |

Both hardware claims are intentionally narrow: FP16 forward-only residual-RMSNorm,
six registered shapes, one device at a time, eager composition as the baseline, and
the retained steady-state protocol. Worst error-to-allowance ratios were 0.9708 on
P100 and 0.9689 on T4 against a failure boundary of 1.0. These are not claims about
other devices, dtypes, backward execution, end-to-end serving, or the kernel's
mechanism. The [published-result registry](results/published/README.md) separates
those states.

## The vertical slice

The current package implements forward-only fused residual addition plus RMSNorm and
connects it to the surrounding systems work:

| Layer | Shipped artifact | Explicit boundary |
| --- | --- | --- |
| Numerical contract | Differentiable PyTorch reference with FP32 reduction math | Accelerated path rejects autograd |
| Kernels | CUDA FP16 block/warp reduction; Triton row reduction with masking and launch heuristics | Both are forward-only; CUDA needs a local toolkit, Triton 3.5+ needs compute capability 8.0+ |
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
stack and start from a clean commit. The CUDA extension additionally requires a local
CUDA toolkit with `nvcc`; the `gpu` extra installs its Ninja build dependency. The
supported Triton releases require NVIDIA compute capability 8.0 or newer; on older
devices, use the explicit FP16 CUDA provider or the eager reference. Pascal targets
also require an explicitly selected CUDA 12.6 framework build; see the
[reproduction guide](docs/reproducing.md#single-gpu-comparison).

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
gpu-systems-lab compare-suite \
  results/local/rmsnorm-example/manifest.json \
  --candidate triton \
  --baseline torch_eager \
  --minimum-reduction-percent 5 \
  --output results/local/rmsnorm-example/triton-vs-eager.json
```

The explicit CUDA provider is FP16-only and can be measured in a separate suite:

```bash
gpu-lab-rmsnorm-suite \
  --rows 128,1024 \
  --hidden 1024,4096,8192 \
  --dtypes float16 \
  --providers torch_eager,cuda_extension \
  --process-runs 5 \
  --output-dir results/local/rmsnorm-cuda-example
gpu-systems-lab compare-suite \
  results/local/rmsnorm-cuda-example/manifest.json \
  --candidate cuda_extension \
  --baseline torch_eager \
  --minimum-reduction-percent 5
```

The orchestrator shuffles provider/dtype order within each process-level run and
launches exactly one provider per fresh Python process. Each child must pass a finite,
scale-aware correctness gate before steady-state timing. The current gate uses the
framework's documented dtype defaults and records the maximum ratio of observed error
to `atol + rtol * abs(reference)` plus compact witnesses that the offline validator
recomputes; any ratio above one fails. Reports retain raw samples, the first provider
invocation, software and device identity,
driver/power/clock state, seeds, canonical replay commands, protocol, and Git state.

The validator checks the manifest plus every child file, not just their individual
schemas. It rejects missing or duplicate schedule entries, path traversal, command or
commit drift, mismatched seeds/providers/dtypes/shapes or stable environment identity,
non-finite JSON numbers, and contract violations. Publishable suites require a clean
Git commit. Use
`--allow-unversioned` only for local exploration.

The comparison command first validates the entire bundle, then applies the registered
speed-claim kill criterion to every run, dtype, and shape. It reports `held` only when
the candidate median is at least 5% lower than the named baseline everywhere; one
failure reports `retracted`. The output includes the exact per-case boundary and a
digest binding the manifest and child reports. It does not turn five observed runs
into a population or future-performance claim. Add `--fail-on-retract` when a CI job
should return status 1 for a valid comparison that retracts the broad claim.

The first-use number is diagnostic, not part of the speed acceptance rule. It is the
first invocation of the named provider after tensor setup; non-eager providers run the
eager reference first for correctness. Steady-state comparisons use the CUDA-event
samples. See the current
[RMSNorm report](src/gpu_systems_lab/schemas/rmsnorm-v5.schema.json) and
[suite manifest](src/gpu_systems_lab/schemas/rmsnorm-suite-v3.schema.json) contracts.

## Profile the kernel

```bash
./scripts/profile_ncu.sh
./scripts/profile_nsys.sh
GPU_LAB_PROFILE_PROVIDER=torch ./scripts/profile_ncu.sh
GPU_LAB_PROFILE_PROVIDER=torch ./scripts/profile_nsys.sh
GPU_LAB_PROFILE_PROVIDER=cuda_extension ./scripts/profile_ncu.sh
GPU_LAB_PROFILE_PROVIDER=cuda_extension ./scripts/profile_nsys.sh
```

The first recipe collects roofline, memory-workload, occupancy, and launch sections.
The second captures the CUDA/NVTX timeline. Reports go below ignored `artifacts/`;
provider-specific names prevent comparisons from overwriting one another. Profiler
collection is kept separate from latency measurement because it perturbs the run.

Export a Systems report to SQLite, then reproduce kernel launches correlated to one
named NVTX range:

```bash
nsys export --type sqlite --output residual-rmsnorm.sqlite \
  artifacts/nsys/residual-rmsnorm-cuda_extension.nsys-rep
gpu-systems-lab analyze-nsys residual-rmsnorm.sqlite \
  --nvtx-range residual_rmsnorm:cuda_extension
```

The analyzer opens SQLite read-only, requires exactly one matching range, and joins
CUDA runtime launch calls on that range's thread to kernel activities by correlation
ID. It reports a SHA-256 binding to the input export.

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
├── csrc/               # CUDA extension source
├── kernels/            # PyTorch contract, CUDA binding, and Triton implementation
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
