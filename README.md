# GPU Systems Lab

[![CI](https://github.com/edcadet10/gpu-systems-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/edcadet10/gpu-systems-lab/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/edcadet10/gpu-systems-lab)](https://github.com/edcadet10/gpu-systems-lab/releases/latest)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **In one sentence:** this project combines one AI-model operation into a custom GPU
> kernel, then publishes the tests and measurements needed to check whether it is
> correct and faster.

[What it is](#what-this-project-is) · [Results](#results-in-plain-english) ·
[Try it](#try-it-without-a-gpu) ·
[GPU experiment](#run-the-gpu-experiment) · [Profiling](#profile-the-kernel) ·
[Multi-GPU](#test-communication-between-gpus) · [Evidence files](results/README.md) ·
[Research](docs/research.md) · [Contribute](CONTRIBUTING.md)

## What this project is

AI models perform long chains of mathematical operations. Each operation may require
the computer to send a small program, called a **kernel**, to the GPU. Sending many
separate kernels creates extra work and can move the same data more times than
necessary.

This project focuses on one representative AI operation: adding a residual connection
(reusing information from an earlier step) and applying RMSNorm, a normalization step
used inside transformer-style models. It contains two versions:

- a normal version built from standard framework operations; and
- a custom GPU version that combines the work into one kernel.

On the traced P100 test, the normal version made eleven GPU launches per operation.
The custom version made one. In simple terms, the normal version made eleven separate
trips to complete the task, while the custom version completed it in one trip.

The repository does more than show optimized code. It checks that the custom answer
is still correct, measures both versions on real GPUs, records the raw results, uses a
profiler to investigate why the result changed, and keeps failed experiments instead
of hiding them.

## What this project is useful for

- **An auditable performance study:** readers can inspect the raw measurements,
  profiler evidence, environment details, and failed attempts.
- **A learning resource:** readers can follow one optimization from ordinary code to
  a custom GPU kernel, benchmark, profiler trace, and final conclusion.
- **A reusable test laboratory:** contributors can add another GPU, kernel, data type,
  or experiment without inventing a new measurement format.
- **A starting point for deeper work:** future experiments can cover attention,
  lower-precision formats, backward passes, multi-node communication, and inference
  serving.

## What this project is not

This is not an AI model, chatbot, or complete production serving system. It does not
claim to reproduce a cluster with thousands of GPUs. It is a focused laboratory that
records one complete optimization experiment and the evidence needed to check its
conclusions.

## Results in plain English

The hardware tests used two data-center GPU models: P100 and T4.

| Question | What happened | What that means |
| --- | --- | --- |
| Did the optimized code still give acceptable answers? | Yes, in every registered P100 and T4 test. | The optimization passed the predefined correctness rule on the tested inputs. |
| Did it reduce runtime? | Yes, in all 30 P100 comparisons and all 30 T4 comparisons. | On these GPU models and test sizes, measured latency fell by 58.8%–76.6% on P100 and 72.6%–83.1% on T4. |
| Why did it improve? | The traced case used one custom GPU launch instead of eleven standard launches. | The trace supports the explanation that combining the work removed launch and intermediate-operation overhead. |
| Was multi-GPU communication tested? | Yes, on two T4 GPUs. | The experiment produced useful measurements, but two tools used different setups, so the project does not claim that one was faster. |
| Can someone inspect the evidence? | Yes. Raw timing samples, environments, checksums, failed attempts, and privacy-filtered traces are public. | Reviewers can recalculate the result or submit a counterexample. |

The results are deliberately limited to the recorded FP16 (16-bit number format)
forward operation, six input sizes, and the exact P100 and T4 environments in the
evidence bundles. They do not prove that the kernel will win on every GPU, data type,
model, or future run. The [P100 evidence](results/published/p100-a48afa5/),
[T4 evidence](results/published/t4-a48afa5/), and
[result registry](results/published/README.md) contain the exact boundaries.

## What is inside the repository

| Part | Plain-English purpose |
| --- | --- |
| Reference implementation | Defines the answer that optimized code must match. |
| Custom GPU implementations (CUDA and Triton) | Runs the operation directly on the GPU. |
| Benchmark runner | Checks correctness first, then records repeatable timing samples. |
| Profiler tools | Shows which GPU kernels ran inside the measured operation. |
| Multi-GPU experiment | Measures an all-reduce—a standard operation that combines data across GPUs—at the slowest participating GPU. |
| Performance calculators | Explores idealized data-movement, hardware-limit, and communication estimates without pretending they are measurements. |
| Published evidence | Preserves successful runs, failed attempts, machine details, and checksums. |

See the [architecture note](docs/architecture.md) for the technical data flow and the
[validation record](docs/validation.md) for the complete test history.

## Try it without a GPU

These commands install simple calculators for estimating data movement, hardware
limits, and multi-GPU communication. They do not reproduce the real GPU measurements.

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

### Run the automated developer checks

On Linux x86-64, these commands check the package, reference math, and a CPU simulation
of the custom Triton implementation:

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

## Test communication between GPUs

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
