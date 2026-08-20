# Benchmark and claims policy

Optimization is a hypothesis, not a label. This repository separates four questions:

1. Does the candidate implement the registered numerical contract?
2. Is it faster for the exact measured shape, dtype, device, and software stack?
3. Does profiling support the proposed mechanism?
4. How far can that observation be generalized?

## Canonical single-GPU protocol

Use `gpu-lab-rmsnorm-suite` for a comparison intended for publication. It creates a
deterministic randomized schedule and launches one provider per fresh Python process.
For every process-level run, providers receive the same seed and shape grid. Seeds
vary between runs. Each child:

1. constructs the registered tensors;
2. executes the eager reference;
3. executes the named provider once and records host-wall-clock first-use latency with
   device synchronization;
4. rejects non-finite or over-tolerance output;
5. performs warmups; and
6. retains every measured CUDA-event sample.

The first-use observation is diagnostic. For non-eager providers it follows one eager
reference invocation, so it captures provider lazy work such as compilation without
claiming full process cold-start latency. It is not used in the steady-state acceptance
line below.

The suite refuses publication-mode execution unless Git resolves to a clean commit.
`--allow-unversioned` exists for local debugging only. The output directory must not
already exist, preventing accidental overwrite or mixture with an earlier run.

## Required metadata and validation

Every submitted result must retain:

- repository commit, dirty state, canonical replay commands, and schedule seed;
- GPU name/index, compute capability, memory size, framework, Triton, CUDA runtime,
  cuDNN, driver, Python, operating system, kernel, and machine architecture;
- observed performance state, SM/memory clocks, and power limit when the management
  query is available, plus an explicit query status when it is not;
- shapes, dtype, epsilon, random seed, provider, warmup count, and sample count;
- raw timing samples rather than only an aggregate;
- correctness tolerance and observed maximum absolute error.

Run the recursive validator on the manifest:

```bash
gpu-systems-lab validate-result path/to/suite/manifest.json
```

It validates all JSON Schemas and also checks complete schedule coverage, unique and
confined child paths, canonical replay commands, commit/dirty-state agreement, and each
child's seed, provider, dtype, and shape grid. A `valid: true` result establishes
structural and internal consistency only; it does not validate hardware behavior or a
performance claim.

## Pre-registered acceptance line for performance pull requests

Before running a candidate-versus-baseline comparison, put this statement in the pull
request:

> For every registered shape and dtype, the candidate must pass the existing numerical
> tolerance. Across five fresh processes, its median latency must be at least 5% lower
> than the named baseline in every run. Any failure retracts the broad claim and narrows
> it to the observed cases, if any.

Five runs are a project review rule, not a statistical population guarantee. Include
the complete suite. Do not discard cold, slow, or inconvenient runs without a
documented measurement fault that applies symmetrically to every provider.

## Comparison rules

- Compare identical tensors, shapes, dtypes, output contracts, and gradient scope.
- Name the baseline precisely: eager composition, compiled composition, vendor
  primitive, or another checked-in implementation.
- Randomize measurement order to reduce temporal bias, but preserve the registered
  schedule even when it is inconvenient.
- Pay setup costs consistently. If compilation or autotuning is amortized, state the
  amortization boundary and retain the first-use observation.
- Treat an optimized framework compiler as a serious comparator; do not assume a
  handwritten kernel wins.
- Report the observed range. Do not call a finite sample a population tail percentile.
- A result on one accelerator architecture does not establish behavior on another.
- Keep losing shapes and negative results in the reviewed result set.

## Profiler check

A byte-traffic explanation needs the `MemoryWorkloadAnalysis` section from Nsight
Compute. A compute-versus-memory explanation needs the roofline section. An overlap or
launch-overhead explanation needs a Systems timeline. Logical models alone are not
evidence of achieved hardware traffic or utilization. Collect profiler data separately
from latency samples because instrumentation perturbs execution.

## Publication gate

Local runs belong under ignored `results/local/`. A reviewed measurement belongs in a
new subdirectory of `results/published/` with the suite manifest and all referenced raw
reports. Before merging it:

- validate the bundle from a clean checkout;
- reproduce the registered comparison or explain why independent reproduction is not
  available;
- attach or archive profiler reports and summarize the counters in reviewable text;
- identify the exact scope that passed and every registered case that failed; and
- state hardware, software, clock/power, exclusivity, and topology boundaries.

Binary profiler reports should be attached to an issue or archival release rather than
committed to Git. The repository currently contains no accepted hardware measurement.

## Distributed measurements

Collective latency is a critical-path property. The included NCCL benchmark reports
the slowest rank for each iteration. Record topology (`nvidia-smi topo -m`), interface,
node count, process placement, and relevant NCCL environment variables alongside any
submitted multi-node result. The ring model is a sanity check, not a predictor of
NCCL's topology-aware algorithm and protocol choices.
