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

### Numerical acceptance rule

Current reports apply the same finite, elementwise relation documented by
[`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing):

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

The dtype defaults are FP16 `(rtol=1e-3, atol=1e-5)`, BF16
`(rtol=1.6e-2, atol=1e-5)`, and FP32 `(rtol=1.3e-6, atol=1e-5)`. Each result records
the maximum absolute error plus the maximum elementwise error-to-allowance ratio. It
also records the reference magnitude at the maximum-absolute-error element and the
absolute error and reference magnitude at the maximum-ratio element. Those compact
witnesses let the offline validator recompute both decisions without retaining entire
output tensors. A non-finite metric or ratio above one stops the child before timing.
The exact framework defaults are used as an external contract rather than tuning a
threshold to this kernel. The shared `1e-5` absolute floor is therefore intentional;
the dtype-specific relative term provides the scaling. This rule prevents a fixed
absolute threshold from becoming smaller than one representable low-precision step at
larger magnitudes while remaining strict near zero.

RMSNorm v3/v4 reports retain their original absolute-only meaning. They are historical
evidence and are never reclassified under the v5 rule.

Version 5 governs a new experiment series registered after the earlier absolute-only
claims were retracted. Passing v5 does not reverse or relabel either historical result.

## Required metadata and validation

Every submitted result must retain:

- repository commit, dirty state, canonical replay commands, and schedule seed;
- GPU name/index, compute capability, memory size, framework, Triton, CUDA runtime,
  CUDA compiler/effective architecture list, cuDNN, driver, Python, operating system,
  kernel, and machine architecture;
- observed performance state, SM/memory clocks, and power limit when the management
  query is available, plus an explicit query status when it is not;
- shapes, dtype, epsilon, random seed, provider, warmup count, and sample count;
- raw timing samples rather than only an aggregate;
- correctness formula, absolute and relative tolerances, observed maximum absolute
  error, and maximum error-to-allowance ratio.

Run the recursive validator on the manifest:

```bash
gpu-systems-lab validate-result path/to/suite/manifest.json
```

It validates all JSON Schemas and also checks complete schedule coverage, unique and
confined child paths, canonical replay commands, commit/dirty-state agreement, and each
child's seed, provider, dtype, shape grid, and stable device/software identity. A
`valid: true` result establishes structural and internal consistency only; it does not
validate hardware behavior or a performance claim.

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

Apply the line without selectively computing only favorable cases:

```bash
gpu-systems-lab compare-suite path/to/suite/manifest.json \
  --candidate triton \
  --baseline torch_eager \
  --minimum-reduction-percent 5 \
  --output path/to/suite/triton-vs-eager.json
```

The evaluator reports the baseline median, candidate median, and maximum permitted
candidate latency as exact decimal strings for each registered run, dtype, and shape.
It deliberately does not add an average speedup or a population-tail estimate. It
refuses dirty, unversioned, fewer-than-five-run suites, and non-positive medians that
cannot support a percentage comparison. Exact float literals are limited to 128
characters so externally supplied bundles cannot force unbounded decimal arithmetic.
`--fail-on-retract` is available for automation; by default a valid negative result
exits successfully so it can be retained and reviewed.

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

A nonempty report and zero exit status are not sufficient. Confirm that the selected
tool release supports the target architecture, run the requested statistics, and
inspect the exported activity tables. For a range-scoped launch claim, use
`gpu-systems-lab analyze-nsys` and retain the SQLite digest, exact NVTX name, correlated
kernel count, and any collection diagnostics. NVTX host duration is not synchronized
kernel latency unless the target explicitly makes it so.

## Publication gate

Local runs belong under ignored `results/local/`. A reviewed measurement belongs in a
new subdirectory of `results/published/` with the suite manifest and all referenced raw
reports. Before merging it:

- validate the bundle from a clean checkout;
- reproduce the registered comparison or explain why independent reproduction is not
  available;
- publish only the minimum privacy-scanned profiler derivative needed to reproduce
  the claim, while retaining source digests and reviewable summaries;
- identify the exact scope that passed and every registered case that failed; and
- state hardware, software, clock/power, exclusivity, and topology boundaries.

Profiler reports and full exports can embed environment strings, including ephemeral
credentials. Keep them private until a content scan proves otherwise; prefer an
allowlisted, vacuumed SQLite derivative for a public release. The registry labels each
hardware bundle as candidate, accepted, or retracted; publication alone does not imply
acceptance.

## Distributed measurements

Collective latency is a critical-path property. The included NCCL benchmark reports
the slowest rank for each iteration. Record topology (`nvidia-smi topo -m`), interface,
node count, process placement, and relevant NCCL environment variables alongside any
submitted multi-node result. Before making an agreement or speed claim across tools,
match the process/thread/GPU model, in-place versus out-of-place semantics, timer and
rank aggregation, message size, dtype, warmups, and retained iterations. When those
semantics differ, keep both measurements as diagnostics and label the cross-tool
comparison not evaluated. The ring model is a sanity check, not a predictor of NCCL's
topology-aware algorithm and protocol choices.
