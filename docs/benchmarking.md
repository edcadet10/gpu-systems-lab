# Benchmark and claims policy

Optimization is a hypothesis, not a label. This repository separates four questions:

1. Does the candidate implement the registered numerical contract?
2. Is it faster for the exact measured shape, dtype, device, and software stack?
3. Does profiling support the proposed mechanism?
4. How far can that observation be generalized?

## Required protocol

Every submitted result must record:

- repository commit and the exact command;
- GPU name and compute capability;
- framework, Triton, CUDA runtime, driver, and profiler versions when available;
- shapes, dtype, epsilon, random seed, provider, warmup count, and sample count;
- raw timing samples rather than only an aggregate;
- correctness tolerance and observed maximum absolute error;
- power, clock, and exclusivity settings if they were changed.

The included benchmark uses CUDA events, performs warmups before measurement, and
synchronizes each measured event. PyTorch's benchmark documentation independently
identifies accelerator synchronization, warmups, and replicates as necessary controls.

## Pre-registered acceptance line for performance pull requests

Before running a candidate-versus-baseline comparison, put this statement in the pull
request:

> For every registered shape and dtype, the candidate must pass the existing numerical
> tolerance. Across five fresh processes, its median latency must be at least 5% lower
> than the named baseline in every run. Any failure retracts the broad claim and narrows
> it to the observed cases, if any.

Five runs are a project review rule, not a statistical population guarantee. Include
all five raw result files. Do not discard cold, slow, or inconvenient runs without a
documented measurement fault that applies symmetrically to both providers.

## Profiler check

A byte-traffic explanation needs the `MemoryWorkloadAnalysis` section from Nsight
Compute. A compute-versus-memory explanation needs the roofline section. An overlap or
launch-overhead explanation needs a Systems timeline. Logical models alone are not
evidence of achieved hardware traffic or utilization.

## Comparison rules

- Compare identical tensors, shapes, dtypes, output contracts, and gradient scope.
- Name the baseline precisely: eager composition, compiled composition, vendor
  primitive, or another checked-in implementation.
- Pay setup costs consistently. If compilation or autotuning is amortized, state the
  amortization boundary and also report first-use behavior when relevant.
- Treat an optimized framework compiler as a serious comparator; do not assume a
  handwritten kernel wins.
- Report the observed range. Do not call a finite sample a population tail percentile.
- A result on one accelerator architecture does not establish behavior on another.

## Distributed measurements

Collective latency is a critical-path property. The included NCCL benchmark reports
the slowest rank for each iteration. Record topology (`nvidia-smi topo -m`), interface,
node count, process placement, and relevant NCCL environment variables alongside any
submitted multi-node result. The ring model is a sanity check, not a predictor of
NCCL's topology-aware algorithm and protocol choices.
