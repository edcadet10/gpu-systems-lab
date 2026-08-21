# Research basis and disconfirming evidence

This note records the primary sources used to choose the first experiment and, more
importantly, the evidence that limits each design claim. Links point to official
documentation, project repositories, standards, or original papers.

## Kernel fusion and IO

The [FlashAttention paper](https://papers.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)
shows why operation count alone can miss the performance impact of reads and writes
between HBM and on-chip memory. Triton's official
[fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
demonstrates the same principle in a compact kernel, while its
[LayerNorm tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py)
provides a tested row-reduction structure and a 64 KiB feature boundary. The
[RMSNorm paper](https://arxiv.org/abs/1910.07467) defines the normalization used here.

**Design implication:** residual-plus-RMSNorm is small enough to audit yet exercises
masked loads, reduction, precision promotion, and fusion.

**Disconfirming evidence:** lower logical traffic does not prove lower DRAM traffic or
latency. Cache reuse, register spills, launch cost, and compiler-generated fusion can
erase the expected advantage. Nsight measurement and a compiled framework baseline
are therefore required before any speed statement.

## Framework integration and timing

PyTorch's current
[user-defined Triton kernel guide](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
documents direct kernels, `torch.library.triton_op`, compiler visibility, fallbacks,
and subsystem composition. The direct launch used here keeps the first implementation
small; a structured operator becomes justified when tensor-subclass, export, or other
dispatcher behavior is added. PyTorch's
[benchmark utility documentation](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
calls out warmups, accelerator synchronization, and replicate variation.
Triton's
[testing API](https://triton-lang.org/main/python-api/triton.testing.html)
provides GPU benchmarking utilities and quantile summaries. This repository instead
retains every event sample so review is not limited to precomputed aggregates.
The CUDA Runtime
[event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
defines event recording, synchronization, and elapsed-time calculation; it also warns
that unrelated stream work between events can inflate the observed interval.
Triton's official
[debugging guide](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
documents that interpreter mode simulates operations sequentially with NumPy on the
CPU and bypasses target compilation.

**Design implication:** correctness runs before timing; the output retains raw samples
and environment metadata; `torch.compile` is available as a comparator.

**Disconfirming evidence:** custom kernels can lose to compiled or vendor operations,
especially for shapes with little work per launch. The repository treats such a loss
as a useful result rather than filtering the shape. Interpreter agreement supports
the operation-level contract but cannot establish GPU code generation or latency.
CUDA events do not make a shared or contended device exclusive, so device isolation
and a retained raw distribution remain part of the measurement boundary.

PyTorch's official
[`torch.utils.cpp_extension` documentation](https://docs.pytorch.org/docs/stable/cpp_extension.html)
defines the just-in-time CUDA build path used by the explicit extension provider.
NVIDIA's [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
prioritizes coalesced global access, while the
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
defines the synchronized warp shuffles used by its reduction. Current Triton lists
NVIDIA compute capability 8.0 and later in its
[compatibility boundary](https://github.com/triton-lang/triton#compatibility); the CUDA
provider exists in part to make older-device experiments explicit rather than
silently treating interpreter success as target support.
PyTorch's maintained
[release compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)
records that its CUDA 12.6 build retains Pascal compute capability 6.0 while CUDA 13
starts at Turing 7.5. The official
[version installer](https://pytorch.org/get-started/previous-versions/) exposes the
CUDA 12.6 wheel index used to make that choice explicit.

**Design implication:** the CUDA source ships in the wheel, compiles lazily for the
visible architecture, uses the current PyTorch stream, and is limited to contiguous
FP16 forward tensors. Compiler/toolkit identity and the first-use build cost remain
part of the retained environment and timing evidence. Pascal experiments pin the
CUDA 12.6 framework build rather than relying on the default package index.

**Disconfirming evidence:** coalescing and fusion do not guarantee a win. The kernel
loads input and residual twice, a block owns an entire row, and runtime compilation
introduces a cache-sensitive setup cost. Correctness, isolated timing, and hardware
profiling must decide whether those tradeoffs hold on each recorded GPU.
The first P100 suite attempt at commit `9c3c52e` compiled but exceeded its registered
FP16 absolute-error gate on shape `(1024, 4096)` (`0.00390625 > 0.002`) before timing.
A compensated-summation replacement at `c3fcf8b` still found one differing FP16 element
among 4,194,304, with the same one-step absolute difference and relative error
`0.0006823539`; it also stopped before timing. Both absolute-gate claims remain
retracted.

Those failures exposed a contract problem separately from a kernel problem: FP16
spacing scales with magnitude, so one representable step can exceed a fixed absolute
threshold. PyTorch's official
[`assert_close` documentation](https://docs.pytorch.org/docs/stable/testing) instead
defines a combined absolute-relative relation and publishes strict defaults per dtype.
The v5 report adopts that relation, records the maximum error-to-allowance ratio, and
retains v3/v4 validation so failed historical runs are never reinterpreted.

**New-series hardware result:** after the v5 rule was publicly registered, the
canonical P100 rerun at commit `a48afa5` passed all six shapes in each of five fresh
processes. Its worst error-to-allowance ratio was `0.9707606881`, close to the failure
line of one. All 30 paired timing decisions also cleared the registered 5% reduction
line versus eager composition; observed per-shape reduction ranges across the five
medians were 58.8%–76.6%. The complete
[candidate bundle](../results/published/p100-a48afa5/) retains every raw sample and
the exact comparison decision.

That timing bundle alone does not establish the proposed mechanism. The corrected
Nsight Compute command reached the target but the hosted platform denied GPU
performance-counter access, and the image did not ship Systems. Those failures are
retained. A separately registered, SHA-pinned Systems follow-up first produced NVTX
but zero CUDA events. The tool's official
[release notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#deprecated-features)
then falsified the selection assumption: releases starting with 2025.4 no longer
support Pascal or Volta. A pre-registered 2025.3.1 rerun captured nonzero CUDA
activity. Correlation inside the exact 20-operation NVTX ranges found 20 launches for
the explicit provider and 220 for eager composition. This supports launch fusion on
the traced P100 shape, but still does not establish achieved bytes, bandwidth,
occupancy, or roofline position.

**Second-generation result:** the same registered FP16 rule held in all 30 decisions
on one GPU of a dual-T4 allocation at commit `a48afa5`; per-shape reduction ranges
were 72.6%–83.1%, and the worst error-to-allowance ratio was `0.9688795876`. The
[complete T4 bundle](../results/published/t4-a48afa5/) also retains matching
element-zero witnesses across the two project ranks and `nwrong=0`
out-of-place/in-place output checks from pinned `nccl-tests`. The measured-commit
project runner did not inspect every output
element, so no full-tensor correctness claim is attached to that historical result;
the current runner closes that gap with a full-tensor extrema check.

Those collective timings are diagnostics, not a comparison. The project uses two
processes with one GPU each and reports the maximum rank latency per iteration; the
upstream run uses one process and one thread driving two GPUs. The successful
allocation also selected different NCCL direct-mode behavior from two earlier
allocations. The registered escape condition therefore applies: without aligned
process and timing semantics, no agreement percentage or speed claim is computed.
Three failed runner attempts—wrong allocation, incompatible NCCL link, and a stripped
driver-library path—remain beside the successful evidence rather than being omitted.

## Reproducibility and report contracts

NVIDIA's
[`nvidia-smi` documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
defines selective GPU queries for driver, performance-state, clock, and power fields.
The [JSON Schema 2020-12 specification](https://json-schema.org/draft/2020-12)
defines the report-validation dialect, while Python's
[`importlib.resources` documentation](https://docs.python.org/3.11/library/importlib.resources.html)
provides installed-package resource access independent of a source checkout.

**Design implication:** schemas ship in the wheel; reports distinguish unavailable
management data from query errors; publication-mode suites require a clean commit;
and one command recursively checks the manifest and all child reports.

**Disconfirming evidence:** a valid schema proves structure, not provenance or honest
measurement. A clean commit does not prove an uncontended GPU. Clock and power queries
are point observations, not full-run telemetry. Process isolation and randomized order
reduce two sources of bias but do not eliminate thermal drift, system noise, or
selection bias. The management documentation also notes that numeric device ordering
is not stable across reboots; the report's device index is therefore local context,
not a persistent hardware identity. Independent reproduction and profiler evidence
remain necessary.

## Profiling and performance models

The [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
defines arithmetic intensity and the memory and compute ceilings in a roofline chart.
The [Nsight Compute user guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
documents its memory-workload, occupancy, launch, and hierarchical roofline sections.
The [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
covers the execution, memory, stream, event, graph, and multi-GPU abstractions below
the framework. The
[Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
documents CUDA and NVTX tracing. Its
[post-collection guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#common-sqlite-examples)
defines the runtime-to-kernel correlation-ID join and the serialized HW/VM/PID/TID
identifier layout used by the checked-in range analyzer.

**Design implication:** the model exposes its units and assumptions, while scripts
collect the hardware counters needed to challenge it.

**Disconfirming evidence:** a roofline ceiling is not achieved throughput, and a
logical tensor-byte count is not a hardware counter. Both remain hypotheses until
measured on the target device.

## Collectives and parallelism

The current [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
documents topology-aware multi-GPU collectives, stream semantics, communicator
lifecycle, asynchronous errors, and point-to-point operations. The original
[Megatron-LM project report](https://research.nvidia.com/labs/adlr/MegatronLM/)
shows how tensor parallelism places targeted collectives around transformer layers.
The maintained
[parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
describes tensor, pipeline, context, and data parallel choices.
PyTorch's official
[distributed communication documentation](https://docs.pytorch.org/docs/stable/distributed)
states that default-group collectives require every process to enter the call and
documents process-group timeouts and coordinated shutdown requirements.
NVIDIA's maintained
[`nccl-tests` repository](https://github.com/NVIDIA/nccl-tests)
provides standard collective correctness and bandwidth comparators across processes,
threads, and MPI launch configurations, including per-iteration event timing and JSON
output.

**Design implication:** a ring alpha-beta model and measured all-reduce form a useful
minimum pair for reasoning about communication volume and critical-path latency. A
numerical correctness decision is reduced across ranks before any rank raises, so a
local mismatch does not intentionally desynchronize subsequent collectives.

**Disconfirming evidence:** NCCL selects algorithms and protocols using actual
topology; a homogeneous ring formula omits trees, hierarchical routes, contention,
reduction work, channelization, and overlap. It cannot validate a multi-node design.
The repository's custom benchmark should be compared with `nccl-tests` before its
measurements are used to diagnose a communication mechanism.

## Low precision and tensor cores

The current [Transformer Engine FP8 guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
explains tensor and block scaling, amax histories, format selection, transpose costs,
and operation eligibility. Its
[precision benchmark guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/speedups.html)
explicitly reports that speedup is shape-dependent and that quantization overhead can
make small GEMMs slower. The
[OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
defines interoperable MX formats. The
[CUTLASS repository and profiler](https://github.com/NVIDIA/cutlass)
cover architecture-specific tiled GEMM, mixed precision, verification, and targeted
kernel profiling.

**Design implication:** the first release exercises mixed storage and accumulation
precision but postpones FP8 until format, scaling, accuracy, and target-hardware tests
can ship together.

**Disconfirming evidence:** narrower storage does not independently establish faster
end-to-end execution or acceptable model quality. Pre-quantized GEMM timing cannot be
presented as application speedup.

## Public GPU automation

GitHub's [secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use)
warns that untrusted pull requests can persistently compromise self-hosted runners and
states that they should almost never be attached directly to public repositories.

**Design implication:** public pull requests run CPU checks on ephemeral hosted
runners. GPU results are produced on isolated infrastructure, attached as data, and
reviewed; this repository does not expose a long-lived GPU runner to forked code.
