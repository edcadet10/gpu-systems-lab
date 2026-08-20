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

## Profiling and performance models

The [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
defines arithmetic intensity and the memory and compute ceilings in a roofline chart.
The [Nsight Compute user guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
documents its memory-workload, occupancy, launch, and hierarchical roofline sections.
The [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
covers the execution, memory, stream, event, graph, and multi-GPU abstractions below
the framework. The
[Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
documents CUDA and NVTX tracing.

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

**Design implication:** a ring alpha-beta model and measured all-reduce form a useful
minimum pair for reasoning about communication volume and critical-path latency. A
numerical correctness decision is reduced across ranks before any rank raises, so a
local mismatch does not intentionally desynchronize subsequent collectives.

**Disconfirming evidence:** NCCL selects algorithms and protocols using actual
topology; a homogeneous ring formula omits trees, hierarchical routes, contention,
reduction work, channelization, and overlap. It cannot validate a multi-node design.

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
