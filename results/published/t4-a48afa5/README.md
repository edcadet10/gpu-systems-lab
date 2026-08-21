# Dual-T4 FP16 evidence

> **Review status:** the pre-registered single-GPU RMSNorm timing claim held. The
> project collective's element-zero witness matched, and both pinned-upstream modes
> recorded `nwrong=0`. Their process models differ, so the registered cross-tool
> comparison is **not evaluated** and no collective speed or agreement claim is made.

## RMSNorm claim and outcome

At clean commit `a48afa5909d0c9ff58934ebaa50e57aff197cdf6`, the explicit FP16
CUDA extension had to pass the v5 numerical relation and reduce median steady-state
latency by at least 5% versus eager composition for every registered shape in each of
five fresh processes. One failure would retract the grid-wide claim.

All 30 run/shape comparisons met that line on GPU 0 of the successful dual-T4
allocation.

| Shape | Eager median range (ms) | CUDA median range (ms) | Reduction range | Worst error ratio |
|---|---:|---:|---:|---:|
| `128 x 1024` | 0.1659–0.1776 | 0.0410–0.0435 | 74.6%–76.5% | 0.9122 |
| `128 x 4096` | 0.1678–0.1764 | 0.0413–0.0451 | 73.5%–76.6% | 0.9141 |
| `128 x 8192` | 0.3215–0.3338 | 0.0841–0.0881 | 72.6%–74.8% | 0.9510 |
| `1024 x 1024` | 0.3154–0.3285 | 0.0659–0.0702 | 77.9%–79.6% | 0.9632 |
| `1024 x 4096` | 1.1199–1.1450 | 0.1901–0.2028 | 82.1%–83.1% | 0.9624 |
| `1024 x 8192` | 2.1156–2.1207 | 0.4281–0.4480 | 78.8%–79.8% | 0.9689 |

Ranges are the minimum and maximum of five per-process medians. They are not
population confidence intervals. Reduction percentages are paired within each
process, and the comparison evaluator consumes the exact JSON decimal literals.

## Collective diagnostics

Both diagnostics used two Tesla T4 devices, a 128 MiB FP16 all-reduce, 10 warmups,
and 50 retained iterations on the recorded PHB topology:

| Diagnostic | Execution model | Raw median | Observed range | Validation witness |
|---|---|---:|---:|---|
| Project benchmark | Two processes, one GPU per process; maximum rank latency for each sample | 18.8139 ms | 18.5934–19.4683 ms | Element zero: expected `2.0`, observed `2.0` |
| Pinned upstream, out of place | One process, one thread, two GPUs | 23.4521 ms | 23.1946–23.6773 ms | `nwrong=0` |
| Pinned upstream, in place | One process, one thread, two GPUs | 23.4844 ms | 23.2584–23.6974 ms | `nwrong=0` |

The medians above are independently recomputed from the retained 50-element arrays.
The upstream tool reports an upper-middle `p50` for an even sample count; that value
is also preserved but is not substituted for the conventional two-middle-value
median in this table.

The timings are deliberately not divided into a speedup or agreement percentage.
The [registered environment record](raw/collective-environment.json) states that the
process models differ, and the logs show different NCCL direct-mode decisions. The
project-only diagnostic also changed from about 33.3 ms in attempts v2/v3 to 18.8 ms
in v4 despite the same nominal GPU model and PHB topology. That observed allocation
variation is a reason to narrow the claim, not a reason to discard earlier runs.

The upstream JSON writer's non-MPI device record contains only its first parsed rank;
the [unmodified stdout log](raw/logs/allreduce-nccl-tests.log) enumerates both ranks,
while the runner's independent driver preflight and project report each require two
devices. The upstream source is pinned to
[`717b683`](https://github.com/NVIDIA/nccl-tests/tree/717b68318278e93f371d8ffb46b076069d7c7851),
and the run records exact driver and NCCL binary digests.

## Boundary

- Hardware: two Tesla T4 GPUs, compute capability 7.5, 15,360 MiB each, PHB path,
  driver 580.159.04, no NVLink, and four visible CPU cores in one NUMA node.
- Stack: Python 3.12.13, PyTorch 2.13.0+cu126, CUDA runtime 12.6, CUDA 12.8
  compiler targeting `sm_75`, Triton 3.7.1, and NCCL 2.29.3.
- RMSNorm: FP16 forward only, epsilon `1e-6`, rows `{128, 1024}`, hidden sizes
  `{1024, 4096, 8192}`, 5 fresh processes, 25 warmups, and 100 retained CUDA-event
  samples per shape/provider/process.
- Collective: one node, 128 MiB FP16, 10 warmups, 50 retained iterations, and the
  execution models named above. It is not a multi-node, overlap, saturation, or
  fault-tolerance result.
- At the measured commit, the project runner checked element zero on every rank before
  consensus, not the full output tensor. The current runner now checks full-tensor
  extrema, but that later fix does not upgrade this historical witness.
- The host did not provide an exclusivity guarantee. No claim is made about another
  GPU generation, dtype, shape, backward pass, end-to-end serving, or future run.
- The bundled profiler attempts did not produce usable reports, so no T4 bandwidth,
  occupancy, roofline, or kernel-mechanism claim is made.

## Failed-attempt ledger

| Attempt | Stop condition | What was retained |
|---|---|---|
| [v1](attempts/v1/) | The requested selector allocated one P100; exact T4 preflight stopped before measurement | Allocation, runtime, logs, failure, checksums |
| [v2](attempts/v2/) | Both T4 measurements completed; pinned upstream link selected an incompatible NCCL binary | Complete RMSNorm/project data plus build failure |
| [v3](attempts/v3/) | Exact NCCL linkage succeeded; stripped driver search path stopped upstream device discovery | Complete RMSNorm/project data, linker proof, partial upstream JSON, failure |
| [v4](raw/) | Completed | Complete successful bundle |

The retained [v1](attempts/v1/), [v2](attempts/v2/), and [v3](attempts/v3/)
bundles record each failure before the next repair. Each repair changed allocation or
runner infrastructure only; the registered benchmark grids and iteration counts did
not change.

## Verify

From the repository root:

```bash
for attempt in results/published/t4-a48afa5/attempts/v{1,2,3}; do
  (cd "$attempt" && sha256sum --check SHA256SUMS)
done
(
  cd results/published/t4-a48afa5/raw
  sha256sum --check SHA256SUMS
)
gpu-systems-lab validate-result \
  results/published/t4-a48afa5/raw/rmsnorm-suite/manifest.json
gpu-systems-lab validate-result \
  results/published/t4-a48afa5/raw/allreduce-project.json
gpu-systems-lab compare-suite \
  results/published/t4-a48afa5/raw/rmsnorm-suite/manifest.json \
  --candidate cuda_extension \
  --baseline torch_eager \
  --minimum-reduction-percent 5
```

Start with the [final status](raw/run_status.json),
[machine-generated RMSNorm comparison](raw/rmsnorm-suite/cuda-extension-vs-eager.json),
[project collective report](raw/allreduce-project.json),
[upstream JSON](raw/allreduce-nccl-tests.json), and
[topology log](raw/logs/topology.log). The successful `raw/` directory and each failed
attempt are verbatim, checksum-protected imports; inconvenient samples and logs were
not removed.
