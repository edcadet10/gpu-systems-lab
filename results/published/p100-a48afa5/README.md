# P100 FP16 residual-RMSNorm evidence

> **Review status:** the pre-registered timing claim held. A separately registered,
> compatibility-pinned Systems trace supports the narrow launch-fusion mechanism: one
> CUDA-extension launch versus eleven eager launches per operation. Hardware-counter,
> achieved-traffic, bandwidth, occupancy, and roofline claims remain unestablished.

## Exact claim and outcome

At clean commit `a48afa5909d0c9ff58934ebaa50e57aff197cdf6`, the explicit FP16
CUDA extension had to pass the v5 numerical relation and reduce median steady-state
latency by at least 5% versus the eager composition for every registered shape in each
of five fresh processes. One failure would retract the grid-wide claim.

All 30 run/shape comparisons met that line. The claim is limited to the hardware,
software, shape, dtype, forward-only operation, and protocol recorded here.

| Shape | Eager median range (ms) | CUDA median range (ms) | Reduction range | Worst error ratio |
|---|---:|---:|---:|---:|
| `128 x 1024` | 0.1731–0.1767 | 0.0412–0.0474 | 72.6%–76.3% | 0.9122 |
| `128 x 4096` | 0.1704–0.1752 | 0.0400–0.0455 | 73.8%–76.6% | 0.9394 |
| `128 x 8192` | 0.1773–0.1885 | 0.0686–0.0731 | 58.8%–61.9% | 0.9559 |
| `1024 x 1024` | 0.1748–0.1833 | 0.0512–0.0547 | 68.7%–71.1% | 0.9595 |
| `1024 x 4096` | 0.5476–0.5673 | 0.1464–0.1510 | 72.6%–73.4% | 0.9708 |
| `1024 x 8192` | 1.0361–1.0701 | 0.2672–0.2702 | 73.9%–74.8% | 0.9641 |

Ranges are the minimum and maximum of the five per-process medians, not population
confidence intervals. Reduction percentages are paired within each process. The
comparison evaluator uses the exact JSON decimal literals.

## Boundary

- GPU: Tesla P100-PCIE-16GB, compute capability 6.0, driver 580.159.04.
- Stack: Python 3.12.13, PyTorch 2.13.0+cu126, CUDA runtime 12.6, CUDA 12.8
  compiler targeting `sm_60`, and cuDNN 9.10.2.
- Workload: FP16, epsilon `1e-6`, rows `{128, 1024}`, hidden sizes
  `{1024, 4096, 8192}`, eager composition versus the explicit CUDA extension.
- Measurement: 5 fresh processes, 25 warmups and 100 retained CUDA-event samples per
  shape/provider/process, base seed 17, schedule seed 2026.
- The synchronized first invocation is retained but excluded from the speed rule.
- The hosted environment did not provide an exclusivity guarantee. Each child retains
  observed clocks, power limit, performance state, and raw samples so reviewers can
  inspect run-to-run variation.
- This result does not establish behavior on another GPU architecture, dtype, shape,
  backward pass, framework compiler, or vendor primitive.

The worst correctness ratio was 0.9708 against the registered failure boundary of
1.0. That narrow margin is material. Earlier v3/v4 absolute-only attempts remain
failed historical experiments; this v5 series was registered separately and does not
reclassify them.

## Profiler result

The corrected Compute command connected to both provider processes, then the host
returned `ERR_NVGPUCTRPERM`; Systems was absent. The four raw logs are retained. No
achieved-bandwidth, occupancy, roofline, or launch-overlap claim is made from this
bundle. Both Compute logs also contain a sitecustomize
`_posixsubprocess` import warning. It did not prevent the profiler from launching and
connecting, but it is another recorded host-environment anomaly rather than evidence
to discard.

The [separate Systems follow-up](profiler/) first falsified a current-tool assumption:
the newer package produced NVTX-only reports because it no longer supports Pascal.
A pre-registered rerun with the last pre-removal release captured CUDA activities.
Correlation inside the exact measured NVTX ranges found 20 launches for 20 explicit
CUDA operations and 220 launches for 20 eager operations. This supports the launch
fusion mechanism only; it does not replace the missing hardware counters.

## Verify

From the repository root:

```bash
(
  cd results/published/p100-a48afa5/raw
  sha256sum --check SHA256SUMS
)
gpu-systems-lab validate-result \
  results/published/p100-a48afa5/raw/rmsnorm-suite/manifest.json
gpu-systems-lab compare-suite \
  results/published/p100-a48afa5/raw/rmsnorm-suite/manifest.json \
  --candidate cuda_extension \
  --baseline torch_eager \
  --minimum-reduction-percent 5
```

Start with the [machine-generated comparison](raw/rmsnorm-suite/cuda-extension-vs-eager.json),
[suite manifest](raw/rmsnorm-suite/manifest.json), [final run status](raw/run_status.json),
and [GPU correctness log](raw/logs/gpu-correctness.log). The `raw/` directory is a
verbatim, checksum-protected import of the retained private runner output; inconvenient
logs and samples were not removed.
