# P100 profiler follow-up

> **Review status:** the launch-fusion mechanism held under a compatibility-pinned
> Systems trace. Inside the exact 20-operation NVTX ranges, the CUDA extension issued
> 20 launches and eager composition issued 220. No hardware-counter, achieved-traffic,
> bandwidth, occupancy, roofline, or tensor-core claim is made.

## Registered protocol

Both attempts profiled clean source commit
`a48afa5909d0c9ff58934ebaa50e57aff197cdf6` on one Tesla P100-PCIE-16GB:

- FP16 residual-RMSNorm at `1024 x 4096`;
- eager and explicit CUDA-extension providers in separate processes;
- 5 warmups followed by 20 operations inside one named NVTX push/pop range;
- `cuda,nvtx,osrt` tracing with CPU sampling and context switches disabled; and
- provider-specific outputs, exact package SHA-256, environment logs, source-state
  checks, report digests, and transient cleanup.

## Attempt v1: valid files, invalid mechanism evidence

The SHA-pinned 2026.2.1 CLI created two nonempty reports and returned success, meeting
the original artifact-level criterion. It did not meet the data-level criterion:
both CUDA kernel summaries were skipped. Exported diagnostics showed
`CUPTI_ERROR_INVALID_DEVICE`, zero CUPTI events, and one NVTX range per report.

The tool's [official release notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html)
explain the boundary: Systems versions starting with 2025.4 do not support Pascal or
Volta. The complete non-sensitive text evidence is under [`v1/`](v1/). This attempt is
a negative mechanism result even though its runner status is `success: true`.

## Attempt v2: compatible CUDA timeline

The rerun used official Systems 2025.3.1, the last release before that architecture
removal. Its package SHA-256 was
`d2484ad0faf6831b11fa0bf73c54232d9ea8beafb50414019e6ba299c4ed5718`.
The runner was strengthened before execution: a skipped CUDA summary, zero kernel
rows, or missing NVTX range failed the run.

| Provider | Operations in NVTX range | Correlated launches | Launches per operation | Unique kernel names |
|---|---:|---:|---:|---:|
| Eager composition | 20 | 220 | 11 | 8 |
| CUDA extension | 20 | 20 | 1 | 1 |

Across the full processes, including tensor initialization and five warmups, the
reports contain 278 eager kernel rows and 28 CUDA-extension rows. The candidate's 25
operation rows all name the same explicit residual-RMSNorm kernel; the other three
rows initialize input tensors.

The checked-in [`analyze-nsys`](../../../../README.md#profile-the-kernel) command derives
the range-scoped counts. It selects the one named NVTX interval, takes CUDA runtime
launch calls on that interval's thread, and joins kernels by correlation ID. The
[eager](v2/correlated/torch.json) and
[CUDA-extension](v2/correlated/cuda_extension.json) JSON summaries bind themselves to
the exact privacy-filtered SQLite SHA-256 values in the public asset manifest; the
runner record separately binds the source reports and full exports.

This evidence supports a narrow statement: for this operation, shape, dtype, device,
and software stack, the explicit provider fuses the eager launch sequence into one
CUDA kernel invocation. It does not identify the P100 timing result's achieved memory
traffic or utilization. The NVTX duration records host submission, and the profiler
perturbs execution; it is not substituted for the separately collected CUDA-event
latency result.

## Public trace assets and privacy boundary

A pre-publication string scan found ephemeral hosted-runner credentials in the raw
binary reports and their full SQLite exports. Those files are deliberately not
published. Their SHA-256 values and sizes remain in
[`profiles.json`](v2/profiles.json) and the runner manifests so the provenance record
cannot silently change.

The release contains only two privacy-filtered SQLite files with the four tables
needed by `analyze-nsys`. Their `StringIds` tables retain only rows referenced by the
activity tables; `VACUUM` removes deleted pages. The release assets are:

- [eager filtered SQLite](https://github.com/edcadet10/gpu-systems-lab/releases/download/evidence-a48afa5/p100-profiler-v2-torch.sqlite);
- [CUDA-extension filtered SQLite](https://github.com/edcadet10/gpu-systems-lab/releases/download/evidence-a48afa5/p100-profiler-v2-cuda-extension.sqlite); and
- [asset checksums](https://github.com/edcadet10/gpu-systems-lab/releases/download/evidence-a48afa5/SHA256SUMS).

[`public-assets.json`](public-assets.json) records the public and source digests,
sizes, table allowlist, exporter package digest, and transformation. The exact
post-export filter is checked in as
[`sanitize-public-export.sql`](sanitize-public-export.sql), and the release digests
are also mirrored in [`public-SHA256SUMS`](public-SHA256SUMS).

## Boundary and retained warnings

- Hardware: Tesla P100-PCIE-16GB, compute capability 6.0, driver 580.159.04.
- Runtime: Python 3.12.13, PyTorch 2.13.0+cu126, CUDA runtime 12.6, CUDA 12.8
  compiler targeting `sm_60`.
- v2 warns that the installed driver advertises CUDA 13.0 while the compatible tool
  uses 12.9 trace libraries. It nevertheless records nonzero CUDA API/kernel data for
  the target process. No severity-3 diagnostic occurs in v2.
- CPU scheduling data is absent because the protocol explicitly disabled CPU sampling
  and context-switch tracing. No CPU scheduling or overlap claim is made.
- CUDA-extension child processes used for build orchestration emit no-CUDA notices;
  the target process contains the expected NVTX range and kernel activities.

## Verify

After downloading the v2 SQLite assets:

```bash
gpu-systems-lab analyze-nsys p100-profiler-v2-torch.sqlite \
  --nvtx-range residual_rmsnorm:torch
gpu-systems-lab analyze-nsys p100-profiler-v2-cuda-extension.sqlite \
  --nvtx-range residual_rmsnorm:cuda_extension
sha256sum --check SHA256SUMS
```

The committed tests also verify every retained text artifact against its original
runner manifest, require the omitted paths to be exactly the sensitive binaries and
full exports, bind the correlated JSON to the filtered SQLite digests, exercise the
string-pruning SQL, and assert the 20-versus-220 launch result.
