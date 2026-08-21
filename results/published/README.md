# Published measurements

| Evidence | Numerical gate | Timing claim | Mechanism claim | Review state |
|---|---|---|---|---|
| [P100 FP16 residual-RMSNorm](p100-a48afa5/) | Held | Held on the registered grid | Launch fusion held; hardware-counter claims unestablished | Candidate |
| [Dual-T4 FP16 RMSNorm and collectives](t4-a48afa5/) | Held | RMSNorm held; collective comparison not evaluated | Pending profiler evidence | Candidate |

Each result set has a dedicated directory containing its exact boundary, claim status,
suite manifest, every referenced child report, raw samples, environment context, and
checksum manifest. A timing result does not silently become a mechanism result: when
the required profiler is unavailable, the failed collection is retained and the
mechanism column remains pending or retracted.

The commit embedded in a bundle is the clean source revision that was measured; a
later review-only commit may publish that immutable bundle without changing its
provenance.

Simulated, CPU-only, interpreter, and analytical-model observations are never
substituted for target-GPU measurements. See the root [result registry](../README.md)
and [benchmark policy](../../docs/benchmarking.md).
