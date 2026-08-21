# Result registry

This directory separates report contracts from measured evidence.

- `rmsnorm.schema.json` and `allreduce.schema.json` are the historical v2 contracts
  shipped in v0.1.1. Their URLs remain stable for old reports.
- The current, package-shipped contracts are
  [`rmsnorm-v5.schema.json`](../src/gpu_systems_lab/schemas/rmsnorm-v5.schema.json),
  [`allreduce-v3.schema.json`](../src/gpu_systems_lab/schemas/allreduce-v3.schema.json),
  and
  [`rmsnorm-suite-v3.schema.json`](../src/gpu_systems_lab/schemas/rmsnorm-suite-v3.schema.json).
- Package-shipped RMSNorm v3/v4 and suite v1/v2 remain valid historical contracts.
  Their absolute-only correctness decisions are not reinterpreted under v5's
  scale-aware rule.
- `local/` is ignored scratch space for new runs.
- `published/` is the reviewed evidence registry. It contains complete FP16
  residual-RMSNorm candidates for [P100](published/p100-a48afa5/) and
  [T4](published/t4-a48afa5/), including all process runs, negative attempts,
  collective diagnostics, and retained profiler limitations.

Validate a current report or complete suite before review:

```bash
python -m pip install -e '.[reports]'
gpu-systems-lab validate-result path/to/report-or-manifest.json
```

For a suite manifest, the validator follows every relative child path and checks the
complete schedule, path confinement, canonical replay command, commit and dirty state,
seed, provider, dtype, shape grid, and each child's JSON Schema contract.

See [the benchmark and claims policy](../docs/benchmarking.md) for the publication
gate. A schema-valid file is necessary evidence hygiene; it is not proof of a speedup.
