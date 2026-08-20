# Result registry

This directory separates report contracts from measured evidence.

- `rmsnorm.schema.json` and `allreduce.schema.json` are the historical v2 contracts
  shipped in v0.1.1. Their URLs remain stable for old reports.
- The current, package-shipped contracts are
  [`rmsnorm-v3.schema.json`](../src/gpu_systems_lab/schemas/rmsnorm-v3.schema.json),
  [`allreduce-v3.schema.json`](../src/gpu_systems_lab/schemas/allreduce-v3.schema.json),
  and
  [`rmsnorm-suite-v1.schema.json`](../src/gpu_systems_lab/schemas/rmsnorm-suite-v1.schema.json).
- `local/` is ignored scratch space for new runs.
- `published/` is the reviewed evidence registry. It intentionally contains no GPU
  measurement yet.

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
