# Validation records

## v0.2.0 measurement-readiness record

Date: 2026-08-20

Environment: Python 3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1, NumPy
2.5.2, Linux x86-64. The host had no NVIDIA driver or GPU.

### Claim and kill criterion

Claim: a clean installed wheel can run the dependency-free models, load its current
schemas without a source checkout, and reject a structurally or semantically
incoherent report bundle before that bundle is proposed as evidence.

Reject the claim if a wheel omits a schema; the base install imports a framework
dependency; non-finite JSON is accepted; or a suite with an incomplete schedule, path
escape, altered command, mismatched commit/seed/provider/dtype/shape, incorrect timing
aggregate, wrong sample count, or incomplete collective-rank metadata validates.

### Commands

```bash
python -m pip check
ruff check .
ruff format --check .
pytest --cov=gpu_systems_lab --cov-report=term-missing
TRITON_INTERPRET=1 pytest -m triton_interpreter -vv
python -m build
gpu-systems-lab validate-result tests/fixtures/valid-rmsnorm-v3.json
gitleaks detect --no-git --source . --redact --exit-code 1
```

Two fresh virtual environments also installed the built wheel: one with no optional
dependencies and one with the `reports` extra. The base environment ran the traffic
model and returned the documented dependency error for report validation. The reports
environment loaded the package-shipped schema and validated the fixture. A real suite
invocation with `--allow-unversioned` exercised the expected no-CUDA failure path.

### Adversarial failures found and repaired

- A schema-valid but unrelated child could initially enter a suite manifest. Recursive
  validation now checks the complete schedule, confined unique paths, canonical replay
  commands, provenance, protocol, identities, and shape coverage.
- An IEEE NaN error could initially bypass a simple greater-than tolerance check.
  Correctness now requires a finite error, writers prohibit non-standard numeric JSON,
  and readers reject non-finite tokens and in-memory values.
- The first real no-GPU orchestration attempt returned a child traceback. Benchmark
  entry points now return a concise one-line requirement error.
- The first collective v3 draft recorded only the reporting device. The accepted
  contract records device and runtime provenance for every global rank and validates
  complete rank coverage.

### Observations

- Ordinary suite: `119 passed, 2 skipped`; only the CUDA hardware case and separately
  invoked interpreter case skipped. Hardware-independent branch coverage was 94.50%.
- Separate Triton interpreter run: `1 passed, 120 deselected`.
- All three package-shipped Draft 2020-12 schemas passed meta-schema checks; producer
  builders emitted reports accepted by their current contracts.
- Tampered bundle, non-finite-number, sample-count, aggregate, and rank-coverage tests
  all failed at their registered gates.
- Source and wheel distributions built; clean-wheel base and reports-extra smokes,
  dependency consistency, lint, formatting, configuration parsing, relative links,
  external-link reachability, forbidden-name scans, and secret scanning passed.
- An independent, read-only whole-repository audit found no blocker or high-severity
  issue. Its regression-coverage and documentation-precision findings were repaired
  before this record was finalized.
- `results/published/` contains no measurement data.

### Outcome and boundary

The claim held for the registered local and clean-wheel checks. This record establishes
measurement readiness, not measurement validity on hardware. It does not establish
CUDA code generation, target-GPU numerical behavior or latency, physical memory
traffic, compiler speedup, NCCL performance, topology effects, multi-node behavior, or
production fault tolerance.

## v0.1.1 hardening record

Date: 2026-08-20

Environment: Python 3.12.3, PyTorch 2.13.0, Triton 3.7.1, NumPy 2.5.2,
Linux x86-64. The host had no NVIDIA driver or GPU.

### Audit claim and rejection conditions

Claim: the required quality dependency set executes the PyTorch reference and
autograd tests plus the Triton interpreter contract test, and a rank-local collective
mismatch is reduced to a shared decision before any rank can enter the next timed
collective.

Reject the claim if a fresh quality-equivalent environment omits the framework test
module, if the interpreter case does not execute, or if either rank proceeds past the
correctness-consensus call after one rank reports a mismatch.

### Commands

```bash
python -m pip install 'torch>=2.10,<2.14' \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,gpu]'
python -m pip check
pytest --cov=gpu_systems_lab --cov-report=term-missing
TRITON_INTERPRET=1 pytest -m triton_interpreter -vv
ruff check .
ruff format --check .
python -m build
gitleaks detect --no-git --source . --redact --exit-code 1
```

The collective failure experiment used two local Gloo ranks with a three-second
process-group timeout. One rank was assigned a failing observation while the other
was assigned a passing observation. This tests control-flow coordination, not NCCL or
GPU numerical behavior.

### Observations

- Baseline reproduction with `.[dev]`: `21 passed, 1 skipped`; the skipped module
  reported that PyTorch was absent.
- Repaired ordinary suite: `48 passed, 2 skipped`; the PyTorch reference, autograd,
  shape, parser, report, schema, and model tests executed. Only the CUDA-gated case
  and the separately invoked interpreter case skipped. Included coverage was 97.71%.
- Separate interpreter run: `1 passed, 49 deselected`.
- Baseline rank-local exit: the peer failed its next barrier after `3.08` seconds,
  matching the configured timeout.
- Repaired rank consensus: both ranks raised the shared correctness failure in at
  most `0.006` seconds; neither entered the next timed collective.
- Both strict Draft 2020-12 result schemas accepted complete report contracts and
  rejected incomplete reports.
- Lint, format, dependency-consistency, configuration parsing, package builds,
  relative-link, and secret checks passed.
- A separate read-only whole-repository audit reported no blocker or high-severity
  finding. Its stale-validation-record finding prompted this versioned record.

### Outcome and boundary

The claim held for the registered CI-equivalent and two-rank Gloo checks. This record
does not establish CUDA code generation, target-GPU correctness or latency, physical
memory traffic, NCCL behavior, topology effects, or multi-node fault tolerance.

## v0.1.0 pre-publication record

Date: 2026-08-20

Environment: Python 3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1, NumPy 2.5.2,
Linux x86-64. The host had no NVIDIA driver or GPU, so no CUDA compilation,
hardware profiling, collective run, or latency comparison was possible.

## Claim and kill criterion

Claim: the first vertical slice is executable and preserves its stated numerical
contract while its registered logical model assigns fewer tensor bytes to the fused
decomposition.

Kill criterion: retract the claim if any registered interpreter case exceeds maximum
absolute error `2e-5`, or if the same-width logical model does not assign fewer bytes
to the fused decomposition for the test shape.

## Commands

```bash
pytest --cov=gpu_systems_lab --cov-report=term-missing
TRITON_INTERPRET=1 pytest -m triton_interpreter -vv
gpu-systems-lab traffic --rows 2 --hidden 4 --element-bytes 2
ruff check .
ruff format --check .
python -m build
gitleaks detect --no-git --source . --redact --exit-code 1
```

## Observations

- CPU/framework suite: 26 passed; CUDA and interpreter tests skipped in the ordinary
  suite; hardware-independent coverage was 97%.
- Interpreter shape `(1, 7)`, FP32: maximum absolute error `1.1920929e-07`.
- Interpreter shape `(3, 17)`, FP32: maximum absolute error `0`.
- Interpreter shape `(2, 513)`, FP32: maximum absolute error `9.53674316e-07`.
- Registered tolerance for those interpreter cases: `2e-5`.
- Logical model for two rows, hidden size four, two-byte elements: fused `64` bytes;
  explicitly materialized decomposition `176` bytes.
- Lint and formatting checks passed; source and wheel distributions built; the secret
  scan found no leaks.

## Outcome and boundary

The claim held for the registered interpreter cases and logical model. This record
does not establish GPU code-generation correctness, physical memory traffic,
performance, multi-GPU behavior, or future reliability.
