# Pre-publication validation record

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
