# Contributing

Thanks for helping make GPU performance work easier to inspect and reproduce. Code,
benchmark data, issue reports, negative results, documentation, and design critiques
are all useful contributions.

## Before opening a pull request

1. Search existing issues and discussions. Open a proposal first for a new kernel,
   dependency, benchmark protocol, or public API.
2. Keep the change focused. A kernel contribution should include its reference,
   contract, tests, benchmark target, and documented support boundary together.
3. Do not include proprietary code, confidential data, model weights, secrets, or
   benchmark results you are not authorized to publish.

## Development setup

```bash
git clone https://github.com/edcadet10/gpu-systems-lab.git
cd gpu-systems-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.10,<2.14' \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,gpu]'
ruff check .
ruff format --check .
pytest --cov=gpu_systems_lab --cov-report=term-missing
TRITON_INTERPRET=1 pytest -m triton_interpreter -vv
python -m build
```

The explicit CPU wheel keeps this setup usable without an NVIDIA driver while still
exercising the PyTorch reference, autograd behavior, and Triton interpreter on Linux
x86-64. GPU contributors can install the framework wheel appropriate for their CUDA
stack, then install `.[dev,gpu]` and run `pytest -m gpu`. Exercising the explicit CUDA
provider also requires a local CUDA toolkit with `nvcc`; its first use builds through
Ninja in the framework extension cache.

The required quality check runs the full sequence above. The Python-version matrix
runs the dependency-free models, parsers, metadata helpers, suite orchestration, and
report validation. Neither job presents interpreter execution as target-GPU code
generation or timing.

## Kernel checklist

- Write the mathematical and dtype contract before the accelerated implementation.
- Include awkward shapes, masked tails, zeros, large finite values, and invalid input
  tests where relevant.
- Use a dtype-appropriate, scale-aware correctness rule and test both large-magnitude
  and near-zero failures; do not loosen a registered gate after observing results.
- State layout, alignment, dtype, compute-capability, and size restrictions explicitly.
- Reject unsupported autograd behavior; do not silently detach.
- Compare with a strong baseline and keep losing shapes in the result set.
- Add an NVTX range or another stable profiler target.
- Document what result would falsify the proposed optimization.
- Run the Triton interpreter test and, when hardware is available, the GPU marker.

## Benchmark submissions

Use `gpu-lab-rmsnorm-suite` with five process-level runs and validate its manifest with
`gpu-systems-lab validate-result`. Use the benchmark-result issue form. Submit the
complete manifest and every referenced raw report for candidate and baseline, plus a
profiler summary. Do not commit large binary profiler reports; attach them to the issue
or an archival release. Follow [docs/benchmarking.md](docs/benchmarking.md).

Local data belongs under ignored `results/local/`. Only reviewed, recursively valid
evidence belongs below `results/published/`; never publish a suite generated with
`--allow-unversioned`.

## Pull requests

- Explain the contract and why the change belongs in this repository.
- Link the issue or discussion.
- Include tests and update relevant documentation.
- Complete the claim and falsification fields in the pull request template.
- Expect review questions about negative cases, numerical behavior, measurement
  controls, and hardware generalization.

Maintainers may ask to split unrelated changes. A passing test suite is necessary but
does not itself establish a performance claim.

## Community conduct and licensing

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). By submitting a
contribution, you agree that it is licensed under the repository's Apache-2.0 license
and that you have the right to submit it.
