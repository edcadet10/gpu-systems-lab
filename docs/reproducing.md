# Reproducing checks and measurements

All commands assume a fresh clone at the commit being tested.

## Dependency-free model and report checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip check
ruff check .
ruff format --check .
pytest \
  tests/test_benchmark_support.py \
  tests/test_benchmark_suite.py \
  tests/test_cli.py \
  tests/test_result_schemas.py \
  tests/test_ring_allreduce.py \
  tests/test_roofline.py \
  tests/test_traffic.py
python -m build
```

This path exercises the analytical models, metadata parsing, schedule orchestration,
strict JSON handling, schema resources, recursive bundle validation, CLI, and package
build. It does not claim to exercise framework or kernel behavior.

## Full CPU and interpreter checks

On Linux x86-64, install a CPU framework wheel before the project extras so this path
does not depend on an NVIDIA driver:

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
```

Interpreter correctness exercises kernel operation semantics but bypasses target
compilation. It provides no GPU code-generation, latency, or profiler evidence.

## Single-GPU comparison

Install the framework build that matches the host CUDA stack, then record context:

```bash
git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=name,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.limit \
  --format=csv
nvidia-smi topo -m
python -m torch.utils.collect_env
```

Run the hardware gate and canonical suite:

```bash
python -m pip install -e '.[gpu]'
pytest -m gpu
gpu-lab-rmsnorm-suite \
  --rows 128,1024 \
  --hidden 1024,4096,8192 \
  --dtypes float16,bfloat16 \
  --providers torch_eager,triton,torch_compile \
  --process-runs 5 \
  --warmup 25 \
  --repeats 100 \
  --base-seed 17 \
  --schedule-seed 2026 \
  --output-dir results/local/rmsnorm-example
gpu-systems-lab validate-result results/local/rmsnorm-example/manifest.json
```

The directory must be new, and the repository must be clean unless the explicitly
non-publishable `--allow-unversioned` flag is supplied. Do not publish a bundle made
with that flag. Preserve every child report, including losing or noisy cases.

To inspect one provider quickly without creating a comparison suite:

```bash
gpu-lab-rmsnorm \
  --rows 128 --hidden 4096 --dtype float16 --providers triton \
  --output results/local/triton-smoke.json
gpu-systems-lab validate-result results/local/triton-smoke.json
```

## Profiling

```bash
./scripts/profile_ncu.sh
./scripts/profile_nsys.sh
```

Profiler collection perturbs execution and is not used as the latency benchmark.
Correlate profiler evidence with a separate timing suite at the same clean commit and
shape. Keep large binary reports out of Git.

## Multi-GPU check

```bash
NCCL_DEBUG=WARN torchrun --standalone --nproc-per-node=8 \
  -m gpu_systems_lab.distributed.benchmark_allreduce \
  --message-bytes 134217728 \
  --dtype float16 \
  --output results/local/allreduce.json
gpu-systems-lab validate-result results/local/allreduce.json
```

For multi-node runs also preserve launcher configuration, rank placement,
network-interface selection, NCCL environment variables, and topology output. Every
rank joins the correctness-status reduction before any rank can raise for a mismatch;
this preserves collective ordering on that failure path.

## Preparing a result contribution

Copy only a complete, recursively valid suite into a new directory below
`results/published/`. Do not rename children without regenerating the manifest because
relative paths and canonical replay commands are part of the checked contract. Follow
the [benchmark and claims policy](benchmarking.md), and include profiler evidence and
the precisely scoped claim in the pull request.
