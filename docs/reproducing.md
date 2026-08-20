# Reproducing results

## Dependency-free model checks

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest \
  tests/test_benchmark_support.py \
  tests/test_cli.py \
  tests/test_result_schemas.py \
  tests/test_ring_allreduce.py \
  tests/test_roofline.py \
  tests/test_traffic.py
python -m build
```

This path deliberately does not claim to exercise framework or kernel behavior.

## Full CPU and interpreter checks

On Linux x86-64, install a CPU framework wheel before the project extras so the
interpreter path does not pull a CUDA-enabled framework wheel:

```bash
python -m pip install 'torch>=2.10,<2.14' \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,gpu]'
python -m pip check
pytest --cov=gpu_systems_lab --cov-report=term-missing
TRITON_INTERPRET=1 pytest -m triton_interpreter -vv
```

This is the same dependency scope used by the required quality check. Interpreter
correctness exercises kernel semantics, but bypasses target compilation and provides
no GPU code-generation or performance evidence.

## GPU checks

Record this context before benchmarking:

```bash
git rev-parse HEAD
nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit \
  --format=csv
nvidia-smi topo -m
python -m torch.utils.collect_env
```

Install and run:

```bash
python -m pip install -e '.[dev,gpu]'
pytest -m gpu
gpu-lab-rmsnorm --output results/local-rmsnorm.json
```

Run each candidate/baseline comparison from five fresh Python processes. Preserve all
JSON files and the exact command. If GPU clocks or power limits are locked, report the
commands and restore the machine's prior settings after the experiment.

## Profiling

```bash
./scripts/profile_ncu.sh
./scripts/profile_nsys.sh
```

Profiler collection perturbs execution and is not used as the latency benchmark.
Correlate profiler evidence with a separate timing run at the same commit and shape.

## Multi-GPU checks

```bash
NCCL_DEBUG=WARN torchrun --standalone --nproc-per-node=8 \
  -m gpu_systems_lab.distributed.benchmark_allreduce \
  --message-bytes 134217728 \
  --output results/local-allreduce.json
```

For multi-node runs also preserve launcher configuration, hostname-to-rank mapping,
network-interface selection, NCCL environment variables, and topology output.
The benchmark makes every rank join a correctness-status reduction before any rank
can raise for a numerical mismatch; this preserves collective ordering on that path.
