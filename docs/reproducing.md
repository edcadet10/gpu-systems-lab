# Reproducing results

## CPU-only checks

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest --cov=gpu_systems_lab --cov-report=term-missing
python -m build
```

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

Without a CUDA device, a small functional smoke test can use Triton's interpreter:

```bash
TRITON_INTERPRET=1 pytest -m triton_interpreter
```

Interpreter correctness exercises kernel semantics but does not compile target GPU
code and provides no performance evidence.

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
