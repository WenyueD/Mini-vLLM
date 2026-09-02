# AutoDL Qwen3.8 Validation

Run Mini-vLLM in its own Python 3.11 virtual environment.  Do not reuse the
Oracle environment, which may have Python 3.12 and different torch, triton,
transformers, and vLLM versions.

## Fresh Mini-vLLM Environment

```bash
cd /root/autodl-tmp/Mini-vLLM

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip uv

export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/minivllm_cache/torchinductor
export TRITON_CACHE_DIR=/root/autodl-tmp/minivllm_cache/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

uv sync
```

`HF_HOME` may point at the same Hugging Face cache used by the Oracle project to
avoid downloading Qwen weights twice.  The `.venv`, `TORCHINDUCTOR_CACHE_DIR`,
and `TRITON_CACHE_DIR` must be separate from the Oracle project.

## Environment Manifest

```bash
mkdir -p results/qwen38_target

{
  date
  hostname
  pwd
  git rev-parse HEAD
  git status --short
  python --version
  python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda', torch.version.cuda); print('device_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
  python -c "import transformers; print('transformers', transformers.__version__)"
  python -c "import triton; print('triton', triton.__version__)"
  python -c "import os; print('HF_HOME', os.environ.get('HF_HOME')); print('TORCHINDUCTOR_CACHE_DIR', os.environ.get('TORCHINDUCTOR_CACHE_DIR')); print('TRITON_CACHE_DIR', os.environ.get('TRITON_CACHE_DIR'))"
  nvidia-smi
} | tee results/qwen38_target/env_manifest.txt
```

## Config Check Without AutoConfig

Transformers 4.57.3 does not recognize `model_type=qwen3_5` through
`AutoConfig`, so use Mini-vLLM's direct `config.json` parser:

```bash
export MINIVLLM_QWEN38_PATH=/root/autodl-tmp/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

ls -lah "$MINIVLLM_QWEN38_PATH"
ls "$MINIVLLM_QWEN38_PATH" | head
python scripts/check_qwen38_config.py | tee results/qwen38_target/config_check.txt
```

## Weight Name Audit Before Loading 27B

This step reads only safetensors headers and instantiates the Mini-vLLM target on
the `meta` device.  It must pass before attempting to allocate the 27B target on
the GPU.

```bash
python scripts/audit_qwen38_weights.py | tee results/qwen38_target/weight_audit.txt
```

If this reports missing model parameters, unknown language weights, or shape
mismatches, stop and fix the mapping first.  Do not start a real 27B load while
the audit fails.

## Load-Only 27B Target Smoke

Run this only after `weight_name_audit=PASS`.  It allocates the Mini-vLLM text
target on one GPU in bf16, streams safetensors weights one file at a time, skips
vision and MTP weights, prints peak CUDA memory, and exits without running
forward or allocating KV cache.

```bash
python scripts/smoke_qwen38_load_only.py \
  --model "$MINIVLLM_QWEN38_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  | tee results/qwen38_target/load_only.txt
```

If direct CUDA staging causes a temporary OOM, retry once with CPU staging:

```bash
python scripts/smoke_qwen38_load_only.py \
  --model "$MINIVLLM_QWEN38_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cpu \
  | tee results/qwen38_target/load_only_cpu_staging.txt
```

The pass marker for this phase is `qwen38_load_only=PASS`.

## Test Commands

```bash
pytest -q
pytest -q tests/test_qwen35_config_and_model.py tests/test_hybrid_cache_transactions.py tests/test_qwen35_weight_loading.py
```

The current unit-test pass is not a real Qwen/Qwen3.8-27B target-only GPU
acceptance result.  Mark 27B load, 32-token greedy decode, official target token
alignment, and cache-vs-recompute logits as pending until a dedicated smoke run
records those artifacts under `results/qwen38_target/`.
