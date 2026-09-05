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

## Forward, Decode, And Engine Smokes

Run these after load-only passes.  The cache-vs-recompute check compares the
B=1 cached target path with a full recompute path.  bf16 final-vocab logits may
show small numeric drift, so the 27B smoke uses a looser all-vocab tolerance than
the unit tests while still requiring the target token decision to match.

```bash
python scripts/smoke_qwen38_single_token.py \
  --model "$MINIVLLM_QWEN38_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --prompt "Hello" \
  | tee results/qwen38_target/single_token_forward.txt

python scripts/smoke_qwen38_generate_no_cache.py \
  --model "$MINIVLLM_QWEN38_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --prompt "Hello" \
  --max-new-tokens 32 \
  | tee results/qwen38_target/no_cache_greedy_32.txt

python scripts/smoke_qwen38_cache_vs_recompute.py \
  --model "$MINIVLLM_QWEN38_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --prompt "Hello" \
  --max-new-tokens 32 \
  --atol 0.3 \
  --rtol 0.05 \
  | tee results/qwen38_target/cache_vs_recompute_32_bf16_tol03.txt

python scripts/smoke_qwen38_engine_target_only.py \
  --model "$MINIVLLM_QWEN38_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --prompt "Hello" \
  --max-new-tokens 32 \
  | tee results/qwen38_target/engine_target_only_32.txt
```

If direct CUDA staging is tight in the engine smoke, retry only that step with
CPU staging:

```bash
python scripts/smoke_qwen38_engine_target_only.py \
  --model "$MINIVLLM_QWEN38_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cpu \
  --prompt "Hello" \
  --max-new-tokens 32 \
  | tee results/qwen38_target/engine_target_only_32_cpu_staging.txt
```

The pass markers are `qwen38_single_token_forward=PASS`,
`qwen38_no_cache_greedy_32=PASS`, `qwen38_cache_vs_recompute=PASS`, and
`qwen38_engine_target_only=PASS`.

If cache-vs-recompute fails only because the all-vocab bf16 logits exceed
tolerance while `cached_token` and `recompute_token` still match, rerun it as a
token-decision acceptance check:

```bash
python scripts/smoke_qwen38_cache_vs_recompute.py \
  --model "$MINIVLLM_QWEN38_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --prompt "Hello" \
  --max-new-tokens 32 \
  --atol 0.3 \
  --rtol 0.05 \
  --allow-logit-drift \
  | tee results/qwen38_target/cache_vs_recompute_32_token_decision.txt
```

Do not use `--allow-logit-drift` if `cached_token` and `recompute_token` differ.

Before DFlash2 Oracle proposal alignment, verify that Mini-vLLM and Oracle
target-only agree on the first greedy token for the same raw prompt.  For the
current Oracle target-only run, `"Hello"` produced first completion token `11`:

```bash
python scripts/smoke_qwen38_single_token.py \
  --model "$MINIVLLM_QWEN38_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --prompt "Hello" \
  --expected-next-token-id 11 \
  --top-k 50 \
  --json-output results/qwen38_target/target_oracle_anchor_hello.json \
  | tee results/qwen38_target/target_oracle_anchor_hello.txt
```

The pass marker is `qwen38_target_oracle_anchor=PASS`.  Do not run DFlash2
proposal-ID comparison while this target-anchor check fails.  If it fails, keep
the JSON artifact and inspect `expected_next_token_rank`,
`expected_next_token_logit`, and `top_token_ids` first; the DFlash2 comparison is
not meaningful until the Mini-vLLM target's first greedy token matches Oracle
target-only.

## DFlash2 B=1 Online Smoke

Run this after `qwen38_dflash2_engine_b1=PASS`.  It compares target-only online
streaming with DFlash2 online streaming for one request, requires exact token ID
equality, and verifies that the DFlash2 path produced speculative proposals.

```bash
export MINIVLLM_DFLASH2_PATH=/root/autodl-tmp/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/<snapshot>
mkdir -p results/qwen38_target

python scripts/smoke_qwen38_dflash2_online_b1.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --dflash2-safe-open-device cuda \
  --prompt "Hello" \
  --max-new-tokens 32 \
  | tee results/qwen38_target/dflash2_online_b1_hello_32.txt
```

If direct CUDA staging is tight, retry the online smoke with CPU staging:

```bash
python scripts/smoke_qwen38_dflash2_online_b1.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cpu \
  --dflash2-safe-open-device cpu \
  --prompt "Hello" \
  --max-new-tokens 32 \
  | tee results/qwen38_target/dflash2_online_b1_hello_32_cpu_staging.txt
```

The pass marker is `dflash2_online_b1=PASS`.  Keep
`dflash2_oracle_alignment=PENDING` until proposal IDs are compared against the
Oracle path.

## DFlash2 B=2 Online Smoke

After both B=1 smokes pass, run two requests concurrently and require evidence
that they share a real target and speculative decode batch:

```bash
python scripts/smoke_qwen38_dflash2_online_b2.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --dflash2-safe-open-device cuda \
  --max-new-tokens 32 \
  --debug-metadata \
  | tee logs/dflash2_online_b2_same_prompt_20260905.log
```

The default submits two identical `Hello` prompts. The pass marker is
`dflash2_online_b2=PASS`; the script also requires exact per-request target-only
equality, `active_batch_size >= 2`, `speculative_batch_size >= 2`, proposals for
both requests in a shared step, and at least one accepted draft token per request.
After the tensor-batched verifier change, the script additionally requires
`qwen35_verification.max_batch_size >= 2` and at least one
`tensor_batched_token_step`.

AutoDL A800 validation on 2026-09-05 passed both same-prompt and heterogeneous
B=2 runs. The same-prompt run recorded `286 proposed / 16 accepted`; the
heterogeneous `Hello` plus `The capital of France is` run recorded
`236 proposed / 24 accepted`, with per-request accepted counts of 8 and 16.
Both runs matched target-only token IDs exactly and reached shared target and
speculative batch size 2. The heterogeneous log is
`logs/dflash2_online_b2_heterogeneous_20260905.log`.

The tensor-batched verifier rerun is recorded in
`logs/dflash2_online_b2_tensor_batched_clean_20260905.log`. It passed exact
target-only equality again and reported `max_batch_size=2`, 108
`tensor_batched_token_steps`, `236 proposed / 24 accepted`, and a peak allocation
of 71.99 GiB. Cached verification improved from 19.43s to 12.95s (33.3%), while
total DFlash2 generation improved from 20.89s to 14.43s (30.9%). The latest
target-only generation took 7.19s. No exception or residual GPU process was
observed after shutdown.

### Speculative-length sweep

The same heterogeneous B=2 workload was rerun with configurable proposal lengths
on 2026-09-05. All four runs passed exact target-only equality and tensor-batch
assertions:

| K | Target-only | DFlash2 | Verify | Proposed / accepted | Result |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.06s | 5.31s | 3.79s | 43 / 17 | 1.33x faster |
| 2 | 7.13s | 6.52s | 5.09s | 71 / 24 | 1.09x faster |
| 3 | 7.20s | 8.22s | 6.77s | 106 / 24 | 0.88x |
| 7 | 6.90s | 14.29s | 12.82s | 236 / 24 | 0.48x |

Logs are `logs/dflash2_online_b2_k{1,2,3,7}_20260905.log`. For this
workload, K=1 is the measured optimum. K=3 adds only one accepted token beyond
K=2, while K=4 through K=7 add none, so their verification cost cannot be
recovered. Use `--num-speculative-tokens 1` for the current performance smoke;
keep K configurable until a broader prompt corpus supports an adaptive policy.

## DFlash2 Oracle Proposal Alignment

Run this after `dflash2_online_b1=PASS`.  DFlash uses a `1 + N` draft query
layout: the anchor is the target-generated bonus token, and only the following
mask positions produce proposal tokens.  Regenerate both fixtures whenever this
section changes; do not compare against an older prompt-final-anchor fixture.

First export the Mini-vLLM fixture from the exact same prompt and checkpoints:
The JSON should contain `fixture_schema_version=2` and
`anchor_semantics=target_bonus_token`.

```bash
mkdir -p results/qwen38_target/oracle_alignment
rm -f \
  results/qwen38_target/oracle_alignment/minivllm_dflash2_hello.json \
  results/qwen38_target/oracle_alignment/oracle_dflash2_hello.json

python scripts/smoke_dflash2_standalone_proposal.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --device cuda:0 \
  --dtype bfloat16 \
  --safe-open-device cuda \
  --prompt "Hello" \
  --json-output results/qwen38_target/oracle_alignment/minivllm_dflash2_hello.json \
  | tee results/qwen38_target/oracle_alignment/minivllm_dflash2_hello.txt
```

Then run the Oracle/vLLM exporter from the Oracle environment.  This script
monkey-patches vLLM's DFlash speculator and captures the first non-dummy,
post-target-sample draft proposal block returned by Oracle.  Prefill captures
with `num_sampled=0` are skipped by default because their anchor is still a
prompt token, not the target-generated bonus token.  The exporter also runs an
Oracle target-only preflight and stops if Oracle DFlash is not target-lossless
for the selected prompt.

```bash
cd /root/autodl-tmp/Mini-vLLM
source /path/to/oracle/.venv/bin/activate
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=fork

grep -n "fixture_schema_version" scripts/export_oracle_dflash2_fixture.py

python scripts/export_oracle_dflash2_fixture.py \
  --target-model "$MINIVLLM_QWEN38_PATH" \
  --draft-model "$MINIVLLM_DFLASH2_PATH" \
  --output results/qwen38_target/oracle_alignment/oracle_dflash2_hello.json \
  --prompt "Hello" \
  --max-tokens 16 \
  --num-speculative-tokens 7 \
  --method dflash \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 256 \
  --dtype bfloat16 \
  --skip-target-preflight \
  --expected-target-first-token-id 11 \
  --anchor-token-mode minivllm \
  --anchor-position-mode minivllm \
  --trust-remote-code \
  --enforce-eager \
  --extra-llm-kwargs '{"cpu_offload_gb": 8, "max_num_seqs": 1, "max_num_batched_tokens": 256}' \
  | tee results/qwen38_target/oracle_alignment/oracle_dflash2_hello.txt
```

If this still OOMs, try `cpu_offload_gb` values of 12 or 16 before trying
quantization.  If the exporter reports that Oracle DFlash is not target-lossless
for first token `11`, do not compare fixtures yet; first find a prompt/config
where Oracle target-only and Oracle DFlash start with the same completion token.

Then compare:

```bash
source .venv/bin/activate

python scripts/compare_dflash2_oracle_fixture.py \
  --minivllm results/qwen38_target/oracle_alignment/minivllm_dflash2_hello.json \
  --oracle results/qwen38_target/oracle_alignment/oracle_dflash2_hello.json \
  --checksum-atol 0.5 \
  | tee results/qwen38_target/oracle_alignment/dflash2_oracle_alignment_hello.txt
```

The pass marker is `dflash2_oracle_alignment=PASS`.

## Test Commands

```bash
pytest -q --basetemp ./.pytest-tmp
pytest -q --basetemp ./.pytest-tmp \
  tests/test_qwen35_config_and_model.py \
  tests/test_hybrid_cache_transactions.py \
  tests/test_qwen35_weight_loading.py \
  tests/test_spec_coordinator.py \
  tests/test_dflash2_contract.py
```

The heterogeneous B=2 GPU run above is the acceptance result for tensor-batched
cached verification. It does not resolve Oracle proposal alignment, which stays
pending until the Oracle target-only and DFlash outputs are target-lossless for
the same prompt and configuration.

The local Windows audit on 2026-09-02 used the repository `.venv` and
`--basetemp ./.pytest-tmp` because the default user temp directory denied pytest
access. The latest local regression is `111 passed, 2 skipped`; focused AutoDL
tests report `20 passed`.
