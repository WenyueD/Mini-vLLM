# Qwen3.8 + DFlash2 Progress - 2026-09-02

## Summary

Local work completed in `D:\LLM项目\MinivLLM`:

- Audited the clean baseline branch and existing Qwen3.8/hybrid-cache tests.
- Re-ran the full local regression suite with the repository Python 3.11 venv.
- Connected Qwen3.5/Qwen3.8 cached verification to speculative cache commit and
  rollback hooks.
- Added a guarded DFlash2 contract module for config validation, ProposalBatch,
  selector top-k tensors, draft-cache accounting, and block-parallel proposer
  invocation.
- Corrected the Qwen/Qwen3.8-27B fixture facts to match the public target
  config: `vocab_size=248320`, `intermediate_size=17408`,
  `num_attention_heads=24`, `num_key_value_heads=4`, `head_dim=256`,
  `linear_num_value_heads=48`.

The DFlash2 neural model, checkpoint weight loader, real block-parallel proposal
math, and end-to-end A800 validation remain pending.

## Commands Run

```powershell
git status --short --branch
git diff --stat
rg --files
rg -n "DFlash|dflash|HybridCache|transaction|hidden_taps|tap|Qwen3_8|qwen3_8|spec" src tests scripts docs
python -m pytest -q
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\.pytest-tmp
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\.pytest-tmp tests\test_spec_coordinator.py tests\test_qwen35_config_and_model.py tests\test_engine_spec_integration.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\.pytest-tmp tests\test_dflash2_contract.py tests\test_spec_coordinator.py tests\test_qwen35_config_and_model.py tests\test_engine_spec_integration.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp .\.pytest-tmp
.\.venv\Scripts\python.exe -m compileall -q src tests
ssh -i C:\Users\dwy30\.ssh\codex_temp -p 15723 -o BatchMode=yes -o ConnectTimeout=10 root@connect.bjb1.seetacloud.com "hostname; date; nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv; ps -eo pid,comm,args --sort=-%mem | head -20; df -h /root/autodl-tmp; ls -lah /root/autodl-tmp"
```

## Local Results

- Global Python 3.13.7 pytest failed at collection because it has no `torch` and
  cannot import `myvllm`; this interpreter is not the project environment.
- Project `.venv` Python 3.11.16 full regression passed with
  `81 passed, 2 skipped`.
- `compileall -q src tests` passed.
- `black --check` was not used as a gate because the existing repository has many
  pre-existing non-black-formatted files outside this change. A touched-file
  check still reported formatting churn, so no broad formatting rewrite was made.

## Remote Status

AutoDL status check could not complete:

- sandboxed SSH failed with key/network permission denial;
- approved SSH retry failed with `banner exchange: Connection to UNKNOWN port -1:
  Connection refused`.

No remote files were written, no remote processes were stopped, and no GPU test
was started.

## P0-P7 Status

| Priority | Status | Evidence |
| --- | --- | --- |
| P0 regression | PASS local | `.venv` pytest: `81 passed, 2 skipped`; compileall passed. |
| P1 target real baseline | PENDING | AutoDL SSH refused; no real 27B A800 run. |
| P2 cache transaction | PASS local | Hybrid manager tests plus coordinator hook tests cover 0/partial/all accepted and rollback. |
| P3 hidden taps | PASS local | Zero-based `[5,19,33,47,61]` fixture and logits-invariance tests; no cross-Oracle numeric checksum yet. |
| P4 DFlash2 standalone | PENDING | Contract/proposer shell and selector tests added; real checkpoint/model math not implemented. |
| P5 B=1 end-to-end | PENDING | Existing mock/ngram framework passes; real DFlash2 B=1 offline/online not run. |
| P6 batch | PENDING | Generic framework has heterogeneous accepted-length tests; Qwen3.8/DFlash2 batch not implemented. |
| P7 performance | PENDING | Correctness and real GPU validation not complete, so no speed benchmark run. |

## References

- Target config checked against
  `https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json`.
- DFlash2 contract fields prepared for
  `https://huggingface.co/incoai/Qwen3.8-27B-DFlash2`.
