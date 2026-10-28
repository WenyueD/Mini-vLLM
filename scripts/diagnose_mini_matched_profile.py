"""Run a warmed, exact-input first-verification diagnostic, not a throughput benchmark."""
import sys, json, time, functools, os
from pathlib import Path
from dataclasses import replace
import torch

base = Path(__file__).resolve().parent
dest = base / 'merged' if os.environ.get('DIAG_MERGED_HEAD') == '1' else base
dest.mkdir(exist_ok=True)
root = base / 'candidate'
sys.path.insert(0, str(root / 'src'))
import myvllm
myvllm.__path__ = [str(root / 'src/myvllm')]
sys.path.insert(0, str(root / 'scripts'))
import benchmark_qwen38_dflash2_concurrency as bench
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.hybrid_cache import HybridCacheManager, FullAttentionLayerState
from myvllm.models import qwen3_8 as qwen
from myvllm.engine.llm_engine import RunnerDFlash2Proposer

cfg = json.loads((base/'inputs.json').read_text())
official = json.loads((base/'official_evidence.json').read_text())
v = official['verify']
queries = {v['positions'][s]: v['input_ids'][s:e] for s,e in zip(v['query_start_loc'],v['query_start_loc'][1:])}
state = dict(enabled=False, active=False, forced=False, verify=False, commit=False)
evidence = {'scope':'B4 K7 exact first verification, warmed, profiler overhead included'}
def save():
    (dest/'mini_evidence.json').write_text(json.dumps(evidence, indent=2))

def workload(tokenizer, workload, total_requests, seed, **kw):
    result = []
    for i, ids in enumerate(cfg['prompts']):
        prompt = tokenizer.decode(ids)
        assert tokenizer.encode(prompt) == ids
        result.append(bench.WorkloadRequest(f'tasks4096-{i:06d}',prompt,tuple(ids),16,'tasks4096'))
    return result
bench.build_workload = workload
native_config = bench.build_config
def config(*a, **kw):
    c = native_config(*a, **kw)
    c.update(max_model_length=3203,qwen35_cached_attention_pad_to_length=3203,
             qwen35_merged_lm_head=os.environ.get('DIAG_MERGED_HEAD') == '1')
    evidence['config'] = c
    save()
    return c
bench.build_config = config
native_reset = bench.reset_runtime_stats
def reset(*a, **kw):
    out = native_reset(*a, **kw)
    state['enabled'] = True
    return out
bench.reset_runtime_stats = reset
ModelRunner.warmup_model = lambda self: None
native_propose = RunnerDFlash2Proposer.propose_batch
def propose(self, sequences, max_tokens):
    out = native_propose(self,sequences,max_tokens)
    if state['enabled'] and not state['forced']:
        evidence['inputs'] = []
        patched = []
        assert len(sequences) == cfg['batch']
        for seq,p in zip(sequences,out):
            prefix = list(seq.token_ids[:-1])
            assert prefix in cfg['prompts']
            query = queries[len(prefix)]
            assert seq.last_token == query[0]
            evidence['inputs'].append(dict(sequence_id=seq.seq_id,prefix=prefix,query=query,natural_proposal=p.token_ids))
            patched.append(replace(p, token_ids=tuple(query[1:])))
        state['forced'] = True
        save()
        return patched
    return out
RunnerDFlash2Proposer.propose_batch = propose

def annotate(owner, method, label):
    original = getattr(owner,method)
    @functools.wraps(original)
    def call(*a, **kw):
        if state['active']:
            with torch.profiler.record_function('diag/'+label):
                return original(*a, **kw)
        return original(*a, **kw)
    setattr(owner,method,call)

for owner,method,label in [
    (qwen.Qwen35TokenwiseLinear,'forward','projection'),
    (qwen.Qwen35RMSNorm,'forward','rmsnorm'),
    (qwen.Qwen35RMSNormGated,'forward','gdn_gated_norm'),
    (qwen.Qwen35GatedDeltaNet,'_causal_conv','gdn_conv'),
    (qwen.Qwen35GatedDeltaNet,'forward','gdn_recurrence_and_elementwise'),
    (qwen,'fused_cached_gdn','gdn_fused_recurrence'),
    (qwen,'l2norm','gdn_l2norm'),
    (qwen.Qwen35Attention,'forward_with_cache_batch','attention_core_and_cache'),
    (qwen,'_qwen35_tokenwise_matmul','attention_matmul'),
    (qwen.Qwen35ForCausalLM,'compute_logits','lm_head'),
    (qwen.Qwen35MLP,'forward','mlp_activation'),
    (HybridCacheManager,'begin_transaction','cache_transaction'),
    (FullAttentionLayerState,'append','cache_append'),
    (ModelRunner,'_sample_qwen35_logits','sampling'),
    (HybridCacheManager,'restore_verified_prefix','commit_restore_prefix'),
    (HybridCacheManager,'finalize_replayed_commit','commit_finalize'),
    (ModelRunner,'_append_dflash2_hidden_history','commit_hidden_history'),
    (ModelRunner,'_install_dflash2_projected_context','commit_context_projection'),
]:
    annotate(owner,method,label)

def profile(stage, method):
    original = getattr(ModelRunner,method)
    def call(self,payload):
        if not state['enabled'] or state[stage]:
            return original(self,payload)
        assert state['forced'], 'Exact-input proposal hook must run before profiling'
        if stage == 'verify':
            assert len(payload.sequences) == cfg['batch']
            for seq, proposal in zip(payload.sequences,payload.proposals):
                assert [seq.last_token, *proposal.token_ids] == queries[len(seq.token_ids)-1]
        state[stage] = True
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as p:
            state['active'] = True
            start = time.perf_counter()
            with torch.profiler.record_function('diag/'+stage):
                out = original(self,payload)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            state['active'] = False
        p.export_chrome_trace(str(dest/f'mini_{stage}.trace.json'))
        (dest/f'mini_{stage}.operators.txt').write_text(p.key_averages().table(sort_by='self_cuda_time_total',row_limit=30))
        evidence[stage] = dict(wall_s_instrumented=elapsed)
        if stage == 'verify':
            evidence[stage]['target_tokens'] = [dict(sequence_id=x.sequence_id,tokens=x.target_token_ids) for x in out]
        else:
            evidence[stage]['accepted_lengths'] = payload
        save()
        return out
    setattr(ModelRunner,method,call)
profile('verify','_verify_qwen35_cached')
profile('commit','commit_verified_cache')
sys.argv = [__file__,'--target-model',cfg['target'],'--draft-model',cfg['draft'],'--workload','tasks4096','--concurrency','4','--requests-per-run','4','--warmup-requests','4','--warmup-output-tokens','16','--repeats','1','--timeout','14400','--num-speculative-tokens','7','--parallel-target-verification','--parallel-verification-chunk-size','8','--vectorized-target-numerics','--fused-gdn-target','--cached-target-micro-batch-size','4','--lightweight-live-stats','--variants','dflash2','--run-id','matched_profile_20261028','--fused-prefix-commit','--reuse-verification-samples','--compact-verification-copies','--short-prefill-attention']
sys.argv.extend(['--results-dir', str(dest/'benchmark_results'),
                 '--logs-dir', str(dest/'benchmark_logs')])
bench.main()
assert state['forced'] and 'inputs' in evidence and 'commit' in evidence
evidence['complete'] = True
save()
