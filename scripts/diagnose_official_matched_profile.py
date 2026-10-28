"""Offline V2 diagnostic: profile first verification, record its exact inputs."""
import os
os.environ.update(VLLM_USE_V2_MODEL_RUNNER='1', VLLM_USE_FLASHINFER_SAMPLER='0',
                  VLLM_ENABLE_V1_MULTIPROCESSING='0', HF_HUB_OFFLINE='1')
import json
import functools
import time
from pathlib import Path
import torch
from vllm import LLM, SamplingParams
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

dest = Path(__file__).parent
cfg = json.loads((dest / 'inputs.json').read_text())
state = {'enabled': False, 'step': 0, 'active': False}
evidence = {'scope': f"First B{cfg['batch']} verification and subsequent state update; instrumented eager diagnostic", 'events': []}
prof = None

def save():
    (dest / 'official_evidence.json').write_text(json.dumps(evidence, indent=2))

def annotate(owner, name, label):
    native = getattr(owner, name)
    @functools.wraps(native)
    def call(*a, **kw):
        if state['active']:
            with torch.profiler.record_function('diag/' + label):
                return native(*a, **kw)
        return native(*a, **kw)
    setattr(owner, name, call)

for name in ['update_requests', 'free_states', 'prepare_inputs', 'prepare_attn',
             'sample', 'postprocess_sampled', 'postprocess_num_computed_tokens']:
    annotate(GPUModelRunner, name, name)

native_propose = DFlashSpeculator.propose
def propose(self, *a, **kw):
    if state['active']:
        with torch.profiler.record_function('diag/draft_propose'):
            out = native_propose(self, *a, **kw)
    else:
        out = native_propose(self, *a, **kw)
    if state['enabled'] and state['step'] <= 2:
        batch = a[0]
        idx = batch.idx_mapping
        evidence['events'].append({'step': state['step'], 'kind': 'proposal',
            'proposal': out.detach().cpu().tolist(),
            'anchors': a[7][idx].detach().cpu().tolist(),
            'num_sampled': a[5].detach().cpu().tolist(),
            'num_rejected': a[6].detach().cpu().tolist()})
        save()
    return out
DFlashSpeculator.propose = propose

native_exec = GPUModelRunner.execute_model
def execute(self, scheduler_output, *a, **kw):
    global prof
    real = state['enabled'] and not kw.get('dummy_run', False)
    if real:
        state['step'] += 1
    if real and state['step'] == 2:
        torch.cuda.synchronize()
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
        prof.__enter__(); state['active'] = True
    t = time.perf_counter()
    if state['active']:
        with torch.profiler.record_function('diag/execute_model'):
            out = native_exec(self, scheduler_output, *a, **kw)
    else:
        out = native_exec(self, scheduler_output, *a, **kw)
    if real and state['step'] == 2:
        torch.cuda.synchronize()
        batch = self.execute_model_state.input_batch
        evidence['verify'] = {'scheduled': scheduler_output.num_scheduled_tokens,
            'total_tokens': scheduler_output.total_num_scheduled_tokens,
            'wall_s_instrumented': time.perf_counter()-t,
            'batch_attrs': list(vars(batch))}
        for name in ['input_ids', 'positions', 'query_start_loc', 'idx_mapping']:
            value = getattr(batch, name, None)
            if isinstance(value, torch.Tensor):
                evidence['verify'][name] = value.detach().cpu().tolist()
        save()
    if real and state['step'] == 3 and prof is not None:
        torch.cuda.synchronize(); state['active'] = False
        prof.__exit__(None, None, None)
        prof.export_chrome_trace(str(dest/'official.trace.json'))
        (dest/'official.operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35))
        prof = None
    return out
GPUModelRunner.execute_model = execute
annotate(GPUModelRunner, 'sample_tokens', 'sample_and_next_draft')

native_call = torch.nn.Module._call_impl
def module_call(self, *a, **kw):
    name = type(self).__name__
    if state['active'] and any(x in name.lower() for x in ['linear', 'attention', 'gateddelta', 'rmsnorm', 'logits']):
        with torch.profiler.record_function('diag/module/' + name):
            return native_call(self, *a, **kw)
    return native_call(self, *a, **kw)
torch.nn.Module._call_impl = module_call

llm = LLM(model=cfg['target'], tensor_parallel_size=1, gpu_memory_utilization=.90,
    max_model_len=3203, dtype='bfloat16', enforce_eager=True, language_model_only=True,
    max_num_seqs=cfg['batch'], max_num_batched_tokens=8192, enable_prefix_caching=False,
    async_scheduling=False, seed=cfg['seed'],
    speculative_config={'method':'dflash','model':cfg['draft'],'num_speculative_tokens':7})
prompts = [{'prompt_token_ids': x} for x in cfg['prompts']]
params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, seed=cfg['seed'])
llm.generate(prompts, params, use_tqdm=False)
state['enabled'] = True
outs = llm.generate(prompts, params, use_tqdm=False)
evidence['outputs'] = [list(x.outputs[0].token_ids) for x in outs]
evidence['complete'] = True
save()
