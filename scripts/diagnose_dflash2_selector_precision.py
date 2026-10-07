"""Shadow official selector scoring on unchanged live Mini-vLLM proposals.

This is a diagnostic run, not a throughput benchmark. It returns the original
proposal, compares alternatives against an already audited target token stream,
and checks cached versus rebuilt draft context at selected cycles.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

import torch

from myvllm.engine.model_runner import ModelRunner
from myvllm.spec_decode.dflash2 import DFlash2CandidateSelector


def walk(candidates, scores):
    rows = torch.arange(candidates.shape[0], device=candidates.device)
    previous = torch.zeros_like(rows)
    result = []
    for step in range(candidates.shape[1]):
        previous = scores[rows, step, previous].argmax(-1)
        result.append(candidates[rows, step, previous])
    return torch.stack(result, dim=1)


def accepted_prefix(proposal, expected):
    for index, (left, right) in enumerate(zip(proposal, expected)):
        if left != right:
            return index
    return min(len(proposal), len(expected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--official-model-source', type=Path, required=True)
    parser.add_argument('--capture-replay', action='store_true')
    parser.add_argument('--skip-startup-warmup', action='store_true')
    args, remaining = parser.parse_known_args()
    dest = args.evidence_dir
    dest.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(next(args.reference_run.glob('target_only_*_summary.json')).read_text())
    truth = {row['request_id']: row['token_ids'] for row in (
        json.loads(line) for line in Path(baseline['artifacts']['request_jsonl']).read_text().splitlines()
    )}
    source = args.official_model_source.read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_score_edges')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.official_model_source), 'exec'), namespace)
    official_score = namespace['_score_edges']
    (dest/'official_score_edges.py').write_text(ast.get_source_segment(source, function))
    (dest/'provenance.json').write_text(json.dumps({
        'reference_run': str(args.reference_run), 'official_source': str(args.official_model_source),
        'official_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'diagnostic_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'returns_original_proposals': True,
        'skip_startup_warmup': args.skip_startup_warmup,
        'capture_replay': args.capture_replay,
        'scope': 'Official scoring function with identical Mini candidates, hidden projection, and weights; not full Oracle runtime equivalence',
    }, indent=2))
    state = {'enabled': False, 'rows': None, 'cycle': 0, 'records': [], 'cache_checks': []}
    native_select = DFlash2CandidateSelector.select_path
    native_propose = ModelRunner.propose_dflash2
    if args.skip_startup_warmup:
        # Diagnostic only: retain all numerical/padding settings and check every
        # emitted prefix against the previously audited fully warmed run.
        ModelRunner.warmup_model = lambda self: None
    if '--warmup-requests' in remaining and remaining[remaining.index('--warmup-requests')+1] == '0':
        state['enabled'] = True

    def selector(self, logits, hidden, anchors):
        native = native_select(self, logits, hidden, anchors)
        if not state['enabled'] or state['rows'] is None:
            return native
        ids = logits.topk(self.top_k, dim=-1).indices
        unary = logits.gather(-1, ids)
        edges = self.score_candidate_edges(ids, unary, hidden, anchors)
        official = official_score(self.predecessor_codebook, self.successor_codebook,
                                  ids, unary.float(), self.hidden_projection(hidden), anchors, self.top_k)
        shadow = walk(ids, official)
        assert torch.equal(walk(ids, edges), native.token_ids)
        mini_fp32 = self.score_candidate_edges(ids, unary.float(), hidden, anchors)
        assert torch.equal(official, mini_fp32), 'additional selector math divergence'
        native_ids, shadow_ids = native.token_ids.tolist(), shadow.tolist()
        assert len(native_ids) == len(state['rows'])
        for row, ((seq, budget), old, new) in enumerate(zip(state['rows'], native_ids, shadow_ids)):
            position = seq.num_completion_tokens
            expected = truth[seq.request_id]
            assert seq.completion_token_ids == expected[:position], 'target stream diverged from fixed reference'
            old, new = old[:budget], new[:budget]
            record = {'cycle': state['cycle'], 'request_id': seq.request_id,
                      'output_position': position, 'budget': budget,
                      'native': old, 'official_score_fp32': new,
                      'native_accepted': accepted_prefix(old, expected[position:]),
                      'fp32_accepted': accepted_prefix(new, expected[position:])}
            state['records'].append(record)
            with (dest/'selector_rows.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            if old != new and not (dest/'first_difference.pt').exists():
                torch.save({'candidate_ids': ids[row].cpu(), 'unary_bf16': unary[row].cpu(),
                            'edges_bf16': edges[row].cpu(), 'edges_official_fp32': official[row].cpu(),
                            'record': record}, dest/'first_difference.pt')
        return native

    @torch.inference_mode()
    def propose(self, sequences, budgets):
        if not state['enabled']:
            return native_propose(self, sequences, budgets)
        state['rows'] = [(seq, min(budget, 7)) for seq, budget in zip(sequences, budgets) if budget > 0]
        try:
            result = native_propose(self, sequences, budgets)
            if state['cycle'] in {0, 1, 2, 16, 128, 512, 1024}:
                state['enabled'] = False
                for seq, _ in state['rows']:
                    context = self._dflash2_contexts[seq.seq_id]
                    cache = self.dflash2_draft_cache
                    positions = cache.context_positions_by_request[seq.request_id]
                    taps = context['hidden_taps']
                    anchor = context['anchor_position']
                    expected_positions = torch.arange(anchor-positions.shape[1], anchor, device=positions.device)[None]
                    assert torch.equal(positions, expected_positions), 'draft context position gap/shift'
                    assert all(value.shape[1] == positions.shape[1] for value in taps.values())
                    fresh = self.dflash2_draft_model.project_context_kv(taps, positions)
                    cached = cache.context_kv_by_request[seq.request_id]
                    delta = max((fresh[layer][part].float()-cached[layer][part].float()).abs().max().item()
                                for layer in fresh for part in (0, 1))
                    kwargs = dict(anchor_token_ids=torch.tensor([context['anchor_token_id']], device=positions.device),
                                  anchor_positions=torch.tensor([anchor], device=positions.device), hidden_taps=None,
                                  token_embedding=self.model.model.embed_tokens, lm_head=self.model.lm_head,
                                  context_positions=positions)
                    old_single, _ = self.dflash2_draft_model.forward_block(**kwargs, context_kv_by_layer=cached)
                    new_single, _ = self.dflash2_draft_model.forward_block(**kwargs, context_kv_by_layer=fresh)
                    check = {'cycle': state['cycle'], 'request_id': seq.request_id, 'anchor_position': anchor,
                             'context_length': positions.shape[1], 'max_kv_abs_difference': delta,
                             'cached_single': old_single.token_ids.tolist()[0],
                             'fresh_single': new_single.token_ids.tolist()[0],
                             'native_batched': list(next(p.token_ids for p in result if p.sequence_id == seq.seq_id))}
                    state['cache_checks'].append(check)
                    (dest/'cache_checks.json').write_text(json.dumps(state['cache_checks'], indent=2))
                    if args.capture_replay and (state['cycle'] == 0 or (state['cycle'] == 1024 and seq.request_id.endswith('000000'))):
                        def cpu(value):
                            if isinstance(value, torch.Tensor):
                                return value.detach().cpu()
                            if isinstance(value, dict):
                                return {k: cpu(v) for k, v in value.items()}
                            if isinstance(value, tuple):
                                return tuple(cpu(v) for v in value)
                            return value
                        boundaries = sorted({0, seq.num_prompt_tokens, *(
                            seq.num_prompt_tokens+r['output_position']-1
                            for r in state['records'] if r['request_id'] == seq.request_id
                        )})
                        torch.save({'context': cpu(context), 'positions': cpu(positions),
                                    'cached_kv': cpu(cached), 'boundaries': boundaries,
                                    'prompt_token_ids': seq.prompt_token_ids,
                                    'committed_token_ids': seq.token_ids,
                                    'check': check}, dest/f"capture_{state['cycle']:04d}_{seq.request_id}.pt")
                state['enabled'] = True
            state['cycle'] += 1
            if state['cycle'] % 128 == 0:
                print('DIAGNOSTIC', state['cycle'], 'cycles', len(state['records']), 'request cycles', flush=True)
            return result
        finally:
            state['rows'] = None

    DFlash2CandidateSelector.select_path = selector
    ModelRunner.propose_dflash2 = propose
    path = Path(__file__).with_name('benchmark_qwen38_dflash2_concurrency.py')
    benchmark = runpy.run_path(str(path), run_name='diagnostic_benchmark')
    globals_ = benchmark['main'].__globals__
    native_reset = globals_['reset_runtime_stats']
    def reset(engine):
        native_reset(engine)
        state['enabled'] = True
    globals_['reset_runtime_stats'] = reset
    sys.argv = [str(path), *remaining]
    benchmark['main']()
    rows = state['records']
    summary = {'request_cycles': len(rows), 'changed_proposals': sum(r['native'] != r['official_score_fp32'] for r in rows),
               'native_accepted': sum(r['native_accepted'] for r in rows),
               'fp32_shadow_accepted': sum(r['fp32_accepted'] for r in rows),
               'proposed': sum(r['budget'] for r in rows),
               'improved_cycles': sum(r['fp32_accepted'] > r['native_accepted'] for r in rows),
               'worsened_cycles': sum(r['fp32_accepted'] < r['native_accepted'] for r in rows),
               'scope': 'counterfactual acceptance on identical original cycle boundaries; not a changed-policy generation run'}
    (dest/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
