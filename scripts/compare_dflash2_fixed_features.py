"""Separate target-feature differences from draft-core differences at one prefix."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from myvllm.spec_decode.dflash2 import DFlash2Config, DFlash2DraftModel, load_dflash2_weights_from_safetensors


def metrics(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    assert a.shape == b.shape
    return {'exact': torch.equal(a,b), 'max_abs': (a-b).abs().max().item(),
            'relative_l2': ((a-b).norm()/b.norm()).item(),
            'cosine': (torch.dot(a,b)/(a.norm()*b.norm())).item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mini-capture', type=Path, required=True)
    p.add_argument('--oracle-fixture', type=Path, required=True)
    p.add_argument('--oracle-tensors', type=Path, required=True)
    p.add_argument('--target-model', type=Path, required=True)
    p.add_argument('--draft-model', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    mini = torch.load(args.mini_capture, map_location='cuda', weights_only=False)
    oracle = torch.load(args.oracle_tensors, map_location='cuda', weights_only=False)
    fixture = json.loads(args.oracle_fixture.read_text())
    assert mini['committed_token_ids'][:-1] == fixture['prompt_token_ids'], 'different target prefix'
    assert mini['context']['anchor_position'] == fixture['anchor_position']
    cfg = DFlash2Config.from_pretrained_path(args.draft_model)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device('meta'):
        model = DFlash2DraftModel(cfg)
    torch.set_default_dtype(previous_dtype)
    model.to_empty(device='cuda'); model.eval()
    load_dflash2_weights_from_safetensors(model, args.draft_model, safe_open_device='cpu')
    oracle_taps = oracle['hidden_taps']
    if isinstance(oracle_taps, list):
        oracle_taps = {layer: value[None] for layer, value in zip(cfg.target_layer_ids, oracle_taps)}
    else:
        oracle_taps = {int(layer): value.reshape(1,-1,cfg.hidden_size) for layer,value in oracle_taps.items()}
    mini_taps = mini['context']['hidden_taps']
    context_length = next(iter(mini_taps.values())).shape[1]
    oracle_taps = {layer:value[:,-context_length:] for layer,value in oracle_taps.items()}
    anchor = int(fixture['anchor_token_id'])
    position = int(fixture['anchor_position'])
    index = json.loads((args.target_model/'model.safetensors.index.json').read_text())['weight_map']
    embed_name = next(k for k in index if k.endswith('embed_tokens.weight') and 'language_model' in k)
    with safe_open(str(args.target_model/index[embed_name]), framework='pt', device='cpu') as source:
        embedding = {token:source.get_slice(embed_name)[token:token+1].to('cuda')
                     for token in {anchor, cfg.mask_token_id}}
    def embed(ids):
        return torch.cat([embedding[i] for i in ids.reshape(-1).tolist()]).reshape(*ids.shape,cfg.hidden_size)
    with safe_open(str(args.target_model/index['lm_head.weight']),framework='pt',device='cpu') as source:
        head = source.get_tensor('lm_head.weight').to('cuda')
    def lm_head(hidden):
        return F.linear(hidden,head)
    captured = {}
    model.norm.register_forward_hook(lambda _m,_i,out: captured.update(proposal_hidden=out[:,1:].detach().clone()))
    for layer_id, layer in enumerate(model.layers):
        layer.register_forward_hook(lambda _m,_i,out,i=layer_id: captured.update({f'layer_{i}':out.detach().clone()}))
    original_edges = model.candidate_selector.score_candidate_edges
    results = {}
    with torch.inference_mode():
        for name, taps in [('oracle_features',oracle_taps),('mini_features',mini_taps)]:
            kwargs = dict(anchor_token_ids=torch.tensor([anchor],device='cuda'),
                          anchor_positions=torch.tensor([position],device='cuda'),hidden_taps=taps,
                          token_embedding=embed,lm_head=lm_head,
                          context_positions=torch.arange(position-next(iter(taps.values())).shape[1],position,device='cuda')[None],
                          return_debug=True)
            native,debug = model.forward_block(**kwargs)
            local_hidden = captured['proposal_hidden'].clone()
            traces = {key:value.clone() for key,value in captured.items() if key.startswith('layer_')}
            model.candidate_selector.score_candidate_edges = lambda ids,unary,h,a: original_edges(ids,unary.float(),h,a)
            fp32,_ = model.forward_block(**kwargs)
            model.candidate_selector.score_candidate_edges = original_edges
            context = model.combine_hidden_taps(taps)
            row = {'native_proposal':native.token_ids.tolist()[0], 'fp32_proposal':fp32.token_ids.tolist()[0],
                   'proposal_hidden_vs_oracle':metrics(local_hidden,oracle['proposal_hidden']),
                   'normalized_context_vs_oracle':metrics(context,oracle['normalized_context'].reshape(1,-1,cfg.hidden_size)[:,-context_length:]),
                   'candidate_token_ids':debug['candidate_token_ids']}
            if oracle.get('draft_layer_trace'):
                row['layer_output_vs_oracle'] = [metrics(traces[f'layer_{i}'],layer['output'])
                                                for i,layer in enumerate(oracle['draft_layer_trace'])]
            results[name] = row
    report = {'scope':'B1 draft-core replay on identical prefix, anchor and target-feature sources; not full-runtime equivalence',
              'oracle_fixture':str(args.oracle_fixture), 'mini_capture':str(args.mini_capture),
              'anchor_token_id':anchor,'mini_anchor_token_id':mini['context']['anchor_token_id'],
              'anchor_position':position, 'oracle_proposal':fixture['proposal_token_ids'],
              'target_tap_metrics':{str(layer):metrics(mini_taps[layer],oracle_taps[layer]) for layer in cfg.target_layer_ids},
              'replays':results}
    args.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
