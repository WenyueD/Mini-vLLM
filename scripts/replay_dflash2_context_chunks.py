"""Replay saved draft context with its original projection chunk boundaries."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from myvllm.spec_decode.dflash2 import DFlash2Config, DFlash2DraftModel, load_dflash2_weights_from_safetensors


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--target-model', type=Path, required=True)
    p.add_argument('--draft-model', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    saved = torch.load(args.capture, map_location='cuda', weights_only=False)
    cfg = DFlash2Config.from_pretrained_path(args.draft_model)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device('meta'):
        model = DFlash2DraftModel(cfg)
    torch.set_default_dtype(previous_dtype)
    model.to_empty(device='cuda')
    model.eval()
    load_dflash2_weights_from_safetensors(model, args.draft_model, safe_open_device='cpu')
    context, positions, cached = saved['context'], saved['positions'], saved['cached_kv']
    taps = context['hidden_taps']
    begin, end = int(positions[0, 0]), int(positions[0, -1])+1
    rebuilt = {k: tuple(torch.empty_like(x) for x in v) for k, v in cached.items()}
    coverage = torch.zeros(end-begin, dtype=torch.int32, device='cuda')
    boundaries = saved['boundaries']
    segments = []
    with torch.inference_mode():
        whole = model.project_context_kv(taps, positions)
        for left, right in zip(boundaries, boundaries[1:]):
            if right <= begin or left >= end:
                continue
            assert right <= end
            available = max(left, begin)
            selected = {layer: F.pad(value[:, available-begin:right-begin], (0, 0, available-left, 0))
                        for layer, value in taps.items()}
            pos = torch.arange(left, right, device='cuda')[None]
            projected = model.project_context_kv(selected, pos)
            for layer in rebuilt:
                for part in (0, 1):
                    rebuilt[layer][part][:, :, available-begin:right-begin].copy_(
                        projected[layer][part][:, :, available-left:])
            coverage[available-begin:right-begin] += 1
            segments.append([left, right])
        assert bool(torch.all(coverage == 1)), 'missing or duplicated context position'
        def delta(other):
            return {str(layer): {name: {
                'exact': torch.equal(cached[layer][part], other[layer][part]),
                'max_abs': (cached[layer][part].float()-other[layer][part].float()).abs().max().item(),
                'relative_l2': ((cached[layer][part].float()-other[layer][part].float()).norm()/cached[layer][part].float().norm()).item(),
            } for part, name in enumerate(['key','value'])} for layer in cached}
        index = json.loads((args.target_model/'model.safetensors.index.json').read_text())['weight_map']
        embed_name = next(k for k in index if k.endswith('embed_tokens.weight') and 'language_model' in k)
        with safe_open(str(args.target_model/index[embed_name]), framework='pt', device='cpu') as source:
            tokens = [context['anchor_token_id'], cfg.mask_token_id]
            embedding = {token: source.get_slice(embed_name)[token:token+1].to('cuda') for token in tokens}
        def embed(ids):
            return torch.cat([embedding[i] for i in ids.reshape(-1).tolist()]).reshape(*ids.shape, cfg.hidden_size)
        with safe_open(str(args.target_model/index['lm_head.weight']), framework='pt', device='cpu') as source:
            head = source.get_tensor('lm_head.weight').to('cuda')
        def lm_head(hidden):
            return F.linear(hidden, head)
        proposals = {}
        for name, kv in [('cached',cached), ('original_chunks',rebuilt), ('whole_window',whole)]:
            result, _ = model.forward_block(torch.tensor([context['anchor_token_id']],device='cuda'),
                torch.tensor([context['anchor_position']],device='cuda'), None, embed, lm_head,
                context_positions=positions, context_kv_by_layer=kv)
            proposals[name] = result.token_ids.tolist()[0]
    report = {'capture':str(args.capture), 'window':[begin,end], 'segments':len(segments),
              'original_chunk_replay':delta(rebuilt), 'whole_window_replay':delta(whole),
              'proposals':proposals, 'saved_proposals':saved['check']}
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
