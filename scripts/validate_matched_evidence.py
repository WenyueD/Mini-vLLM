"""Validate saved actual inputs before treating profile runs as matched."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
read = lambda name: json.loads((root/name).read_text(encoding='utf-8'))
cfg = read('inputs.json')
official = read('official_evidence.json')
mini = read('mini_evidence.json')
merged = read('merged/mini_evidence.json')
assert official['complete'] and mini['complete'] and merged['complete']
v = official['verify']
assert v['total_tokens'] == 32
queries = {v['positions'][s]:v['input_ids'][s:e] for s,e in zip(v['query_start_loc'],v['query_start_loc'][1:])}
summary = {}
for label, data in [('mini',mini),('merged',merged)]:
    assert len(data['inputs']) == 4
    decisions = {x['sequence_id']:x['tokens'] for x in data['verify']['target_tokens']}
    accepted = []
    seen = set()
    for row in data['inputs']:
        i = cfg['prompts'].index(row['prefix'])
        assert i not in seen
        seen.add(i)
        assert row['query'] == queries[len(row['prefix'])]
        n = int(data['commit']['accepted_lengths'][str(row['sequence_id'])])
        assert official['outputs'][i][0] == row['query'][0]
        assert decisions[row['sequence_id']][:n] == official['outputs'][i][1:n+1]
        assert row['query'][1:n] == decisions[row['sequence_id']][:n-1]
        if n < len(row['query']):
            assert row['query'][n] != decisions[row['sequence_id']][n-1]
        accepted.append(n-1)
    summary[label] = {'actual_batch':len(seen),'accepted_proposals':accepted,
        'natural_proposals_equal_forced':all(x['natural_proposal']==x['query'][1:] for x in data['inputs'])}
assert mini['verify']['target_tokens']==merged['verify']['target_tokens']
(root/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
