"""Attribute GPU events to CPU launch scopes using profiler correlation IDs."""
import json, sys, heapq
from pathlib import Path
from collections import defaultdict, Counter
root = Path(sys.argv[1])
result = {}
for path in sorted(root.glob('*.trace.json')):
    events = json.loads(path.read_text())['traceEvents']
    scopes = [e for e in events if e.get('ph') == 'X' and e.get('name','').startswith('diag/')]
    roots = sorted([e for e in scopes if e['name'] in ('diag/execute_model','diag/sample_and_next_draft','diag/verify','diag/commit')],key=lambda e:e['ts'])
    for i,e in enumerate(roots):
        e['_segment'] = f'{i}:{e["name"]}'
    timelines = defaultdict(list)
    for e in scopes:
        timelines[e['tid']].append((e['ts'],0,e))
    for e in events:
        if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{}):
            timelines[e['tid']].append((e['ts'],1,e))
    owners = {}
    for tid, timeline in timelines.items():
        heap = []
        for idx,(ts,kind,e) in enumerate(sorted(timeline,key=lambda x:(x[0],x[1]))):
            while heap and heap[0][2] <= ts:
                heapq.heappop(heap)
            if kind == 0:
                heapq.heappush(heap,(-ts,idx,ts+e['dur'],e['name']))
            else:
                segment = next((r['_segment'] for r in roots if r['tid']==tid and r['ts']<=ts<r['ts']+r['dur']), 'outside')
                owners[e['args']['correlation']] = (segment,heap[0][3] if heap else 'unannotated')
    groups = defaultdict(lambda:dict(gpu_us=0,count=0))
    kernels = defaultdict(lambda:dict(gpu_us=0,count=0))
    totals = defaultdict(lambda:dict(gpu_us=0,count=0))
    for e in events:
        if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):
            continue
        segment,owner = owners.get(e.get('args',{}).get('correlation'),('unknown','unknown'))
        for key,table in [(segment+' | '+owner,groups),(segment+' | '+e['name'],kernels),(segment,totals)]:
            table[key]['gpu_us'] += e.get('dur',0)
            table[key]['count'] += 1
    inclusive = defaultdict(lambda:dict(cpu_us=0,count=0))
    for e in scopes:
        inclusive[e['name']]['cpu_us']+=e['dur']
        inclusive[e['name']]['count']+=1
    result[path.name] = dict(segments=[dict(name=e['_segment'],cpu_ms=e['dur']/1000) for e in roots],
        gpu_totals=dict(totals),gpu_by_launch_scope=dict(sorted(groups.items(),key=lambda x:-x[1]['gpu_us'])),
        top_gpu_operations=dict(sorted(kernels.items(),key=lambda x:-x[1]['gpu_us'])[:40]),cpu_inclusive=dict(inclusive))
(root/'trace_summary.json').write_text(json.dumps(result,indent=2))
for name,data in result.items():
    print(name, json.dumps(data['segments']),json.dumps(data['gpu_totals']))
    for key,value in list(data['gpu_by_launch_scope'].items())[:15]:
        print(key,value)
