"""Aggregate trial metrics per training seed, then bootstrap independent seeds."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root'); p.add_argument('--output',required=True)
a=p.parse_args(); groups=defaultdict(list)
for path in Path(a.root).rglob('metrics.json'):
    data=json.loads(path.read_text())
    if 'training_config' not in data: continue
    cfg=data['config']; tr=data['training_config']
    key=(tr['system'],tr['variant'],cfg['controller'],cfg['mode'],cfg['task'],cfg['sigma'],cfg['eta'])
    trials=data['trials']; completed=[t for t in trials if t['completed']]
    entry={'seed':tr['seed'],'source':str(path),'completion_rate':data['completion_rate']}
    if completed:
        for metric in ('cost','position_rmse_m','final_norm','median_latency_ms','p95_latency_ms'):
            values=[r[metric] for r in completed if r.get(metric) is not None]
            if values: entry[metric]=float(np.mean(values))
    if data['success_rate'] is not None: entry['success_rate']=data['success_rate']
    groups[key].append(entry)
rng=np.random.default_rng(123); result=[]
for key,entries in groups.items():
    seeds=[e['seed'] for e in entries]
    if len(set(seeds))!=len(seeds): raise ValueError(f'Duplicate training seeds for {key}: {seeds}')
    row={'group':dict(zip(('system','variant','controller','mode','task','sigma','eta'),key)),'seed_results':entries,'statistics':{}}
    metrics=set().union(*(e.keys() for e in entries))-{'seed','source'}
    for metric in sorted(metrics):
        v=np.array([e[metric] for e in entries if metric in e]); n=len(v)
        interval=None
        if n>1:
            boot=rng.choice(v,(10000,n),replace=True).mean(1)
            interval=np.quantile(boot,[.025,.975]).tolist()
        row['statistics'][metric]={'n_seeds':n,'mean':float(v.mean()),'std':float(v.std(ddof=1)) if n>1 else None,'bootstrap_95ci':interval}
    result.append(row)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False))
print(f'Saved {len(result)} groups to {a.output}')
