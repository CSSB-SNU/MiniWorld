"""Read completed head-last experiments without modifying selection or older reports."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

ap=argparse.ArgumentParser();ap.add_argument('--jobs',nargs='+',type=int,required=True);a=ap.parse_args()
r=Path(__file__).resolve().parent
records=[];evidence={}
def row(t):
    xs=t['paired_ratios'];rng=random.Random(92698)
    boot=sorted(statistics.median(rng.choices(xs,k=len(xs))) for _ in range(3000))
    return dict(baseline_ms=t['baseline_us']/1000,candidate_ms=t['candidate_us']/1000,
                time_reduction_percent=100*(1-1/t['speedup']),speedup=t['speedup'],ci95=[boot[75],boot[2924]])
for j in a.jobs:
    for p in sorted(r.glob(f'anthropic-cta_heads*-{j}.json')):
        d=json.loads(p.read_text())
        if not d.get('complete'):continue
        evidence[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for ending in (False,True):
            upstream=min((x for x in d['records'] if x['kind']=='paired_module' and x['ending']==ending),key=lambda x:x['baseline_us'])
            previous=next(x for x in d['records'] if x['kind']=='paired_previous' and x['ending']==ending)
            profile=next(x for x in d['records'] if x['kind']=='profile' and x['path']=='ours' and x['ending']==ending)
            by={}
            for e in profile['events']:by.setdefault(e['kernel'],[]).append(e['us'])
            rec=dict(artifact=d['artifact'],length=d['length'],ending=ending,job=j,
                     rounds=d['rounds'],replays=d['replays'],vs_selected=row(previous),
                     vs_anthropic=row(upstream),stage_us={k:statistics.median(v) for k,v in by.items()})
            records.append(rec)
            print('%-22s %4d %-5s %.4f ms selected %+6.2f%% upstream %+6.2f%%'%(
                d['artifact'],d['length'],str(ending),rec['vs_selected']['candidate_ms'],
                rec['vs_selected']['time_reduction_percent'],rec['vs_anthropic']['time_reduction_percent']))
(r/'cta-heads-pilots.json').write_text(json.dumps(dict(qualified=False,records=records,evidence_sha256=evidence),indent=2)+'\n')
