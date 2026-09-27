"""Summarize complete pilot cells only; this is not a promotion/qualification gate."""
from pathlib import Path
import json
import statistics

r=Path(__file__).resolve().parent
records=[]
for job in (18703,18706,18710,18716,18720,18729,18734):
    for path in sorted(r.glob(f'anthropic-*-{job}.json')):
        d=json.loads(path.read_text())
        if not d.get('complete'):continue
        for ending in (False,True):
            rows=[x for x in d['records'] if x['kind']=='paired_module' and x['ending']==ending]
            base=min(rows,key=lambda x:x['baseline_us'])
            old=next(x for x in d['records'] if x['kind']=='paired_previous' and x['ending']==ending)
            profile=next(x for x in d['records'] if x['kind']=='profile' and x['path']=='ours' and x['ending']==ending)
            by={}
            for e in profile['events']:by.setdefault(e['kernel'],[]).append(e['us'])
            record=dict(artifact=d['artifact'],length=d['length'],ending=ending,job=job,
                        candidate_ms=old['candidate_us']/1000,hot6t_ms=old['baseline_us']/1000,
                        reduction_vs_hot6t=100*(1-1/old['speedup']),
                        reduction_vs_anthropic=100*(1-1/base['speedup']),
                        stage_us={k:statistics.median(v) for k,v in by.items()})
            records.append(record)
            print('%-18s %4d %-5s %.4f ms hot6t %+6.2f%% upstream %+6.2f%%'%(
                d['artifact'],d['length'],str(ending),record['candidate_ms'],
                record['reduction_vs_hot6t'],record['reduction_vs_anthropic']))
(r/'head4-pilots.json').write_text(json.dumps(dict(qualified=False,records=records),indent=2)+'\n')
