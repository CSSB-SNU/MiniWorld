"""Qualified per-length selection, matched head ablations, and rejected controls."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True);a=ap.parse_args()
r=Path(__file__).resolve().parent
evidence={}
def read(name):
    p=r/name;evidence[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
def row(t,L):
    rng=random.Random(92696);xs=t['paired_ratios']
    boot=sorted(statistics.median(rng.choices(xs,k=len(xs))) for _ in range(5000))
    return dict(length=L,ending=t['ending'],baseline_ms=t['baseline_us']/1000,
                candidate_ms=t['candidate_us']/1000,reduction_percent=100*(1-1/t['speedup']),
                speedup=t['speedup'],ci95=[boot[125],boot[4874]],release_form=t.get('release_form'))

selection={384:'h4kv_local1',768:'hot6t',1024:'hot4t'}
for art,j,ls,als,pilot,L in [('h4kv_local1',18722,[128,384],[64,128,384],18716,384),
                           ('hot4t',18721,[1024],[1024],18710,1024)]:
    for length in (64,128,384,768,1024):
        d=read(f'check-{art}-{length}-{pilot}.json');assert d['complete'] and len(d['records'])==6
    for length in als:assert read(f'audit-{art}-{length}-{j}.json')['complete']
    for length in ls:
        for tool in ('memcheck','racecheck','synccheck'):
            for case in ('mixed','all_masked','large_logits'):
                assert read(f'{tool}-{art}-{length}-{case}-{j}.json')['complete']
    p=r/f'head4-qualify-{j}.log';log=p.read_text();evidence[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert log.count('ERROR SUMMARY: 0 errors')==len(ls)*6
    assert log.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')==len(ls)*3
    assert read(f'qualified-{art}-{L}-{j}.json')['complete']
    assert read(f'anthropic-{art}-{L}-{j}.json')['complete']
old=read('algorithm-results.json');assert old['selected']=='hot6t'

primary=[];previous=[];stages=[]
for L,art in selection.items():
    build=read(art+'/build-ready.json')
    for name,digest in build['sha256'].items():assert hashlib.sha256((r/art/name).read_bytes()).hexdigest()==digest
    module=read(f'selected-module-{L}-{a.job}.json');assert module['complete'] and module['selected_entry']
    kinds=[x['kind'] for x in module['records']]
    for kind,n in [('correctness',10),('fp64_full_module_sample',6),('fullgraph_compile',2),
                   ('changed_input_weight_mask_graph',2),('no_grad',2)]:assert kinds.count(kind)==n,(L,kind)
    data=read(f'selected-anthropic-{L}-{a.job}.json')
    assert data['complete'] and data['selected_entry'] and data['artifact']==art
    assert data['rounds']==64 and data['replays']==40
    for ending in (False,True):
        choices=[x for x in data['records'] if x['kind']=='paired_module' and x['ending']==ending]
        primary.append(row(min(choices,key=lambda t:t['baseline_us']),L))
        if L!=768:
            rec=row(next(x for x in data['records'] if x['kind']=='paired_previous' and x['ending']==ending),L)
            assert rec['ci95'][0]>1,rec
            previous.append(rec)
        prof=next(x for x in data['records'] if x['kind']=='profile' and x['path']=='ours' and x['ending']==ending)
        by={}
        for event in prof['events']:by.setdefault(event['kernel'],[]).append(event['us'])
        expected=['inference_ln_bias','inference_residual','nvjet']
        if L==384:expected+=['head4_kv_projection','qg_attention_head4kv<384, 1, 2, 1>']
        elif L==768:expected+=['qkv_attention_inference<768, 6, 1, 6>']
        else:expected+=['qkv_attention_inference<1024, 4, 2, 4>']
        for name in expected:assert any(name in k for k in by),(L,name)
        stages.append(dict(length=L,ending=ending,kernels={k:statistics.median(v) for k,v in by.items()}))

ablation=read('head4-projection-18732.json');assert ablation['complete']
first=read('head4-projection-18712.json');assert first['complete']
preserved=read('head4-training-preserved.json');assert preserved['complete'] and not preserved['mismatches']
for L in (128,384,768,1024):
    assert read(f'frontcheck-h4kv_front1-{L}-18733.json')['complete']
    for tool in ('memcheck','racecheck','synccheck'):
        assert read(f'front-{tool}-h4kv_front1-{L}-18733.json')['complete']
front_log=(r/'head4-front-check-18733.log').read_text()
assert front_log.count('ERROR SUMMARY: 0 errors')==8
assert front_log.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')==4
for name in ('candidate.py','native.py','anthropic_compare.py','module_check.py',
             'head4_projection_check.py','head4_front_check.py','head4_selected.sbatch',
             'head4-front-check-18733.log'):
    evidence[name]=hashlib.sha256((r/name).read_bytes()).hexdigest()
pilots=read('head4-pilots.json');ncu=read('head4-ncu.json')
for artifact in {x['artifact'] for x in pilots['records']}:
    build=read(artifact+'/build-ready.json')
    for name,digest in build['sha256'].items():assert hashlib.sha256((r/artifact/name).read_bytes()).hexdigest()==digest
report=dict(selection=selection,qualified=True,job=a.job,installed_in_production=False,
            primary_rows=primary,previous_rows=previous,stages=stages,head_ablation=ablation,
            initial_ablation=first,pilots=pilots,ncu=ncu,evidence_sha256=evidence,
            all_masked_matches_anthropic=False)
(r/'head4-results.json').write_text(json.dumps(report,indent=2)+'\n')

lines=['# K/V placement and four-head reuse','',
       f'Qualified experimental default: `candidate.load()` selects L384 `h4kv_local1`, L768 `hot6t`, '+
       f'and L1024 `hot4t`. Actual selected entry verified by job{a.job}. Production dispatch and all48 '+
       'training18246 files remain unchanged. No SOL90 claim.','',
       '## Complete inference FWD versus original Anthropic','',
       'Node02 / normal_h100, B1/C128/H4/D32 BF16, both orientations, 64 AB/BA rounds x40 graph '+
       'replays. Includes LN, bias, every projection, attention, gating and input-preserving residual. '+
       'Use the faster original residual form per cell. Percentages are paired ratios, times are separate medians.', '',
       '| L | Direction | Anthropic ms | Selected ms | Time reduction | Speedup 95% CI |',
       '|---:|---|---:|---:|---:|---|']
for t in primary:
    lines.append('| %d | %s | %.4f | %.4f | %+.2f%% | %.4f–%.4f |'%(t['length'],'ending' if t['ending'] else 'starting',t['baseline_ms'],t['candidate_ms'],t['reduction_percent'],*t['ci95']))
lines+=['','## Increment over previously qualified hot6t','','| L | Direction | hot6t ms | Selected ms | Time reduction |','|---:|---|---:|---:|---:|']
for t in previous:lines.append('| %d | %s | %.4f | %.4f | %+.2f%% |'%(t['length'],'ending' if t['ending'] else 'starting',t['baseline_ms'],t['candidate_ms'],t['reduction_percent']))
lines+=['','L768 retains hot6t; no improvement is claimed there.','',
        '## What four heads buy','','Four D32 heads form one dense N128 projection. The wide kernel loads Z once for both '+
        'K and V and all four heads. The strong narrow control also puts its four head CTAs next to '+
        'each other, so this comparison excludes the avoidable slow head-major grid. Both controls '+
        'have the same row-local attention kernel, bitwise-equal projections/full outputs, and changed-input '+
        'projection graph checks. Paired 48x40 measurements:','',
        '| L | Adjacent single-head KV us | Joint head4 KV us | Projection time reduction |',
        '|---:|---:|---:|---:|']
for t in ablation['records']:
    if t['kind']=='projection':lines.append('| %d | %.2f | %.2f | %.2f%% |'%(t['length'],t['baseline_us'],t['candidate_us'],100*(1-1/t['speedup'])))
lines+=['','The first head-major control was2.15–2.34x slower; much of that difference was input-cache '+
        'locality, not the projection width alone. Head-specific K/V values and softmax remain independent.','',
        '## K/V lifetime and CTA scheduling','','L384 materializes KV once and streams two small K/V+bias tiles per warpgroup. Query groups '+
        'and the four heads for one outer row are adjacent in grid X. Q/gate remain fused with attention. '+
        'L768/1024 retain fully on-chip QKV; L1024 uses four cooperative warpgroups and two bias stages, '+
        '126 registers and no spills, removing the old6+6+4 query tail. Its lower warp count loses at768, '+
        'so768 retains six warpgroups. There is no new dedicated producer warpgroup in these selected kernels.','',
        'At768 NCU reports streaming attention HBM traffic3.932→0.605GB when only the CTA grid order '+
        'changes; core time1.414→1.109ms. KV projection adds0.160ms and0.431GB, so combined '+
        'projection/core still loses to hot6t1.239ms/0.406GB. These are warm-cache NCU diagnostics '+
        '(17 passes/kernel), not the complete-FWD selection timings. The export emitted the known '+
        'Python-site warning, but reports, metric rows and exit status completed successfully.','',
        '## Controls','','Pilot complete-FWD times, starting / ending ms. Unselected candidates are not promoted '+
        'or represented as fully qualified.','',
        '| Candidate | L384 | L768 | L1024 |','|---|---:|---:|---:|']
for art in sorted({x['artifact'] for x in pilots['records']}):
    cells=[]
    for L in (384,768,1024):
        ts=sorted([x for x in pilots['records'] if x['artifact']==art and x['length']==L],key=lambda x:x['ending'])
        assert len(ts)==2,(art,L)
        cells.append(' / '.join('%.3f'%x['candidate_ms'] for x in ts))
    lines.append('| '+art+' | '+' | '.join(cells)+' |')
lines+=['','The LN/bias/KV front fusion removes a Z reread but loses to separate front8 plus KV projection. '+
        '128-bit shared-memory accesses improve that control without making it the winner. Scalar front '+
        'fusion passes front-specific bitwise Z/bias, sampled FP64 KV, graph and three-sanitizer checks '+
        'at128/384/768/1024 (18733); the vectorized control has numerical pilots only.','',
        '## Validation and scope','','Selected native sources pass30 FP64/edge fixtures each. L384 candidate has retry/changed-path '+
        'graphs at64/128/384 and all three sanitizers at128/384 (18722). L1024 candidate has retry/graph '+
        'and all three sanitizers at1024 (18721). Existing hot6t768 qualification is18586/18598. The selected '+
        f'default entry passes complete-module FP64, mask/affine cases, changed-input/weight/mask graphs, '+
        f'fullgraph and no_grad/inference_mode at all three lengths in{a.job}, with actual kernel attribution. '+
        'All-masked semantics remain ours zero-update versus upstream mean-V; LN affine fixtures match '+
        'upstream BF16 rounding. No unrestricted numerical parity or backward qualification is claimed.','',
        'Job18702 failed before compilation because sources had not been generated with the available '+
        'python3 executable;18703 reran successfully. Every subsequent pilot/control job completed.','',
        'Evidence: [results JSON](head4-results.json), [pilots](head4-pilots.json), [NCU](head4-ncu.json), '+
        '[design](HEAD4_KV_DESIGN.md), selected logs and per-shape JSONs.']
(r/'HEAD4_KV_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('HEAD4_QUALIFIED',a.job,selection)
