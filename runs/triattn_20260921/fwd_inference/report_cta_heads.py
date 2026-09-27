"""Publish completed layout experiments; this never changes the selected implementation."""
import collections
import hashlib
import json
from pathlib import Path

r=Path(__file__).resolve().parent
evidence={}
def read(name):
    p=r/name;evidence[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
pilots=read('cta-heads-pilots.json')
ncu=read('cta-heads-ncu.json');assert ncu['complete']
jobs={'cta_heads4_s2':18806,'cta_heads4_s1':18806,'cta_heads4_planar':18806,
      'cta_heads4_repack':18812,'cta_heads4_scalar':18824,'cta_heads4_indep':18824}
for art,j in jobs.items():
    for L in (64,128,384,768,1024):
        d=read(f'check-{art}-{L}-{j}.json');assert d['complete'] and len(d['records'])==6
    build=read(art+'/build-ready.json')
    for name,digest in build['sha256'].items():assert hashlib.sha256((r/art/name).read_bytes()).hexdigest()==digest
    assert len([x for x in pilots['records'] if x['artifact']==art])==6
for L in (128,384,768,1024):assert read(f'front-front8_hlast-{L}-18806.json')['complete']
front=read('cta-heads-front-pair-18818.json');assert front['complete']
for art in ('cta_heads4_s2','cta_heads4_repack'):
    for L in (64,128,768,1024):assert read(f'audit-{art}-{L}-18818.json')['complete']
    for tool in ('memcheck','racecheck','synccheck'):
        for case in ('mixed','all_masked','large_logits'):
            assert read(f'{tool}-{art}-128-{case}-18818.json')['complete']
    module=read(f'qualified-{art}-384-18818.json');assert module['complete']
    counts=collections.Counter(x['kind'] for x in module['records'])
    assert counts['fp64_full_module_sample']==6 and counts['fullgraph_compile']==2 and counts['changed_input_weight_mask_graph']==2
for tool in ('memcheck','racecheck','synccheck'):assert read(f'{tool}-front8_hlast-128-18818.json')['complete']
log=(r/'cta-heads-audit-18818.log').read_text()
assert log.count('ERROR SUMMARY: 0 errors')==14
assert log.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')==7
snapshot=json.loads((r.parent/'fwd_training/checkpoint18246/snapshot.json').read_text())['sha256']
pkg=r.parents[1]/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention'
for p,digest in snapshot.items():assert hashlib.sha256((pkg/p).read_bytes()).hexdigest()==digest,p
best=[]
for L in (384,768,1024):
    choices=[a for a in jobs if a not in ('cta_heads4_planar','cta_heads4_indep')]
    def cells(art):return sorted([x for x in pilots['records'] if x['artifact']==art and x['length']==L],key=lambda x:x['ending'])
    art=min(choices,key=lambda a:sum(x['vs_selected']['candidate_ms'] for x in cells(a)))
    best.extend(cells(art))
report=dict(complete=True,selected_changed=False,production_changed=False,all_candidates_rejected=True,
            sol90=False,training_preserved_files=len(snapshot),pilots=pilots,front=front,ncu=ncu,best_head_last=best,
            qualification_scope='s2/repack: native/retry graphs all long lengths, sanitizers L128, full module L384; other candidates numerical pilots only',
            evidence_sha256=evidence)
assert all(x['vs_selected']['speedup']<1 for x in pilots['records'])
(r/'cta-heads-results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Head-last bias and four-head attention CTA results','',
       '2026-09-26: implemented contiguous BF16 `[1,L,L,4]` bias and one CTA covering all four heads. '+
       'Six CUDA/TMA variants were measured. None beats the current selected complete inference FWD; '+
       'the selected entry remains L384 `h4kv_local1`, L768 `hot6t`, L1024 `hot4t`, all with `front8`. '+
       'Production dispatch and all 48 training18246 files remain unchanged. SOL90 is not achieved.','',
       '## Best head-last candidate by length','',
       'Node02 / normal_h100, BF16 B1/C128/H4/D32, complete input-preserving inference FWD, '+
       '24 balanced AB/BA rounds x30 graph replays. Every projection, LN, bias, attention, gate and '+
       'residual is included. No bias-layout conversion is performed in these measured paths. '+
       'Times are separate medians; percentage changes use median paired ratios. These are rejected pilots, not qualified promotions.','',
       '| L | Direction | Candidate | Current ms | Head-last ms | Slower than current | Anthropic ms | Slower than Anthropic |',
       '|---:|---|---|---:|---:|---:|---:|---:|']
for x in best:
    p=x['vs_selected'];q=x['vs_anthropic']
    lines.append('| %d | %s | %s | %.4f | %.4f | %+.2f%% | %.4f | %+.2f%% |'%(x['length'],'ending' if x['ending'] else 'starting',x['artifact'],p['baseline_ms'],p['candidate_ms'],-p['time_reduction_percent'],q['baseline_ms'],-q['time_reduction_percent']))
lines+=['','## All controls','','Starting / ending complete-FWD milliseconds:','',
        '| Candidate | L384 | L768 | L1024 |','|---|---:|---:|---:|']
for art in jobs:
    cells=[]
    for L in (384,768,1024):
        xs=sorted([x for x in pilots['records'] if x['artifact']==art and x['length']==L],key=lambda x:x['ending'])
        cells.append(' / '.join('%.3f'%x['vs_selected']['candidate_ms'] for x in xs))
    lines.append('| '+art+' | '+' | '.join(cells)+' |')
lines+=['','`s2`: two packed KV/bias stages, one shared all-head completion barrier, direct vector bias reads. '+
        '`s1`: one stage. `repack`: in-place shared-memory conversion to head planes, then ldmatrix reads; '+
        'no extra HBM tensor. `scalar`: direct reads of just the required head value. '+
        '`planar`: same shared four-head CTA using head-major global bias and ldmatrix. '+
        '`indep`: planar control with independently loaded and retired per-head KV/bias rings. '+
        'All candidates share normalized Z during Q/gate projection.','',
        '## Producer-only ablation','','The old front already writes `[1,4,L,L]` directly, so there was no separate transpose '+
        'kernel or `[1,L,L,4]` intermediate to remove. The new front writes one eight-byte vector per '+
        'query/key pair. Z and logical bias are bitwise equal, including changed-input/affine/weight/mask '+
        'graph replay. Layout conversion appears only in untimed validation. Matched 48x40 producer graphs:','',
        '| L | Direction | Planar us | Head-last us | Time reduction |','|---:|---|---:|---:|---:|']
for x in front['records']:
    lines.append('| %d | %s | %.2f | %.2f | %+.2f%% |'%(x['length'],'ending' if x['ending'] else 'starting',x['baseline_us'],x['candidate_us'],100*(1-1/x['speedup'])))
lines+=['','## Profiling and interpretation','',
        'Warm-cache L768 NCU measured `s2` attention at 1.914 ms and 604.7 MB DRAM traffic (18809), '+
        'versus the existing row-local streaming attention at 1.104 ms and 604.9 MB (fresh18831). '+
        'The separate joint KV projector adds approximately 0.160 ms and 430.8 MB in both paths. '+
        'The score matrix is never written to HBM. Sharing Z reduces L2 read sectors, but DRAM traffic '+
        'is essentially unchanged because the previous CTA order already reused it through cache. '+
        'Short-scoreboard stalls rise from 0.363 to 3.686 per issued instruction; barrier stalls '+
        'rise from 1.054 to 1.531. The planar common-barrier control takes1.430ms and the independent '+
        'head-ring control1.306ms, both at approximately604.8MB; their short-scoreboard stalls remain '+
        'near0.36 while barrier stalls are2.104 and1.320 respectively. Packed bias read dependencies and grouping/synchronization costs '+
        'outweigh the producer gain. NCU is diagnostic; the paired full-FWD runs determine the result. '+
        'This rejects these implementations, not every possible head-last algorithm.','',
        '## Validation and limits','',
        'Every candidate passes 30 native fixtures: six cases at L64/128/384/768/1024, including sampled '+
        'FP64, masks, large logits and changed-input/weight/bias graphs. `s2` and `repack` additionally '+
        'pass retry-path/changed-path graphs at64/128/768/1024, all three sanitizers at128, and complete '+
        'module FP64/masks/affine/changed-input-weight-mask/fullgraph/no_grad tests at384 (18818). '+
        'The front passes independent FP64 at128/384/768/1024 and all three sanitizers at128. '+
        'No full-length sanitizer qualification or production promotion is claimed. Original Anthropic '+
        'all-masked behavior still differs: our zero update versus upstream mean-V; no unrestricted '+
        'parity claim. Original source hashes and actual profiled kernel names are recorded per pilot.','',
        'Jobs18806,18812,18824 are native/full-FWD pilots;18809/18831 are NCU;18818 is validation and '+
        'producer ablation. [Design](CTA_HEADS_DESIGN.md), [complete evidence](cta-heads-results.json), '+
        '[raw paired pilot summary](cta-heads-pilots.json), [profiling](cta-heads-ncu.json). '+
        'All six owned jobs completed with exit0. NCU exports emitted the pre-existing Python-site '+
        'encoding warning, but all CSV metrics, reports and process exits were verified.']
(r/'CTA_HEADS_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('CTA_HEADS_REJECTED',len(pilots['records']),'cells; training preserved',len(snapshot))
