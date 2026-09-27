"""Complete-FWD selection and rejected algorithm controls; never substitute pilot timings."""
import hashlib
import json
from pathlib import Path
import random
import statistics

r=Path(__file__).resolve().parent
pilot_jobs=dict(splitq4=18549,splitq2=18549,hot4=18557,hot4r=18564,joint4=18564,
                hot6r=18573,hot6p=18573,hot6t=18575,async6=18577,async4=18577,cluster2=18587,cluster2s=18597)
failures=dict(hot4='Rejected: one-key FP64 regression at L128; unrounded denominator versus BF16 numerator.',
              hot6p='Rejected: late-live FP64 regression at L64; peak guard does not cover the error.',
              cluster2='Rejected: nonfinite output at L768; cross-CTA scratch lifetimes were not synchronized.')
descriptions=dict(splitq4='4-WG static query CTA; resident KV recomputed across query groups',
                  splitq2='2-WG static query CTA; more repeated KV projections',
                  joint4='One N128 WGMMA computes K,V,Q,gate from the same Z load; static query groups',
                  hot4='4-WG max-free normal path, FP32 denominator and stable on-chip retry',
                  hot4r='4-WG max-free path with scalar sum of BF16 probabilities',
                  hot6r='6-WG variant of scalar BF16 probability sum',
                  hot6p='6-WG FP32 denominator with peak-dominance retry guard',
                  hot6t='6-WG tensor-core P*V and P*1 from identical BF16 probabilities; stable retry',
                  async6='6-WG resident KV with PV outstanding until next QK wait',
                  async4='4-WG resident KV with PV outstanding until next QK wait',
                  cluster2='Two resident-QKV CTAs share bias through TMA multicast',
                  cluster2s='TMA multicast with cluster-wide projection/bias scratch ownership barriers')
hashes={};controls=[]
def read(name):
    path=r/name
    hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
def row_from(rec,L):
    rng=random.Random(92688);ratios=rec['paired_ratios']
    estimates=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(5000))
    return dict(length=L,ending=rec['ending'],baseline_ms=rec['baseline_us']/1000,
                candidate_ms=rec['candidate_us']/1000,speedup=rec['speedup'],
                reduction_percent=100*(1-1/rec['speedup']),ci95=[estimates[125],estimates[4874]],
                release_form=rec.get('release_form'))
for artifact,job in pilot_jobs.items():
    record=dict(artifact=artifact,job=job,description=descriptions[artifact],rows=[])
    if artifact in failures:
        record['status']=failures[artifact]
    else:
        for L in (384,768,1024):
            data=read('anthropic-%s-%d-%d.json'%(artifact,L,job));assert data['complete']
            for ending in (False,True):
                choices=[x for x in data['records'] if x['kind']=='paired_module' and x['ending']==ending]
                record['rows'].append(row_from(min(choices,key=lambda x:x['baseline_us']),L))
        record['status']='Selected; qualified18586/18598.' if artifact=='hot6t' else 'Not selected; numerical pilot passed.'
    build=read(artifact+'/build-ready.json')
    for name,digest in build['sha256'].items():assert hashlib.sha256((r/artifact/name).read_bytes()).hexdigest()==digest
    record['build']=build
    controls.append(record)

job=18598;initial_job=18586;artifact='hot6t';primary=[];previous=[]
for L in (64,128,384,768,1024):
    native=read('check-hot6t-%d-18575.json'%L)
    assert native['complete'] and len(native['records'])==6
    audit=read('audit-hot6t-%d-%d.json'%(L,initial_job));assert audit['complete']
for L in (128,384,768,1024):
    for tool in ('memcheck','racecheck','synccheck'):
        if L==1024 and tool=='racecheck':
            for case in ('mixed','all_masked','large_logits'):
                data=read('racecheck-hot6t-1024-%s-%d.json'%(case,job))
                assert data['complete'] and len(data['records'])==1
        else:
            which=job if L==1024 and tool=='synccheck' else initial_job
            data=read('%s-hot6t-%d-%d.json'%(tool,L,which))
            assert data['complete'] and len(data['records'])==3
log=(r/('algo-qualify-%d.log'%initial_job)).read_text()
assert log.count('ERROR SUMMARY: 0 errors')==7
assert log.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')==3
log=(r/('algo-finish-%d.log'%job)).read_text()
assert log.count('ERROR SUMMARY: 0 errors')==1
assert log.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')==3
for L in (384,768,1024):
    data=read('qualified-hot6t-%d-%d.json'%(L,job));assert data['complete']
    kinds=[x['kind'] for x in data['records']]
    for kind,n in [('correctness',10),('fp64_full_module_sample',6),('fullgraph_compile',2),
                   ('changed_input_weight_mask_graph',2),('no_grad',2)]:assert kinds.count(kind)==n,(L,kind)
    data=read('anthropic-hot6t-%d-%d.json'%(L,job));assert data['complete']
    assert data['rounds']==64 and data['replays']==40
    for ending in (False,True):
        choices=[x for x in data['records'] if x['kind']=='paired_module' and x['ending']==ending]
        primary.append(row_from(min(choices,key=lambda x:x['baseline_us']),L))
        prev=next(x for x in data['records'] if x['kind']=='paired_previous' and x['ending']==ending)
        previous.append(row_from(prev,L))
        profile=next(x for x in data['records'] if x['kind']=='profile' and x['ending']==ending and x['path']=='ours')
        for stage in ('qkv_attention_inference','inference_ln_bias','inference_residual','nvjet'):
            assert any(stage in e['kernel'] for e in profile['events']),stage
assert all(row['ci95'][0]>1 for row in previous)
report=dict(selected=artifact,experiment_default=artifact,qualification_job=job,initial_qualification_job=initial_job,installed=False,primary_rows=primary,
            previous_rows=previous,controls=controls,evidence_sha256=hashes,
            ncu=read('algorithm-ncu.json'),all_masked_matches_anthropic=False)
(r/'algorithm-results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# QKV fusion algorithm follow-up','',
       'Selected experimental candidate: `hot6t + front8`. `candidate.load()` now selects it; '+
       '`candidate.load("resident6")` retains the old control. Production engine dispatch is unchanged.', '',
       'Whole FWD versus our previous fusion: L384 essentially unchanged (0.19–0.25% reduction), '+
       'L768 improves 6.37–6.57%, and L1024 improves 8.46–8.50%. Versus original Anthropic, '+
       'L384 is 18.78–20.08% faster and L768 is 1.74–3.30% faster, but L1024 remains 1.47–3.45% slower.', '',
       'This is inference-only native CUDA/TMA/WGMMA. K/V remain in CTA shared memory. Query and '+
       'gate projections are produced per query batch. Q/K/V/gate/LSE are not saved globally. '+
       'The new normal path computes both the attention numerator P*V and denominator P*1 on '+
       'Tensor Cores using identical BF16 P. It avoids repeated row-max updates and output '+
       'rescaling. A per-warpgroup guard retries unsafe tiles with the existing stable softmax '+
       'while QKV are still resident. Retries allocate no global projection tensors.', '',
       '## Qualified complete inference FWD versus Anthropic', '',
       'Node02 / normal_h100, B1/C128/H4/D32 BF16, 64 AB/BA rounds x 40 graph replays. '+
       'Same contract and source baseline as [the original comparison](ANTHROPIC_COMPARISON.md): '+
       'LN, bias, QKV, attention, gate, output projection, residual and orientation. The faster '+
       'of the two input-preserving Anthropic residual forms is selected per cell. Nontrivial '+
       'LN weights are BF16-representable to match upstream rounding. Positive reduction means faster.', '',
       '| L | Direction | Anthropic ms | New fusion ms | Time reduction | Speedup 95% CI |',
       '|---:|---|---:|---:|---:|---|']
for row in primary:
    lines.append('| %d | %s | %.4f | %.4f | %+.2f%% | %.4f–%.4f |'%
                 (row['length'],'ending' if row['ending'] else 'starting',row['baseline_ms'],row['candidate_ms'],
                  row['reduction_percent'],*row['ci95']))
lines+=['','## Paired comparison with our previous resident6 fusion','','| L | Direction | Previous ms | New ms | Time reduction | Speedup 95% CI |',
        '|---:|---|---:|---:|---:|---|']
for row in previous:
    lines.append('| %d | %s | %.4f | %.4f | %+.2f%% | %.4f–%.4f |'%
                 (row['length'],'ending' if row['ending'] else 'starting',row['baseline_ms'],row['candidate_ms'],
                  row['reduction_percent'],*row['ci95']))
lines+=['','## Algorithm controls','','Pilot times are for screening; qualified times above determine the result. '+
        'Each cell below is starting / ending ms. Raw paired Anthropic ratios for every control are in the JSON. '+
        'Candidates rejected for speed did not run the full sanitizer gate.', '',
        '| Candidate | L384 ms | L768 ms | L1024 ms | Status |','|---|---|---|---|---|']
for rec in controls:
    cells=[]
    for L in (384,768,1024):
        rr=[x for x in rec['rows'] if x['length']==L]
        cells.append(' / '.join('%.3f'%x['candidate_ms'] for x in rr) or '—')
    lines.append('| %s | %s | %s |'%(rec['artifact'],' | '.join(cells),rec['status']))
lines+=['']+[('- `%s`: %s.'%(name,descriptions[name])) for name in pilot_jobs]
lines+=['','## Validation and numerical limits','',
        'Thirty native fixtures at L64/128/384/768/1024 pass the unchanged FP64 gates, including '+
        'one-key, all-masked, late-live and large-logit cases. Runtime retry counters verify zero '+
        'retries for normal random inputs and positive retries for extreme logits and bias offsets. '+
        'A single retrying tile among normal tiles checks barrier phases across query batches. '+
        'Graph replays change between hot and retry paths without recapture.', '',
        'At L128/384/768/1024, memcheck, racecheck and synccheck each pass mixed, all-masked and '+
        'large-logit inputs with zero errors/hazards/warnings. Complete-module validation covers '+
        'both directions, five masks, sampled FP64, changed-input/weight/mask graphs, fullgraph '+
        'compilation, inference_mode and no_grad. Initial qualification18586 timed out in the '+
        'combined L1024 racecheck at 360 seconds; its incomplete racecheck result is not counted. '+
        'Job18598 reran each L1024 race fixture independently with a 600-second limit, then '+
        'completed the remaining module checks and timings. Source/binary hashes and actual profile dispatch '+
        'are verified. No backward qualification is claimed.', '',
        'Arithmetic is tolerance-qualified, not bitwise identical to the previous online softmax. '+
        'The all-masked behavior remains zero update, whereas Anthropic uses uniform-mean-V '+
        'attention; that existing semantic difference is not removed. Cold/hot path guard choices '+
        'are runtime data dependent. Large-logit fixtures intentionally exercise the slower stable retry.', '',
        '## HBM traffic after QKV fusion','',
        'NCU profiles only the fused core at L768 with five warmups, cache-control none and '+
        'clock-control none. Use paired full-FWD timings above for selection. The export produced '+
        'a Python site warning, but the reports and metric rows were written and parsed successfully.', '',
        '| Core | NCU ms | HBM read+write MB | L2 read+write GB | SM throughput % |',
        '|---|---:|---:|---:|---:|']
for rec in report['ncu']['records']:
    m=rec['metrics']
    lines.append('| %s | %.4f | %.1f | %.3f | %.2f |'%(rec['artifact'],m['gpu__time_duration.sum']['value'],
                 rec['dram_total_bytes']/1e6,rec['l2_read_write_bytes']/1e9,
                 m['sm__throughput.avg.pct_of_peak_sustained_elapsed']['value']))
lines+=['','The scalar hot4r control reduces HBM traffic but runs slower. The selected hot6t '+
        'runs faster while moving slightly more HBM data. Once global QKV buffers are removed, '+
        'the remaining on-chip arithmetic, register pressure, issue scheduling and synchronization '+
        'must also be optimized. Register spills still exist; source-level tensor removal does '+
        'not imply zero local-memory traffic. No SOL90 claim.', '',
        'Evidence: [machine-readable results](algorithm-results.json), [NCU data](algorithm-ncu.json), '+
        '`algo-qualify-18586.log`, `algo-finish-18598.log` and the per-shape JSON files. The benchmark and qualification '+
        'scripts are `algorithm.sbatch`, `algorithm_qualify.sbatch`, `anthropic_compare.py`, '+
        '`hot_audit.py`, `check.py`, and `module_check.py`.', '']
(r/'QKV_ALGORITHM_RESULTS.md').write_text('\n'.join(lines))
for row in primary:print('ANTHROPIC',row)
for row in previous:print('PREVIOUS',row)
