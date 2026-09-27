"""Audit evidence and report output-fusion gains separately from core controls."""
import argparse,collections,csv,hashlib,json,random,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--selected-job',type=int,required=True)
a=p.parse_args();r=Path(__file__).resolve().parent;evidence={}
def read(name):
    f=r/name;evidence[name]=hashlib.sha256(f.read_bytes()).hexdigest()
    return json.loads(f.read_text())
def row(t,L):
    rng=random.Random(92696);xs=t['paired_ratios']
    boot=sorted(statistics.median(rng.choices(xs,k=len(xs))) for _ in range(5000))
    return dict(length=L,ending=t['ending'],baseline_ms=t['baseline_us']/1000,
                candidate_ms=t['candidate_us']/1000,time_reduction_percent=100*(1-1/t['speedup']),
                speedup=t['speedup'],speedup_ci95=[boot[125],boot[4874]])
def build(art):
    d=read(art+'/build-ready.json')
    for name,digest in d['sha256'].items():assert hashlib.sha256((r/art/name).read_bytes()).hexdigest()==digest
    return d
def module(d):
    assert d['complete'];counts=collections.Counter(x['kind'] for x in d['records'])
    for k,n in [('correctness',10),('fp64_full_module_sample',6),('fullgraph_compile',2),
                ('changed_input_weight_mask_graph',2),('no_grad',2)]:assert counts[k]==n,(k,counts)
boundary=[];output_pilots=[]
for art in ('outproj_c1','outproj_c2','outproj_c4'):
    build(art)
    for L in (64,128,384,768,1024):
        d=read(f'check-{art}-{L}-18937.json');assert d['complete'] and len(d['records'])==12
        for t in d['records']:
            assert t['relative']==0 and t['max_abs']==0
            if 'timing' in t:boundary.append(dict(artifact=art,**row(dict(ending=t['ending'],**t['timing']),L)))
    for L in (384,768,1024):
        d=read(f'anthropic-{art}-{L}-18937.json');assert d['complete']
        output_pilots += [dict(artifact=art,**row(t,L)) for t in d['records'] if t['kind']=='paired_previous']
for L in (128,384,768,1024):
    for tool in ('memcheck','racecheck','synccheck'):
        assert read(f'{tool}-outproj_c1-{L}-18946.json')['complete']
log=(r/'outproj-qualify-18946.log').read_text()
assert log.count('ERROR SUMMARY: 0 errors')==8
assert log.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')==4
primary=[];previous=[];stages=[]
for L,art in ((384,'h4kv_local1'),(768,'hot6t'),(1024,'hot4t')):
    build(art);module(read(f'qualified-outproj_c1-{L}-18946.json'))
    assert read(f'anthropic-outproj_c1-{L}-18946.json')['complete']
    d=read(f'output-selected-module-{L}-{a.selected_job}.json');module(d)
    assert d['actual_default_entry'] and d['selected_entry'] and d['out_artifact']=='outproj_c1'
    d=read(f'output-selected-anthropic-{L}-{a.selected_job}.json')
    assert d['complete'] and d['actual_default_entry'] and d['rounds']==64 and d['replays']==40
    for ending in (False,True):
        opts=[t for t in d['records'] if t['kind']=='paired_module' and t['ending']==ending]
        best=min(opts,key=lambda t:t['baseline_us'])
        primary.append(dict(release_form=best['release_form'],**row(best,L)))
        t=row(next(t for t in d['records'] if t['kind']=='paired_previous' and t['ending']==ending),L)
        assert t['speedup_ci95'][0]>1;t['baseline']='previous selected cores with cuBLAS+front8.post'
        previous.append(t)
        prof=next(t for t in d['records'] if t['kind']=='profile' and t['ending']==ending and t['path']=='ours')
        by={}
        for e in prof['events']:by.setdefault(e['kernel'],[]).append(e['us'])
        for name in ('inference_ln_bias','inference_output_residual','attention'):
            assert any(name in k for k in by),(L,name,by)
        assert not any('nvjet' in k or 'inference_residual<' in k for k in by)
        stages.append(dict(length=L,ending=ending,kernels={k:statistics.median(v) for k,v in by.items()}))
build('front8')
core_pilots=[];core_builds={}
for job,arts in [(18936,('pipe32_c4','serial32_c4','pipe32_c2')),
                 (18947,('pipecopy64_c2','pipecopy32_c4')),
                 (18961,('pipeclean64_c2','pipeclean32_c4'))]:
    log=(r/f'pipeline32-pilot-{job}.log').read_text()
    for art in arts:
        core_builds[art]=build(art)
        for L in (64,128,384,768,1024):
            d=read(f'check-{art}-{L}-{job}.json');assert d['complete'] and len(d['records'])==6
        for L in (64,128,768,1024):assert read(f'audit-{art}-{L}-{job}.json')['complete']
        for L in (384,768,1024):
            d=read(f'anthropic-{art}-{L}-{job}.json');assert d['complete']
            core_pilots += [dict(artifact=art,job=job,**row(t,L)) for t in d['records'] if t['kind']=='paired_previous']
        fragment=log.split('DTORCH_EXTENSION_NAME=triattn_infer_'+art+' ',1)[1].split('BUILD_READY',1)[0]
        core_builds[art]['compiler_serialization_diagnostics']=[ln for ln in fragment.splitlines() if 'Potential Performance Loss' in ln]
assert all(t['speedup']<1 for t in core_pilots), 'A winning core requires separate qualification and report selection'
traffic=[];units={'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9,'ns':1e-9,'us':1e-6,'ms':1e-3,'s':1}
for L in (384,768,1024):
    for direction in ('starting','ending'):
        stem=f'outprofile-outproj_c1-{L}-{direction}-18953'
        assert read(stem+'.json')['complete']
        f=r/(stem+'.csv');evidence[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
        data=list(csv.DictReader(f.open()));u=data[0];kernels=[]
        for t in data[1:]:
            if not t.get('ID','').isdigit():continue
            size=sum(float(t[k].replace(',',''))*units[u[k]] for k in ('dram__bytes_read.sum','dram__bytes_write.sum'))
            sec=float(t['gpu__time_duration.sum'].replace(',',''))*units[u['gpu__time_duration.sum']]
            kernels.append(dict(kernel=t['Kernel Name'],hbm_bytes=size,seconds=sec))
        new=next(t['hbm_bytes'] for t in kernels if 'inference_output_residual' in t['kernel'])
        old=sum(t['hbm_bytes'] for t in kernels if 'inference_output_residual' not in t['kernel'])
        traffic.append(dict(length=L,ending=direction=='ending',baseline_bytes=old,candidate_bytes=new,
                            reduction_percent=100*(1-new/old),kernels=kernels))
snapshot=json.loads((r.parent/'fwd_training/checkpoint18246/snapshot.json').read_text())['sha256']
pkg=r.parents[1]/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention'
for name,digest in snapshot.items():assert hashlib.sha256((pkg/name).read_bytes()).hexdigest()==digest,name
for name in ('candidate.py','module_check.py','anthropic_compare.py','outproj_check.py','outproj-qualify-18946.log'):
    evidence[name]=hashlib.sha256((r/name).read_bytes()).hexdigest()
report=dict(complete=True,qualified_output='outproj_c1',selected_job=a.selected_job,primary_rows=primary,
            previous_rows=previous,output_boundary=boundary,output_pilots=output_pilots,stages=stages,
            core_pilots=core_pilots,core_builds=core_builds,output_hbm=traffic,
            preserved_training_files=len(snapshot),production_changed=False,all_masked_matches_anthropic=False,
            evidence_sha256=evidence)
(r/'output-pipeline-results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Output fusion and single-head pipeline experiments','',
       'Qualified inference experiment default: existing per-length cores plus native `outproj_c1`. '+
       'Production dispatch and training remain unchanged. The four-head attention CTA direction stays closed.','',
       '## Complete inference FWD','',
       f'Node02, normal_h100; actual `candidate.load()` entry, job{a.selected_job};64 AB/BA rounds x40 CUDA Graph '+
       'replays; B1/C128/H4/D32 BF16, both directions. Includes LN, bias, all projections, attention, gate '+
       'and input-preserving residual. Compare with the faster of two original Anthropic residual forms. '+
       'Times are individual medians; reductions and confidence intervals use paired ratios.','',
       '| L | Direction | Anthropic ms | Ours ms | Reduction vs Anthropic | Reduction vs previous ours |',
       '|---:|---|---:|---:|---:|---:|']
for t,prev in zip(primary,previous):
    lines.append('| %d | %s | %.4f | %.4f | %.2f%% | %.2f%% |'%(t['length'],'ending' if t['ending'] else 'starting',t['baseline_ms'],t['candidate_ms'],t['time_reduction_percent'],prev['time_reduction_percent']))
lines+=['','## Selected output boundary','',
        'One128-thread CTA computes64x128 output values with WGMMA. TMA loads attention output, output '+
        'weight and residual; the residual load can overlap GEMM. The epilogue preserves '+
        '`BF16(X + BF16(A @ Wo.T))`, reuses completed input scratch for output, and TMA stores directly '+
        'in the required starting/ending orientation. X stays unchanged. Register count90, no spills, '+
        'dynamic shared memory66560B. No dedicated producer warpgroup is justified for this short boundary; '+
        'one elected thread issues the asynchronous transfers.','',
        'The projected-output temporary is eliminated. The idealized traffic model is5S to3S, '+
        'where S=2*L*L*128. Actual warmed NCU totals include caching and GEMM tiling:','',
        '| L | Direction | Previous HBM MB | Fused HBM MB | Reduction |',
        '|---:|---|---:|---:|---:|']
for t in traffic:
    lines.append('| %d | %s | %.1f | %.1f | %.1f%% |'%(t['length'],'ending' if t['ending'] else 'starting',t['baseline_bytes']/1e6,t['candidate_bytes']/1e6,t['reduction_percent']))
lines+=['','NCU18953 used five warmups, cache-control none and clock-control none; profiling is separate from '+
        'paired timings. The known NCU Python-site export warning occurred; all six reports, CSVs, '+
        'numerical checks and the Slurm job completed.','',
        'The2/4-warpgroup output variants pass the same numerical pilots but do not beat the1-group '+
        'boundary, so the simpler1-group candidate is selected. Only this selected output gets full qualification.','',
        '## Core controls','',
        'All controls retain single-head attention CTAs and resident KV. `serial32_c4` isolates the '+
        'smaller key tile. `pipe32` uses two alternating QK accumulators. `pipecopy` keeps QK accumulators '+
        'separate from scalar softmax work. `pipeclean` additionally drains before reading QK results, '+
        'then issues the next QK before current softmax. These are source-level schedules; compiler '+
        'serialization diagnostics are recorded in the JSON and prohibit claiming effective overlap.','',
        '| Candidate | L384 ms start/end | L768 ms start/end | L1024 ms start/end | Compiler warnings |',
        '|---|---:|---:|---:|---:|']
for art,b in core_builds.items():
    cells=[]
    for L in (384,768,1024):
        ts=sorted([t for t in core_pilots if t['artifact']==art and t['length']==L],key=lambda t:t['ending'])
        cells.append(' / '.join('%.3f'%t['candidate_ms'] for t in ts))
    lines.append('| '+art+' | '+' | '.join(cells)+f" | {len(b['compiler_serialization_diagnostics'])} |")
lines+=['','All seven core controls are slower than the previous selected core paths. They are rejected; '+
        'the three existing per-length cores remain selected. No core speedup or successful hardware '+
        'overlap is claimed. The smaller-tile serial control also loses, so serialization alone does not '+
        'explain the regression.','',
        'Core control timings use the historical cuBLAS tail, so compare them with their same-run '+
        'previous-selection rows in JSON, not the new fused-output times. Native FP64 fixtures and '+
        'retry/changed graphs pass; these pilots do not constitute full sanitizer qualification.','',
        '## Qualification and scope','',
        'Selected output: all60 output fixtures at L64/128/384/768/1024 in both directions are bitwise equal '+
        'to cuBLAS+post, including zeros, large values and cancellation. Independent FP64 samples, '+
        'out-of-place/input-preservation checks and changed-input/weight graph replay pass. All three '+
        'sanitizers pass at128/384/768/1024. Full module qualification and actual-default qualification cover '+
        'mask/affine fixtures, FP64 samples, changed-input/weight/mask graphs, no_grad and fullgraph '+
        f'at384/768/1024. All{len(snapshot)} training18246 files still match their snapshot.','',
        'Use `load()` for the new qualified experiment; `load(out_artifact=None)` retains the previous '+
        'per-length selection. Explicit `load("hot6t")` and other core names retain their historical '+
        'unfused tail unless `out_artifact="outproj_c1"` is given.','',
        'All-masked semantics remain ours zero update versus Anthropic mean-V; this is not an unrestricted '+
        'drop-in replacement. Inference only, no backward saves or gradient qualification. SOL90 is not achieved.','',
        'Initial output builds18932/18934 failed on CuTe namespace collisions and host binding ambiguity; '+
        'fixed before18937. No failed build is performance evidence. Evidence: '+
        '[JSON](output-pipeline-results.json), [design](PIPELINE_OUTPUT_DESIGN.md).']
(r/'PIPELINE_OUTPUT_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('OUTPUT_QUALIFIED',a.selected_job,'outproj_c1','training files',len(snapshot))
