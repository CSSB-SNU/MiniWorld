"""Summarize qualified inference-only module timing without confusing it with native pilots."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

ap = argparse.ArgumentParser()
ap.add_argument('--job',required=True)
ap.add_argument('--front',default='front8')
ap.add_argument('--dispatch-job')
a = ap.parse_args()
r = Path(__file__).resolve().parent
rows = []; evidence = {}
for L in (64,128,384,768,1024):
    d = json.loads((r/('check-resident6-%d-18451.json'%L)).read_text())
    assert d['complete'] and len(d['records'])==6
for L in (128,384,768,1024):
    d = json.loads((r/('%s-check-%d-%s.json'%(a.front,L,a.job))).read_text())
    assert d['complete'] and len(d['records'])==12
    for tool in ('memcheck','racecheck','synccheck'):
        d = json.loads((r/('%s-resident6-%d-18463.json'%(tool,L))).read_text())
        assert d['complete'] and d['native_only']
core_log = (r/'sanitize-18463.log').read_text()
assert core_log.count('ERROR SUMMARY: 0 errors')==8
assert core_log.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')==4
front_log = (r/('qualify-%s.log'%a.job)).read_text()
assert front_log.count('ERROR SUMMARY: 0 errors')==2
assert front_log.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')==1
for L in (384,768,1024):
    path = r/('qualified-%s-resident6-%d-%s.json'%(a.front,L,a.job))
    d = json.loads(path.read_text()); assert d['complete']
    evidence[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert d['rounds']==64 and d['replays']==40
    for item in d['records']:
        if item['kind'] != 'paired_module': continue
        ratios = item['paired_ratios']; rnd = random.Random(92688)
        estimates = sorted(statistics.median(rnd.choices(ratios,k=len(ratios))) for _ in range(5000))
        rows.append(dict(length=L,ending=item['ending'],baseline_ms=item['baseline_us']/1000,
                         candidate_ms=item['candidate_us']/1000,speedup=item['speedup'],
                         reduction_percent=100*(1-1/item['speedup']),
                         speedup_ci95=[estimates[125],estimates[4874]]))
    dispatch = d
    if a.dispatch_job:
        pp = r/('dispatch-%s-resident6-%d-%s.json'%(a.front,L,a.dispatch_job))
        dispatch = json.loads(pp.read_text()); assert dispatch['complete']
        evidence[pp.name] = hashlib.sha256(pp.read_bytes()).hexdigest()
    for ending in (False,True):
        events = next(x['events'] for x in dispatch['records'] if x['kind']=='cuda_profile' and
                      x['ending']==ending and x['path']=='candidate')
        names = [x['kernel'] for x in events]
        assert any('qkv_attention_inference' in n for n in names)
        assert any('inference_ln_bias' in n for n in names)
        assert any('inference_residual' in n for n in names)
        assert not any('layer_norm_fwd' in n or 'training_fwd' in n or '_gate_out_fwd' in n for n in names)
for artifact in ('resident6',a.front):
    d = json.loads((r/artifact/'build-ready.json').read_text())
    for p,digest in d['sha256'].items(): assert hashlib.sha256((r/artifact/p).read_bytes()).hexdigest()==digest
report = dict(job=a.job,front=a.front,core='resident6',installed=False,rows=rows,evidence_sha256=evidence,
              dispatch_job=a.dispatch_job,
              qualification=dict(core_native=True,front_fp64=True,module_fp64=True,graph=True,
                                 fullgraph=True,core_sanitizers=True,front_sanitizers=True),
              cuda_graph_rounds=64,replays_per_round=40,
              baseline='actual installed eval/inference_mode module; complete FWD',
              builds={x:json.loads((r/x/'build-ready.json').read_text()) for x in ('resident6',a.front)})
(r/'results.json').write_text(json.dumps(report,indent=2)+'\n')
lines = ['# Qualified inference-only forward: internal baseline','',
         'These timings compare the installed ordinary eval module, not Anthropic\'s original fused block.',
         'Use [the Anthropic comparison](ANTHROPIC_COMPARISON.md) for the primary external baseline.','',
         'Explicit experiment: `resident6` + `%s`; production dispatch is unchanged.'%a.front,'',
         'Complete eval/inference_mode module, native CUDA/TMA plus cuBLAS output projection. '
         'Nonzero output weights, alternating AB/BA CUDA Graph timings (64 rounds x 40 replays), '
         'node02 / normal_h100. Time reduction uses the median paired ratio.','',
         '| L | Direction | Baseline ms | Inference ms | Time reduction | Speedup 95% bootstrap CI |',
         '|---:|---|---:|---:|---:|---|']
for row in rows:
    lo,hi = row['speedup_ci95']
    lines.append('| %d | %s | %.4f | %.4f | %.2f%% | %.4f–%.4f |'%
                 (row['length'],'ending' if row['ending'] else 'starting',row['baseline_ms'],
                  row['candidate_ms'],row['reduction_percent'],lo,hi))
lines += ['', 'Evidence: `qualified-%s-resident6-{384,768,1024}-%s.json`, '%(a.front,a.job)+
          '`results.json`, native/core checks18451 and sanitizers18463. The qualification job '
          'also checks the selected front against FP64 and runs memcheck/racecheck/synccheck.',
          '', 'The native core allocates only gated output. The full path has no Q/K/V/gate/LSE '
          'or LN-statistic saves. Resident6 still has register spills; source-level tensor '
          'removal is not a measurement of actual HBM bytes. No inference SOL90 claim.','']
(r/'RESULTS.md').write_text('\n'.join(lines))
for row in rows: print(row)
