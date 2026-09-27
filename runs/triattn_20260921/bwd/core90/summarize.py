"""Summarize the installed package and measured SOL, without inferring SOL from latency."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--final-job',type=int,required=True)
ap.add_argument('--bias-ncu',type=int,required=True)
ap.add_argument('--dq-ncu',type=int,required=True)
ap.add_argument('--node',choices=('node01','node02'),default=os.environ.get('SLURMD_NODENAME','node02'))
a=ap.parse_args();root=Path(__file__).resolve().parent
installed=json.loads((root/'installation.json').read_text())
final_log=(root/f'final-{a.final_job}.log').read_text()
for marker in ('DEFAULT_FLAGS_PASS','WGRAD_OPCHECK_PASS','GUARDS_PASS 10',
               'COLD_COMPILE_PASS','AMP_PASS torch.bfloat16 True','FP16_NATIVE_DECLINED'):
    assert marker in final_log, ('missing installed check', marker)
report={'final_job':a.final_job,'node':a.node,'installation':installed,'cases':[],'profiles':[],'sol90_achieved':False}
lines=['# Installed backward core improvements','',
       '**The two core kernels have not reached SOL90.** The installed changes reduce',
       'latency and memory traffic; they do not constitute a SOL90 result.','',
       'Baseline: installed job16663, including shared-Z Q/K/V/gate weight gradients.',
       'The baseline manifests and every hashed file are frozen in `baseline/`.',
       'Both directions use all input/parameter gradients and all five native fusion flags.','',
       f"Installed dK/dV: `{installed['bias_manifest.json']['artifact']}`; dQ: `{installed['dq_manifest.json']['artifact']}`.",
       'All new device code is native CUDA C++, using TMA and WGMMA.','',
       f'The current continuation runs on `{a.node}`, with low `normal_h100` QoS.',
       'All `core90/*.sbatch` defaults reflect that user constraint.','',
       '## Full module timing','',
       'Twelve balanced AB/BA rounds, fifteen CUDA graph replays per measurement.',
       'Time reductions use the median paired speed ratio; times are arm medians.','',
       '| L | Direction | Backward before ms | After ms | Reduction | F+B reduction |',
       '|---:|---|---:|---:|---:|---:|']
for length in (384,768,1024):
    data=json.loads((root/f'combined-{a.final_job}-L{length}.json').read_text())
    assert data['complete']
    assert not data.get('baseline_from_package',False), 'Campaign table must use the original frozen baseline'
    assert data.get('baseline_checkpoint','baseline')=='baseline', 'Wrong checkpoint for campaign table'
    for kind in ('bias','dq'):
        manifest=installed[kind+'_manifest.json']
        digest=data.get('binary_digests',{}).get('candidate',{}).get(kind)
        if digest is None:
            digest=hashlib.sha256(Path(data['candidate_binaries'][kind]).read_bytes()).hexdigest()
        assert digest==manifest['files'][manifest['binary']], ('stale installed timing',kind)
    for ending in (0,1):
        b=data['measurements'][f'backward_e{ending}'];fb=data['measurements'][f'forward_backward_e{ending}']
        assert max(b['gradient_relative_l2'])<.015 and max(fb['gradient_relative_l2'])<.015
        rec=dict(length=length,ending=bool(ending),backward=b,forward_backward=fb)
        report['cases'].append(rec)
        lines.append(f"| {length} | {'ending' if ending else 'starting'} | {b['baseline_us']/1000:.4f} | {b['candidate_us']/1000:.4f} | {100*(1-1/b['speedup']):.2f}% | {100*(1-1/fb['speedup']):.2f}% |")
lines+=['','## Measured L768 SOL','',
        'NCU `SpeedOfLight`: maximum measured compute/memory throughput. These are',
        'hardware utilization metrics, not ratios against previous execution times.','',
        '| Kernel | ms | Compute SOL | Memory SOL | Maximum |','|---|---:|---:|---:|---:|']
for kind,job in [('bias',a.bias_ncu),('dq',a.dq_ncu)]:
    proof=json.loads((root.parent/'below80'/f'profile-{job}.json').read_text())
    manifest=installed[kind+'_manifest.json']
    library='triattn_bias_fusion.so' if kind=='bias' else 'triattn_dq.so'
    assert proof['source_digests']['build/'+library]==manifest['files'][manifest['binary']]
    rows=list(csv.DictReader((root.parent/'below80'/f'profile-{job}.csv').open()))
    units=next(r for r in rows if not r['ID'].isdigit())
    for row in rows:
        if not row['ID'].isdigit():continue
        def number(key):return float(row[key].replace(',',''))
        scale={'ns':1e-6,'us':1e-3,'ms':1.,'s':1000.,
               'nsecond':1e-6,'usecond':1e-3,'msecond':1.,'second':1000.}[units['gpu__time_duration.sum']]
        ms=number('gpu__time_duration.sum')*scale
        sm=number('sm__throughput.avg.pct_of_peak_sustained_elapsed')
        mem=number('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed')
        name='dQ' if kind=='dq' else ('Bias final reduction' if 'reduce_bias' in row['Kernel Name'] else 'dK/dV + bias partial')
        rec=dict(kernel=name,job=job,ms=ms,compute_sol=sm,memory_sol=mem,sol=max(sm,mem))
        for metric,label in [('dram__bytes_read.sum','dram_read_bytes'),('dram__bytes_write.sum','dram_write_bytes')]:
            rec[label]=number(metric)*{'B':1.,'KB':1e3,'MB':1e6,'GB':1e9,
                                      'byte':1.,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9}[units[metric]]
        report['profiles'].append(rec)
        lines.append(f'| {name} | {ms:.4f} | {sm:.2f}% | {mem:.2f}% | {max(sm,mem):.2f}% |')
core=[p for p in report['profiles'] if p['kernel']!='Bias final reduction']
report['sol90_achieved']=all(p['sol']>=90 for p in core)
assert not report['sol90_achieved'],'Update the explicitly unmet-target report if both kernels actually reach90.'
lines+=['','## Changes and correctness','',
        '- Use register-source WGMMA operands for BF16 probability/dS. Reuse retired',
        '  resident shared tiles for TMA gradient stores; no extra HBM tensor.',
        '- R8 dK/dV retains deterministic FP32 bias partials. The partial tensor and',
        '  its final reduction reads are each half the previous R4 size. At L768,',
        '  the eliminated write plus read is 1.812 GB. R8 changes FP32 association,',
        '  so bias gradients are not required to be bitwise identical.',
        '- dQ retains the resident Q/dO/stats and double-buffered K/V/bias pipeline.',
        '- dQ overlaps probability calculations with its pending dP WGMMA group.',
        '  All reader barriers remain in place.',
        '- Loader metadata records actual row-group8; legacy group4 native calls',
        '  remain accepted for old callers and the A/B qualification harness.',
        '- Final artifacts are rebuilt with NVCC `--objdir-as-tempdir` to isolate',
        '  temporary files across the separate process namespaces used by tools.',
        '- FP64, none/mixed/one-key/all-masked cases, exact bias cancellation,',
        '  memcheck/racecheck/synccheck, all module gradients, SGD updates, frozen',
        '  parameters, fullgraph/cold compilation, opt-outs and AMP are checked.','',
        f"Bias qualification: job{installed['bias_manifest.json']['qualification_job']}. ",
        f"dQ qualification: job{installed['dq_manifest.json']['qualification_job']}; independent FP64 job{installed['dq_manifest.json']['fp64_job']}.",
        f'Installed package check: `final-{a.final_job}.log`. Full timings:',
        f'`combined-{a.final_job}-L*.json`. Current attribution is in `ATTRIBUTION.md`.','',
        '## Rejected directions and remaining work','',
        'See `EXPERIMENTS.md` and `results.json` for individual jobs. Larger key',
        'tiles, cross-CTA multicast, cooperative grouped-row dQ, extra shared',
        'probability staging, FP32 partial TMA stores and higher forced occupancy',
        'did not improve the selected end-to-end result. Register spills, extra',
        'synchronization and reduced tensor/warp overlap offset their savings.',
        'Two R4 CTAs combining dS through DSM also took53-60% longer in isolation',
        'at the target lengths despite keeping the same R8 global partial size.',
        'TMA eviction hints retaining K/V or bias gave no consistent dQ gain.',
        'The core SOL90 target remains open. No forward changes were made.']
if installed['bias_manifest.json']['artifact'].startswith('rs8_async'):
    i=lines.index('- Loader metadata records actual row-group8; legacy group4 native calls')
    lines[i:i]=[
        '- Two otherwise idle producer warps reduce the eight shared dS tiles',
        '  while the consumers compute the next query tile. Separate ready/empty',
        '  barriers protect both dS stages, including every reducer reader.',
        '  FP32 association and HBM partial volume are unchanged from R8.',
        f"  All full-module gradients are bitwise equal to {installed['bias_manifest.json'].get('core90_followup_baseline','checkpoint17069')}.",
        '  The producer/consumer register budgets are56/224 per thread.']
    if installed['bias_manifest.json']['artifact']=='rs8_async_q4':
        lines[i:i]=['- Q/dO/stats use four shared prefetch stages, eliminating local spills',
                    '  and reducing TMA waits without introducing another HBM tensor.']
(root/f'installed-results-{a.final_job}.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'installed_results.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'final_job':a.final_job,'sol90_achieved':report['sol90_achieved'],'profiles':report['profiles']},indent=2))
