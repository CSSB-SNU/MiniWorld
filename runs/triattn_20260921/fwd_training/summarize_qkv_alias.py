"""Reproduce the follow-up report from raw measurements and an installed checkpoint."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

ap=argparse.ArgumentParser()
ap.add_argument('--promotion',type=int,required=True)
a=ap.parse_args()
R=Path(__file__).resolve().parent
CONTROLS=[
    ('qkv_compact_alias',18193,18202), ('qkv_compact_alias4',18194,18203),
    ('qkv_compact_save_overlap',18195,18204), ('qkv_compact_retire6',18196,18209),
    ('qkv_compact_retire4',18197,18210), ('qkv_compact_pshared',18198,18211),
    ('qkv_compact_qcarry',18207,18215), ('qkv_compact_qcarry_retire',18208,18216),
    ('qkv_compact_biasprod',18226,18237), ('qkv_compact_biasprod5',18227,18238),
]


def read(name):return json.loads((R/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values):
    rng=random.Random(18240)
    medians=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(10000))
    return [medians[250],medians[9749]]


def timing(row):
    return dict(baseline_us=row['baseline_us'],candidate_us=row['candidate_us'],
                speedup=row['speedup'],time_reduction_pct=100*(1-1/row['speedup']),
                paired_ratios=row['paired_ratios'],median_ratio_bootstrap95=ci(row['paired_ratios']))


def profile(artifact):
    name='stream-profile-18235-%s.csv'%artifact
    rows=list(csv.DictReader((R/name).open()));units=rows.pop(0)
    def number(row,key):
        scale={'byte':1e-6,'Kbyte':1e-3,'Mbyte':1,'Gbyte':1e3,
               'ns':1e-6,'us':1e-3,'ms':1}[units[key]]
        return float(row[key])*scale
    return dict(evidence=name,length=1024,
        read_mb=sum(number(x,'dram__bytes_read.sum') for x in rows),
        write_mb=sum(number(x,'dram__bytes_write.sum') for x in rows),
        duration_ms=sum(number(x,'gpu__time_duration.sum') for x in rows),
        kernels=[dict(name=x['Kernel Name'],
            sm_pct=float(x['sm__throughput.avg.pct_of_peak_sustained_elapsed']),
            l2_pct=float(x['lts__throughput.avg.pct_of_peak_sustained_elapsed']),
            dynamic_shared_kb=float(x['launch__shared_mem_per_block_dynamic']),
            registers=float(x['launch__registers_per_thread'])) for x in rows])


promotion=read('qkv-promotion-%s.json'%a.promotion)
assert promotion['state']=='complete',promotion['state']
result=dict(promotion=promotion,baseline='checkpoint18006 / qg_scoped',
            installed_lengths=[1024],unchanged_lengths=[384,768],sol90_achieved=False,
            controls=[],full_module=[],profiles=dict(baseline=profile('baseline'),
                candidate=profile('qkv_compact_retire6')))
for artifact,build,bench in CONTROLS:
    manifest=read(artifact+'/build-ready.json')
    for path,digest in manifest['sha256'].items():assert sha(R/artifact/path)==digest
    item=dict(artifact=artifact,build_job=build,bench_job=bench,build=manifest,timings={})
    for length in [64,128,384,768,1024]:
        path='stream-core-%s-L%s.json'%(bench,length)
        data=read(path);assert data['complete'] and data['artifact']==artifact
        assert len(data['records'])==6
        for rec in data['records']:
            assert max(rec['projection_rel'])==rec['combined_o_rel']==rec['combined_lse_max']==0
            assert rec['attention_only_o_rel']==rec['attention_only_lse_max']==0
        item['timings'][str(length)]=dict(evidence=path,**timing(data['records'][0]))
    counts={}
    for section in (R/artifact/'sass.txt').read_text().split('Function :')[1:]:
        if 'ILi768' not in section.splitlines()[0]:continue
        groups=[];n=0
        for line in section.splitlines():
            if 'HGMMA.' in line:n+=1
            if 'WARPGROUP.DEPBAR' in line:
                if n:groups.append(n)
                n=0
        counts=dict(collections.Counter(groups))
    item['L768_hgmma_instructions_per_wait_histogram']=counts
    result['controls'].append(item)

for label,pattern,lengths in [
    ('pilot18214','stream-module-18214-L%s.json',[384,768,1024]),
    ('staged18240','stream-module-18240-L%s.json',[1024]),
    ('installed%s'%a.promotion,'qkv-installed-bench-%s-L%%s.json'%a.promotion,[1024]),
]:
    for length in lengths:
        name=pattern%length;data=read(name);assert data['complete']
        for row in data['records']:
            assert max(row['errors'])==0
            result['full_module'].append(dict(label=label,evidence=name,length=length,
                direction='ending' if row['ending'] else 'starting',kind=row['kind'],**timing(row)))
snapshot=read('checkpoint%s/snapshot.json'%a.promotion)
pkg=R.parents[1]/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention'
for path,digest in snapshot['sha256'].items():assert sha(pkg/path)==digest,path
result['installed_snapshot']=snapshot
result['qualification']=dict(staged=18232,cold_amp_guards=18233,
                            sanitizer_128_768=18222,sanitizer_1024=18234)
result['incomplete_measurement']=dict(job=18223,
    reason='CUDA-only profiler returned an empty event list for ending graph after long timing; result incomplete',
    resolution='CPU+CUDA graph attribution before long timing; complete rerun18240 and installed rerun')
result['complete']=True
(R/'qkv-alias-results.json').write_text(json.dumps(result,indent=2)+'\n')

lines=['# Full Q/K/V/gate fusion selected at L1024','',
    '2026-09-26. Installed promotion%s, `qkv_compact_retire6`, compared with frozen18006. '
    '**L1024 uses full Q/K/V/gate + attention fusion. L384/L768 retain Q+gate fusion. SOL90 remains unmet.**'%a.promotion,'',
    'H100, B1/H4/D32/C128 BF16, both directions, node02 / normal_h100. '
    'All six native outputs and existing backward saves are preserved. '
    'No backward algorithm or binary changed.','',
    '## Installed complete workload','',
    '64 AB/BA rounds x40 CUDA Graph replays per arm. Reduction is '
    '`100 * (1 - 1 / median(paired baseline/candidate ratios))`; arm medians need not reproduce that ratio. '
    'Both full FWD and F+B pass a bootstrap95% lower-bound speedup greater than1 before and after installation.','',
    '| Direction | FWD ms, baseline -> installed | FWD reduction | F+B ms, baseline -> installed | F+B reduction |',
    '|---|---:|---:|---:|---:|']
for direction in ['starting','ending']:
    selected={x['kind']:x for x in result['full_module'] if x['label']=='installed%s'%a.promotion and x['direction']==direction}
    f,b=selected['forward'],selected['forward_backward']
    lines.append('| %s | %.4f -> %.4f | %.2f%% | %.4f -> %.4f | %.2f%% |'%(
        direction,f['baseline_us']/1000,f['candidate_us']/1000,f['time_reduction_pct'],
        b['baseline_us']/1000,b['candidate_us']/1000,b['time_reduction_pct']))
lines += ['', 'The earlier complete16x20 pilot18214 and independent staged64x40 run18240 are '
    'retained in [qkv-alias-results.json](qkv-alias-results.json), including all paired rounds and '
    'confidence intervals. The L384 pilot is approximately tied, and L768 is slower, '
    'so neither length is selected for full fusion.','',
    '## What changed','',
    'Projection Z/weights and attention Q/bias have disjoint lifetimes. They now occupy the same '
    'shared-memory union. K/V are projected first into persistent shared tiles. After the Q+gate '
    'MMA retires its Z readers, the Q/gate outputs overwrite that dead Z tile and are saved for backward. '
    'A CTA barrier separates projection storage from attention reuse.','',
    'AtL1024, this permits four projection warpgroups instead of two, followed by six attention '
    'warpgroups. The selected kernel retires PV explicitly before continuing. It uses768 threads, '
    '230400B dynamic shared memory,80 registers and one CTA per SM. PTXAS reports16B spill stores '
    'and16B spill loads forL1024. The selected kernel is not spill-free or fully overlapped: '
    'emitted SASS still waits after each HGMMA even though C7515 is absent.','',
    'Separate4-consumer/2-producer candidates use104 consumer and32 producer registers after '
    'projection. Their61440-register requirement exactly matches the initial CTA pool. They restore '
    '8/2/4-instruction projection/QK/PV groups. Their schedule adds barrier work and reduces '
    'active attention warpgroups; measured speed is lower. Q-carry, shared-P and extra-stage '
    'controls also lose. The absence of a compiler warning alone is not an overlap or speed claim.','',
    'The register and asynchronous-operand ordering follows [NVIDIA WGMMA guidance]'
    '(https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/wgmma_programming.html). '
    'The actual emitted instruction groups and full-workload timings determine selection.','',
    '## All new native controls','',
    'Four projections plus attention, against installed18006.12 paired rounds x20 replays. '
    'Speedup greater than1 is faster. These are not full-module timings.','',
    '| Artifact | Build / bench | L384 | L768 | L1024 |','|---|---|---:|---:|---:|']
for x in result['controls']:
    lines.append('| %s | %s / %s | %s |'%(x['artifact'],x['build_job'],x['bench_job'],
        ' | '.join('%.4fx'%x['timings'][str(length)]['speedup'] for length in [384,768,1024])))
b,c=result['profiles']['baseline'],result['profiles']['candidate']
lines += ['', '## HBM and profiling limits','',
    'NCU18235, L1024 four-projection-plus-attention region: reads%.2f -> %.2fMB, '
    'writes%.2f -> %.2fMB; total traffic falls%.2f%%. Units are normalized from NCU MB/GB columns.'%(
        b['read_mb'],c['read_mb'],b['write_mb'],c['write_mb'],
        100*(1-(c['read_mb']+c['write_mb'])/(b['read_mb']+b['write_mb']))),'',
    'That isolated NCU measurement takes%.5f -> %.5fms and does not reproduce the full-module '
    'graph speedup. The retained paired full-module measurements are the performance selection '
    'evidence. The fused kernel has SM throughput59.75%% and L2 throughput38.30%%; neither establishes SOL90.'%(
        b['duration_ms'],c['duration_ms']),'',
    '## Qualification and dispatch','',
    '- All10 controls pass30 native fixtures each:5 lengths x6 mask/logit cases. '
    'Q/K/V/gate, O and LSE are bitwise equal; independent sampled FP64 and changed-input graph replay pass.',
    '- Selected staged qualification18232:18 full-module fixtures plus8 independent FP64 gradient fixtures. '
    'Output, all input/parameter gradients, dropout, optimizer updates, fullgraph and frozen parameters pass.',
    '- Cold verifier18233:fullgraph, custom-op schema/fake/AOT, BF16 AMP, mask=None, partial weights, '
    'weights-only gradients and guards pass atL1024.',
    '- Sanitizers18222 (L128/L768) and18234 (L1024):memcheck, racecheck and synccheck, '
    'mixed/all-masked; zero errors/hazards.',
    '- Long job18223 was incomplete because CUDA-only profiling returned no events for the ending graph. '
    'It is not promotion evidence. The harness now attributes three actual graph replays with CPU+CUDA '
    'events before timing; job18240 and the installed run complete both directions.',
    '- Fresh installed verification checks default full-QKV dispatch, the QKV opt-out, and unchanged '
    'L384/L768 QG dispatch. All preexisting code and binaries match18006 except the dispatcher '
    'and its associated manifests. Checkpoint%s records%s files.'%(a.promotion,len(snapshot['sha256'])),
    '', '`MINIWORLD_TRIATTN_QKV_FWD=0` restores18006 Q+gate fusion. Existing QG/Q/training-forward '
    'opt-outs also disable the new path. Unsupported shapes use the existing dispatch.', '',
    '[Promotion record](qkv-promotion-%s.json), [checkpoint](checkpoint%s/snapshot.json), '
    '[selected source](qkv_compact_retire6/fused.cu), [all evidence](qkv-alias-results.json), '
    '[previous full-QKV follow-up](QKV_STREAM_RESULTS.md), [previous installation](QG_FUSION_RESULTS.md).'%(
        a.promotion,a.promotion), '']
(R/'QKV_ALIAS_RESULTS.md').write_text('\n'.join(lines))
print('Wrote QKV_ALIAS_RESULTS.md / qkv-alias-results.json; verified installed checkpoint',a.promotion)
