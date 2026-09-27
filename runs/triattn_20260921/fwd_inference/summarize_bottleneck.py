"""Summarize actual selected kernels; profiling is diagnostic, not a speedup claim."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--job',required=True,type=int);a=ap.parse_args()
r=Path(__file__).resolve().parent
units={'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9,'ns':1e-9,'us':1e-6,'ms':1e-3,'s':1}
records=[];evidence={}
keys=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum',
      'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
      'lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum',
      'lts__throughput.avg.pct_of_peak_sustained_elapsed',
      'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
      'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed',
      'sm__warps_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active',
      'launch__shared_mem_per_block_dynamic','launch__registers_per_thread']
for L,art in [(384,'h4kv_local1'),(768,'hot6t'),(1024,'hot4t')]:
    p=r/f'bottleneck-{art}-{L}-{a.job}.csv'
    assert json.loads(p.with_suffix('.json').read_text())['complete']
    evidence[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows=list(csv.DictReader(p.open()));u=rows[0]
    for d in rows[1:]:
        if not d.get('ID','').isdigit():continue
        ks=keys+[k for k in d if 'stalled' in k and 'per_issue_active' in k]
        metrics={k:dict(value=float(d[k].replace(',','')),unit=u[k]) for k in ks}
        sec=metrics['gpu__time_duration.sum']['value']*units[u['gpu__time_duration.sum']]
        hbm=sum(metrics[k]['value']*units[u[k]] for k in ('dram__bytes_read.sum','dram__bytes_write.sum'))
        l2=32*sum(metrics[k]['value'] for k in ('lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum'))
        rec=dict(length=L,artifact=art,kernel=d['Kernel Name'],kernel_ms=sec*1000,
                 hbm_bytes=hbm,hbm_TBps=hbm/sec/1e12,l2_sector_bytes=l2,l2_sector_TBps=l2/sec/1e12,metrics=metrics)
        records.append(rec)
        print(L,art,d['ID'],'ms',round(sec*1e3,4),'HBM GB',round(hbm/1e9,4),'TBps',round(rec['hbm_TBps'],3),'L2 sector GB',round(l2/1e9,3))
old=json.loads((r/'head4-results.json').read_text())
stages=[]
for x in old['stages']:
    if x['ending']:continue
    parts={}
    for k,v in x['kernels'].items():
        if 'inference_ln_bias' in k:part='ln_bias'
        elif 'head4_kv_projection' in k:part='kv_projection'
        elif 'attention' in k:part='attention_core_including_qg_or_qkv'
        elif 'nvjet' in k:part='output_projection'
        elif 'inference_residual' in k:part='residual'
        else:part='other'
        parts[part]=parts.get(part,0)+v
    stages.append(dict(length=x['length'],source_job=18745,scope='synchronized profiler stage medians, not additive graph timings',microseconds=parts))
counts=[]
for L in (384,768,1024):
    scores=4*L**3;activation=2*128*L**2
    counts.append(dict(length=L,score_elements=scores,qk_pv_flops=128*scores,
                       denominator_extra_flops=16*scores,bias_logical_bytes=2*scores,
                       bias_unique_bytes=2*4*L**2,output_tensor_bytes=activation,
                       output_residual_fusion_removable_bytes=2*activation,
                       idealized_saving_us_at_3TBps=2*activation/3e12*1e6))
report=dict(complete=True,job=a.job,selected_kernels=records,prior_full_stage_profiles=stages,
            arithmetic_and_io_model=counts,evidence_sha256=evidence,
            method='five warmups; cache-control none; clock-control none; 20 NCU passes per kernel; actual selected artifacts',
            limits='Aggregate utilization and stall counters establish candidates, not a unique causal bottleneck. No candidate kernel or performance improvement is introduced.')
(r/'bottleneck-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
