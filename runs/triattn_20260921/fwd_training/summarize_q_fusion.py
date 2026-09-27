"""Collect raw experiment evidence without substituting profiling for timings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--promotion',type=int);a=ap.parse_args()
r=Path(__file__).resolve().parent
def read(n):return json.loads((r/n).read_text())
def ci(row):
    x=np.asarray(row['paired_ratios']);rng=np.random.default_rng(17756)
    return np.quantile(np.median(x[rng.integers(0,len(x),(10000,len(x)))],axis=1),[.025,.975]).tolist()
def metrics(job,artifact):
    name='qkv-profile-%d-%s.csv'%(job,artifact)
    rows=list(csv.DictReader((r/name).open()));units=rows[0]
    keys=['dram__bytes_read.sum','dram__bytes_write.sum','lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum',
          'gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed',
          'lts__throughput.avg.pct_of_peak_sustained_elapsed','launch__registers_per_thread',
          'launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem']
    out=[]
    for row in rows[1:]:
        rec=dict(kernel=row['Kernel Name'])
        for k in keys:
            value=float(row[k]);unit=units[k]
            if k=='gpu__time_duration.sum':value*=dict(ns=.001,us=1,ms=1000,s=1000000)[unit];unit='us'
            rec[k]=dict(value=value,unit=unit)
        out.append(rec)
    return dict(source=name,kernels=out,
                summed_hbm_read_mb=sum(x['dram__bytes_read.sum']['value'] for x in out),
                summed_hbm_write_mb=sum(x['dram__bytes_write.sum']['value'] for x in out),
                summed_l2_read_sectors=sum(x['lts__t_sectors_op_read.sum']['value'] for x in out),
                summed_profiled_us=sum(x['gpu__time_duration.sum']['value'] for x in out))
report=dict(date='2026-09-26',baseline_promotion=17628,sol90=False,experiments=[],profiles={},
            phase_probe=read('qkv-phase-17702.json'))
entries=[('resident_qkv',17677,17686,17693),('resident_qkv_reuse',17687,17701,17704),
         ('q_only_fused',17708,17714,17737),('resident_kv_parallel4b',17721,17728,None),
         ('resident_kv_parallel6b',17722,17729,None),('q_only_cta6',17727,17734,None),
         ('q_only_head4',17743,17745,17756)]
for artifact,build,bench,module in entries:
    rows=[]
    for L in (256,384,768,1024):
        data=read('qkv-core-%d-L%d.json'%(bench,L));assert data['complete']
        assert all(max(rec['projection_rel'])==0 and rec['combined_o_rel']==0 and rec['combined_lse_max']==0 for rec in data['records'])
        x=data['records'][0]
        rows.append(dict(length=L,baseline_us=x['baseline_us'],candidate_us=x['candidate_us'],speedup=x['speedup'],
                         six_cases_bitwise=True,raw='qkv-core-%d-L%d.json'%(bench,L)))
    entry=dict(artifact=artifact,build_job=build,core_job=bench,module_job=module,projection_attention=rows,
               build=read(artifact+'/build-ready.json'))
    if module:
        entry['full_module']=[]
        for L in (384,768,1024):
            name='qkv-module-%d-L%d.json'%(module,L);data=read(name);assert data['complete']
            for row in data['records']:
                item=dict(row,length=L,raw=name)
                item['reduction_pct']=(1-1/row['speedup'])*100
                item['paired_ratio_ci95']=ci(row)
                entry['full_module'].append(item)
    report['experiments'].append(entry)
for job,artifacts in ((17703,('baseline','resident_qkv_reuse')),(17740,('baseline','q_only_fused')),(17767,('baseline','q_only_head4'))):
    report['profiles'][str(job)]={art:metrics(job,art) for art in artifacts}
report['qualification']=dict(staged=17762,staged_extra=17765,sanitize=17766,
    initial_fullgraph_failure=17724,fullgraph_fix=17732,final_adapter=17741,partial_amp=17750,
    cancelled_unpublished_build=17712)
if a.promotion:
    report['promotion']=read('q-promotion-%d.json'%a.promotion)
    if report['promotion']['state']=='complete':
        report['installed_full_module']=[]
        for L in (384,768,1024):
            name='q-installed-bench-%d-L%d.json'%(a.promotion,L);data=read(name);assert data['complete']
            for row in data['records']:
                item=dict(row,length=L,raw=name,reduction_pct=(1-1/row['speedup'])*100,paired_ratio_ci95=ci(row))
                report['installed_full_module'].append(item)
(r/'qkv-fusion-results.json').write_text(json.dumps(report,indent=2)+'\n')
for entry in report['experiments']:
    print(entry['artifact'],[(x['length'],round(x['speedup'],4)) for x in entry['projection_attention']])
if 'installed_full_module' in report:
    for row in report['installed_full_module']:
        if row['kind']!='backward':print('INSTALLED',row['length'],row['ending'],row['kind'],row['baseline_us'],row['candidate_us'],row['reduction_pct'],row['paired_ratio_ci95'])
