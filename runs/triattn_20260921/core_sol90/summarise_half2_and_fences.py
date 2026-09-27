from pathlib import Path
import hashlib,json,re

HERE=Path(__file__).resolve().parent
installed=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so'
expected='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
assert hashlib.sha256(installed.read_bytes()).hexdigest()==expected
out=dict(installed_sha256=expected,installed_unchanged=True,sol90_achieved=False,
         installed_profile=dict(job=14525,us=793.536,sm_sol_pct=61.041971),
         compile_controls=[],half2_model=[])
for job,variant in [(14782,'basepackfencedep'),(14787,'basepackfenceqdep'),
                    (14800,'m128compactprod3p2dep'),(14802,'basepackfencewarpdep')]:
    s=(HERE/('build-'+variant+'.log')).read_text()
    assert 'BUILT '+variant in s
    properties=re.findall(r'Function properties for (\S+)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers',s)
    hot=[x for x in properties if '1073741824' in x[0] or 'attentionILb0E' in x[0]]
    assert len(hot)==1,hot
    warnings={code:len(re.findall(r'\('+code+r'\)',s)) for code in ['C7511','C7512','C7514','C7515','C7517','C7519','C7520']}
    assert warnings['C7512'] or warnings['C7520']
    out['compile_controls'].append(dict(job=job,variant=variant,rejected_before_gpu=True,
        hot_resources=dict(zip(['stack_bytes','spill_store_bytes','spill_load_bytes','registers'],map(int,hot[0][1:]))),warnings=warnings))
for direction in (0,1):
    d=json.loads((HERE/('half2-exp-model-%d.json'%direction)).read_text())
    approximate=[r for r in d['results'] if r['degree']!=0]
    assert all(r['finite'] and r['ratio']<=1.05 for r in approximate)
    costs=d['register_probe_us']
    out['half2_model'].append(dict(job=14796,ending=d['ending'],rows=d['rows'],
        max_model_rms_ratio=max(r['ratio'] for r in approximate),micro_us=costs,
        cubic_native_time_ratio=costs['3']/costs['0'],quartic_native_time_ratio=costs['4']/costs['0']))
out['half2_limits']='FP64 logits/model BF16 probability only; not actual attention qualification. Microprobe is not attention latency.'
out['compact_producer_limit']=dict(naive_budget_invalid=True,threads=544,warps=17,
    assumed_warps_for_hardware_cta_limit=20,registers_at_120=76800,hardware_cta_register_limit=65536,
    ptxas_entry_limit=96,source='/usr/local/cuda-12.9/include/cuda_occupancy.h:1496')
path=HERE/'half2-and-fence-results.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print(path)
