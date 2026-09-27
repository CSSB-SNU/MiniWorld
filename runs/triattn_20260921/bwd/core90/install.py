"""Install two qualified backward cores against the frozen six-manifest baseline."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PKG=ROOT.parents[2]/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda'
ap=argparse.ArgumentParser()
for name in ('bias','dq'):
    ap.add_argument('--'+name,required=True)
    ap.add_argument('--'+name+'-job',type=int,required=True)
ap.add_argument('--fp64-job',type=int,required=True)
ap.add_argument('--combined-job',type=int,required=True)
ap.add_argument('--bias-rows',type=int,required=True)
a=ap.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,content):
    temp=path.with_name(path.name+'.new');temp.write_text(content);os.replace(temp,path)
def write_json(path,data):write(path,json.dumps(data,indent=2)+'\n')
def copy(src,dst):
    temp=dst.with_name(dst.name+'.new');shutil.copy2(src,temp);os.replace(temp,dst)

# Check every input before mutating the installed package.
for f,d in json.loads((ROOT/'baseline/snapshot.json').read_text()).items():
    assert sha(PKG/f)==d,('intervening edit',f)
for job in (a.bias_job,a.dq_job):
    log=(ROOT.parent/'below80'/f'qualify-{job}.log').read_text()
    assert 'API_PASS 10' in log
    assert log.count('ERROR SUMMARY: 0 errors')>=4
    assert log.count('0 hazards displayed (0 errors, 0 warnings)')>=2
bias_fp64=json.loads((ROOT.parent/'below80'/f'fp64-{a.bias_job}.json').read_text())
assert len(bias_fp64)==24
cancellation=json.loads((ROOT.parent/'bias_fusion'/f'cancellation-{a.bias_job}.json').read_text())
assert cancellation['passed'] and cancellation['candidate_nonzero']==0
assert cancellation['artifact'].endswith('/'+a.bias)
for L in (64,128):
    for mask in ('none','mixed','one_key','all_masked'):
        d=json.loads((ROOT/f'fp64-{a.fp64_job}-L{L}-{mask}.json').read_text())
        assert d['length']==L and d['mask']==mask
        assert 'fp64_rms' in d or mask=='all_masked'
for L in (384,768,1024):
    d=json.loads((ROOT/f'combined-{a.combined_job}-L{L}.json').read_text())
    assert d['complete'] and len(d['measurements'])==4
    assert all(v['speedup']>1 and max(v['gradient_relative_l2'])<.015 for v in d['measurements'].values())
    assert f'/{a.bias}/' in d['candidate_binaries']['bias']
    assert f'/{a.dq}/' in d['candidate_binaries']['dq']
sources={}
for kind,folder,artifact,filename,lib in (
    ('bias','bias_fusion',a.bias,'grouped.cu','triattn_bias_fusion.so'),
    ('dq','dq',a.dq,'fused.cu','triattn_dq.so')):
    p=ROOT.parent/folder/artifact
    hashes=json.loads((p/'build-ready.json').read_text())
    for f,d in hashes.items():assert sha(p/f)==d,('stale build',f)
    assert '--objdir-as-tempdir' in (p/'build.log').read_text()
    job=a.bias_job if kind=='bias' else a.dq_job
    for length in (64,256):
        for tool in ('memcheck','racecheck','synccheck'):
            proof=json.loads((ROOT.parent/'below80'/f'{tool}-{job}-L{length}.json').read_text())
            assert proof['complete'] and proof['source_digests']==hashes
    if kind=='dq':
        assert hashes==json.loads((ROOT/f'fp64-{a.fp64_job}-L128-mixed.json').read_text())['native_build']
    name=(p/'module-name.txt').read_text().strip()
    sources[kind]=(p,filename,lib,name)

# The loader records the actual native row group on each distinct extension.
# Baseline extensions without the attribute retain their historical group4 ABI.
bp=(PKG/'bias_backward.py').read_text()
old='        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_EXT=module'
new=old.replace(';_EXT=module',";module.row_group=data.get('row_group',4);_EXT=module")
assert bp.count(old)==1;bp=bp.replace(old,new)
point='def _fake(q,k,v,b,m,out,dy,native_dq):'
helper='''def native_backward(q,k,v,b,m,delta,dy):
    ext=_extension()
    return ext.backward(q,k,v,b,m,delta,dy,getattr(ext,'row_group',4))

'''
assert point in bp;bp=bp.replace(point,helper+point)
bp=bp.replace('_extension().backward(q,k,v,b,m,delta,dy,4)','native_backward(q,k,v,b,m,delta,dy)')
gp=(PKG/'gate_backward.py').read_text()
old='bias_backward._extension().backward(q,k,v,b,m,delta,dy,4)'
assert gp.count(old)==1;gp=gp.replace(old,'bias_backward.native_backward(q,k,v,b,m,delta,dy)')
builder=(PKG/'build_bias.py').read_text()
assert 'row_group=4' in builder
builder=builder.replace('row_group=4',"row_group=previous['row_group']")
builder=builder.replace('extra_cuda_cflags=["-O3",', 'extra_cuda_cflags=["-O3", "--objdir-as-tempdir",')
dqbuilder=(PKG/'build_dq.py').read_text()
assert 'extra_cuda_cflags=["-O3",' in dqbuilder
dqbuilder=dqbuilder.replace('extra_cuda_cflags=["-O3",', 'extra_cuda_cflags=["-O3", "--objdir-as-tempdir",')
manifests={name:json.loads((PKG/name).read_text()) for name in ('bias_manifest.json','dq_manifest.json','gate_manifest.json')}

for kind,(p,filename,lib,name) in sources.items():
    copy(p/'build'/lib,PKG/(name+'.so'))
    copy(p/filename,PKG/('bias_fusion.cu' if kind=='bias' else 'dq.cu'))
write(PKG/'bias_backward.py',bp);write(PKG/'gate_backward.py',gp);write(PKG/'build_bias.py',builder);write(PKG/'build_dq.py',dqbuilder)
for kind,artifact,job in (('bias',a.bias,a.bias_job),('dq',a.dq,a.dq_job)):
    data=manifests[kind+'_manifest.json'];old_binary=data['binary'];name=sources[kind][3]
    data.update(module_name=name,binary=name+'.so',artifact=artifact,qualification_job=job,core90_combined_job=a.combined_job)
    data['files']={f:sha(PKG/f) for f in sorted((set(data['files'])-{old_binary})|{name+'.so'})}
    if kind=='bias':data.update(row_group=a.bias_rows,legacy_abi_row_group=4,dq_artifact=a.dq)
    else:data['fp64_job']=a.fp64_job
gate=manifests['gate_manifest.json']
gate['files']={f:sha(PKG/f) for f in gate['files']}
gate['bias_row_group']=a.bias_rows
for filename,data in manifests.items():write_json(PKG/filename,data)
write_json(ROOT/'installation.json',manifests)
print('INSTALLED',a.bias,a.dq,flush=True)
