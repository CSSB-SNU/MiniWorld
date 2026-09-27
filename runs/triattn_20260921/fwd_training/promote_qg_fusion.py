"""Install a qualified projection-attention boundary, with measured rollback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from miniworld_engine.kernels.triangle_attention import cuda

ap=argparse.ArgumentParser();ap.add_argument('--artifact',required=True)
for n in ('qualify','verify','bench','sanitize','profile'):ap.add_argument('--'+n,type=int,required=True)
a=ap.parse_args();r=Path(__file__).resolve().parent;pkg=Path(cuda.__file__).parent
stage=r/('stage-'+a.artifact);job=os.environ['SLURM_JOB_ID']
report=dict(job=job,artifact=a.artifact,state='checking',evidence=vars(a))
result=r/('qg-promotion-'+job+'.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save():result.write_text(json.dumps(report,indent=2)+'\n')
def run(script,*args):subprocess.run([sys.executable,'-u','-B',str(r/script),*map(str,args)],check=True)
def ci(row):
    x=np.asarray(row['paired_ratios']);assert len(x)>=64 and np.isfinite(x).all()
    rng=np.random.default_rng(17756)
    return np.quantile(np.median(x[rng.integers(0,len(x),(10000,len(x)))],axis=1),[.025,.975]).tolist()
def bench(data,installed):
    assert data['complete'] and data['candidate_build']==build
    assert data['installed_dispatch']==installed and len(data['records'])==6
    for row in data['records']:
        assert max(row['errors'])==0,row
        if row['kind']=='forward':assert ci(row)[0]>1,row
        if row['kind']=='forward_backward':assert row['speedup']>1,row

build=read(r/a.artifact/'build-ready.json')
manifest=read(stage/'qg_fwd_manifest.json')
for n,h in build['sha256'].items():assert sha(r/a.artifact/n)==h
for n,h in manifest['files'].items():assert sha(stage/n)==h
assert manifest['artifact']==a.artifact and manifest['lengths']==[384,768,1024]
qual=read(r/('qg-module-%d-L384.json'%a.qualify))
assert qual['complete'] and qual['staged'] and qual['candidate_build']==build
assert qual['candidate_python_sha256']==manifest['files']['qg_projection_attention.py']
assert len(qual['records'])==26
verify=read(r/('qg-verify-%d.json'%a.verify))
assert verify['complete'] and verify['staged'] and verify['artifact']==a.artifact
assert any(x['kind']=='guards' and x['passed'] for x in verify['records'])
for L in manifest['lengths']:bench(read(r/('qg-module-%d-L%d.json'%(a.bench,L))),False)
sanlog=(r/('qg-sanitize-%d.log'%a.sanitize)).read_text()
assert sanlog.count('ERROR SUMMARY: 0 errors')==4
assert sanlog.count('0 hazards displayed (0 errors, 0 warnings)')==2
for L in (128,768):
    for tool in ('memcheck','racecheck','synccheck'):
        d=read(r/('qg-%s-%d-L%d.json'%(tool,a.sanitize,L)))
        assert d['complete'] and d['artifact']==a.artifact and d['build']==build
assert read(r/('qg-profile-%d-%s.json'%(a.profile,a.artifact)))['complete']
# The entire starting package must still match the frozen baseline.
snapshot=r/'checkpoint17774';frozen=read(snapshot/'snapshot.json')
for n,h in frozen['sha256'].items():
    if n.startswith('cuda/'):assert sha(pkg/n[len('cuda/'):])==h,(n,'baseline changed')
    elif n=='triton/main.py':assert sha(pkg.parent/'triton/main.py')==h
assert not (pkg/'qg_fwd_manifest.json').exists(),'another fusion installation already exists'
names=[*manifest['files'],'qg_fwd_manifest.json','ln_backward.py','ln_manifest.json','q_fwd_manifest.json']
backup=r/('before-qg-'+job);backup.mkdir()
existing=[n for n in names if (pkg/n).exists()]
for n in existing:shutil.copy2(pkg/n,backup/n)
report.update(state='installing',backup=str(backup));save()
try:
    for n in manifest['files']:shutil.copy2(stage/n,pkg/n)
    ln=(pkg/'ln_backward.py').read_text()
    needle='def forward(model,pair,mask=None):\n'
    assert ln.count(needle)==1
    ln=ln.replace(needle,needle+'''    from miniworld_engine.kernels.triangle_attention.cuda import qg_projection_attention
    if qg_projection_attention.can_use(model,pair,mask):
        return qg_projection_attention.forward(model,pair,mask)
''')
    (pkg/'ln_backward.py').write_text(ln)
    lnmanifest=read(pkg/'ln_manifest.json')
    lnmanifest['files']['ln_backward.py']=sha(pkg/'ln_backward.py')
    lnmanifest['qg_forward_promotion']=job
    (pkg/'ln_manifest.json').write_text(json.dumps(lnmanifest,indent=2)+'\n')
    manifest.update(promotion=job,evidence=vars(a),dispatcher_sha256=sha(pkg/'ln_backward.py'))
    (pkg/'qg_fwd_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    qmanifest=read(pkg/'q_fwd_manifest.json')
    qmanifest['dispatcher_sha256']=sha(pkg/'ln_backward.py')
    qmanifest['dispatcher_qg_promotion']=job
    (pkg/'q_fwd_manifest.json').write_text(json.dumps(qmanifest,indent=2)+'\n')
    report.update(state='verifying',manifest=manifest);save()
    run('verify_qg.py','--artifact',a.artifact,'--installed','--output',r/('qg-installed-verify-'+job+'.json'))
    assert read(r/('qg-installed-verify-'+job+'.json'))['complete']
    for L in manifest['lengths']:
        p=r/('qg-installed-bench-%s-L%d.json'%(job,L))
        run('qg_module_check.py','--artifact',a.artifact,'--installed','--length',L,'--mode','bench',
            '--rounds',64,'--replays',40,'--output',p)
        bench(read(p),True)
    # Audit existing code and binaries; only dispatcher and two manifests change.
    for n,h in frozen['sha256'].items():
        if n.startswith('cuda/') and n not in ('cuda/ln_backward.py','cuda/ln_manifest.json','cuda/q_fwd_manifest.json'):
            assert sha(pkg/n[len('cuda/'):])==h,n
    checkpoint=r/('checkpoint'+job);(checkpoint/'cuda').mkdir(parents=True);(checkpoint/'triton').mkdir()
    files=set()
    for p in pkg.glob('*manifest.json'):
        d=read(p);files.add(p.name);files.update(d.get('files',d.get('sha256',{})))
    for n in sorted(files):shutil.copy2(pkg/n,checkpoint/'cuda'/n)
    shutil.copy2(pkg.parent/'triton/main.py',checkpoint/'triton/main.py')
    hashes={str(p.relative_to(checkpoint)):sha(p) for p in checkpoint.rglob('*') if p.is_file()}
    (checkpoint/'snapshot.json').write_text(json.dumps(dict(promotion=job,sha256=hashes),indent=2)+'\n')
    report.update(state='complete',snapshot=str(checkpoint),snapshot_files=len(hashes));save()
    print('QG_FUSION_PROMOTION_COMPLETE',job,flush=True)
except BaseException as e:
    for n in names:
        if n in existing:shutil.copy2(backup/n,pkg/n)
        else:(pkg/n).unlink(missing_ok=True)
    report.update(state='rolled_back',error=repr(e));save()
    raise
