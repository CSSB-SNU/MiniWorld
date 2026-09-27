"""Install only the qualified winning forward; roll back if installed checks fail."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from miniworld_engine.kernels.triangle_attention import cuda
from miniworld_engine.kernels.triangle_attention.triton import main as core

ROOT=Path(__file__).resolve().parent
pkg=Path(cuda.__file__).parent
dispatcher=Path(core.__file__)
artifact=ROOT/'cooperative_q2'
job=os.environ['SLURM_JOB_ID']
out=ROOT/('promotion-'+job+'.json')
report=dict(job=job,artifact='cooperative_q2',state='checking',qualification_job=17290,
            benchmark_job=17289,profile_job=17294,full_sanitizer_job=17300)


def save():out.write_text(json.dumps(report,indent=2)+'\n')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


build=json.loads((artifact/'build-ready.json').read_text())
for f,h in build['sha256'].items():assert digest(artifact/f)==h
qual=json.loads((ROOT/'qualify-17290.json').read_text())
assert qual['complete'] and qual['candidate_build']==build
qlog=(ROOT/'qualify-17290.log').read_text()
assert qlog.count('ERROR SUMMARY: 0 errors')==4 and qlog.count('0 hazards displayed (0 errors, 0 warnings)')==2
flog=(ROOT/'full-sanitize-17300.log').read_text()
assert flog.count('ERROR SUMMARY: 0 errors')==2 and '0 hazards displayed (0 errors, 0 warnings)' in flog
for tool in ('memcheck','racecheck','synccheck'):
    assert json.loads((ROOT/(tool+'-17300-L768.json')).read_text())['complete']
for L in (384,768,1024):
    data=json.loads((ROOT/('bench-17289-L%d.json'%L)).read_text())
    assert data['complete'] and data['candidate_build']==build
    assert len(data['records'])==6
    assert all(r['speedup']>1.02 for r in data['records'] if r['kind'] in ('forward','forward_backward'))
profile=json.loads((ROOT/'candidate-profile-17294.json').read_text())
assert profile['complete'] and profile['build']==build
# Detect any unrelated change in the qualified installed backward package.
for name,info in qual['manifests'].items():
    assert json.loads((pkg/name).read_text())==info
    for f,h in info.get('files',info.get('sha256',{})).items():assert digest(pkg/f)==h

backup=ROOT/('before-'+job);backup.mkdir()
original=dispatcher.read_text()
(backup/'main.py').write_text(original)
names=['training_forward.py','training_forward.cu',build['module']+'.so','fwd_manifest.json']
existing=[]
for name in names:
    if (pkg/name).exists():shutil.copy2(pkg/name,backup/name);existing.append(name)
report.update(state='installing',dispatcher_before_sha256=digest(dispatcher),backup=str(backup))
save()
try:
    text=original
    anchor='''    B, H, L, _, D = q.shape
    sm_scale = D**-0.5
'''
    assert text.count(anchor)==1
    start=text.index('def _tri_attn_fwd(q:')
    pos=text.index(anchor,start)
    text=text[:pos]+'''    from miniworld_engine.kernels.triangle_attention.cuda import training_forward
    if training_forward.can_use(q, k, v, bias):
        return training_forward.forward(q, k, v, bias)
'''+text[pos:]
    shutil.copy2(ROOT/'package/training_forward.py',pkg/'training_forward.py')
    shutil.copy2(artifact/'fused.cu',pkg/'training_forward.cu')
    shutil.copy2(artifact/build['binary'],pkg/(build['module']+'.so'))
    dispatcher.write_text(text)
    manifest=dict(module_name=build['module'],binary=build['module']+'.so',artifact='cooperative_q2',
        torch=str(torch.__version__),python_abi=sys.implementation.cache_tag,arch='sm_90a',
        lengths=[384,768,1024],query_tile=64,key_tile=64,threads=128,kv_stages=3,
        normalizer='fp32_unrounded_probability_sum',lse='fp32_base2',
        qualification_job=17290,benchmark_job=17289,profile_job=17294,full_sanitizer_job=17300,
        dispatcher=dict(path='../triton/main.py',sha256=digest(dispatcher)),
        files={f:digest(pkg/f) for f in ('training_forward.py','training_forward.cu',build['module']+'.so','fa3_utils.h')})
    (pkg/'fwd_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report['manifest']=manifest;report['state']='verifying';save()
    subprocess.run([sys.executable,'-u','-B',str(ROOT/'verify_installed.py'),
                    '--output',str(ROOT/('installed-'+job+'.json'))],check=True)
    for L in (384,768,1024):
        path=ROOT/('installed-bench-'+job+'-L%d.json'%L)
        subprocess.run([sys.executable,'-u','-B',str(ROOT/'module_check.py'),
                        '--artifact','cooperative_q2','--length',str(L),'--mode','bench','--installed',
                        '--output',str(path)],check=True)
        data=json.loads(path.read_text())
        assert data['complete'] and data['installed_dispatch']
        assert all(r['speedup']>1.02 for r in data['records'] if r['kind'] in ('forward','forward_backward'))
    for f,h in manifest['files'].items():assert digest(pkg/f)==h
    assert digest(dispatcher)==manifest['dispatcher']['sha256']
    report['state']='complete';save()
    print('PROMOTION_COMPLETE_FWD',job,flush=True)
except BaseException as error:
    dispatcher.write_text(original)
    for name in names:
        if name in existing:shutil.copy2(backup/name,pkg/name)
        else:(pkg/name).unlink(missing_ok=True)
    report.update(state='rolled_back',error=repr(error));save()
    raise
