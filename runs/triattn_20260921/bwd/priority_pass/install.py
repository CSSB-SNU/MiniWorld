"""Install the qualified bias-load improvement and shared-input weight gradients."""
import hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PKG=ROOT.parents[2]/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(path,data):
    temp=path.with_name(path.name+'.new');temp.write_text(json.dumps(data,indent=2)+'\n');os.replace(temp,path)
def copy(src,dst):
    temp=dst.with_name(dst.name+'.new');shutil.copy2(src,temp);os.replace(temp,dst)
# Detect any intervening package edit before touching it.
for f,d in json.loads((ROOT/'baseline/snapshot.json').read_text()).items():assert sha(PKG/f)==d,f
for folder,job,count in [('below80',16653,10),('wgrad',16662,12)]:
    log=(ROOT.parent/folder/f'qualify-{job}.log').read_text()
    assert f'API_PASS {count}' in log
    assert log.count('ERROR SUMMARY: 0 errors')>=2
    assert '0 hazards displayed (0 errors, 0 warnings)' in log
for L in (384,768,1024):
    d=json.loads((ROOT/f'combined-16661-L{L}.json').read_text())
    assert d['complete'] and len(d['measurements'])==4
    assert all(v['speedup']>1 and max(v['gradient_relative_l2'])<.015 for v in d['measurements'].values())
for folder,artifact in [('bias_fusion','ldmatrix_bias'),('wgrad','shared_z_tensor')]:
    path=ROOT.parent/folder/artifact
    for f,d in json.loads((path/'build-ready.json').read_text()).items():assert sha(path/f)==d,f
oldln=json.loads((PKG/'ln_manifest.json').read_text())
oldbias=json.loads((PKG/'bias_manifest.json').read_text())
lnsource=(PKG/'ln_backward.py').read_text()
old='        dw=[g.T@zz if needed else None for g,needed in zip(dy,ctx.needs_input_grad[5:])]'
new='''        from miniworld_engine.kernels.triangle_attention.cuda import wgrad_backward
        needed=ctx.needs_input_grad[5:]
        if all(needed[:4]) and wgrad_backward.can_use(dy[:4],zz):
            dw=[*wgrad_backward.backward(dy[:4],zz),dy[4].T@zz if needed[4] else None]
        else:
            dw=[g.T@zz if n else None for g,n in zip(dy,needed)]'''
assert lnsource.count(old)==1
assert not (PKG/'wgrad_manifest.json').exists()
# Binaries use new names, so an already-loaded old module remains valid.
biaspath=ROOT.parent/'bias_fusion/ldmatrix_bias'
biasname=(biaspath/'module-name.txt').read_text().strip();biasbin=biasname+'.so'
copy(biaspath/'build/triattn_bias_fusion.so',PKG/biasbin)
copy(biaspath/'grouped.cu',PKG/'bias_fusion.cu')
wpath=ROOT.parent/'wgrad/shared_z_tensor'
wname=(wpath/'module-name.txt').read_text().strip();wbin=wname+'.so'
copy(wpath/'build/triattn_wgrad.so',PKG/wbin)
copy(wpath/'grouped.cu',PKG/'wgrad.cu')
for f in ('wgrad_backward.py','build_wgrad.py'):copy(ROOT.parent/'wgrad/package'/f,PKG/f)
wm=dict(module_name=wname,binary=wbin,torch=oldln['torch'],python_abi=oldln['python_abi'],
        arch='sm_90a',lengths=[384,768,1024],splits={'384':2304,'768':8960,'1024':15936},
        artifact='shared_z_tensor',partial_dtype='fp32',qualification_job=16662,combined_job=16661,
        files={f:sha(PKG/f) for f in ['wgrad.cu','wgrad_backward.py','build_wgrad.py','fa3_utils.h',wbin]})
# Manifest ready before Python dispatch starts using the new optional artifact.
write_json(PKG/'wgrad_manifest.json',wm)
(PKG/'ln_backward.py.new').write_text(lnsource.replace(old,new));os.replace(PKG/'ln_backward.py.new',PKG/'ln_backward.py')
bm=dict(oldbias);bm.update(module_name=biasname,binary=biasbin,artifact='ldmatrix_bias',qualification_job=16653,priority_combined_job=16661)
bm['files']={f:sha(PKG/f) for f in sorted((set(oldbias['files'])-{oldbias['binary']})|{biasbin})}
lm=dict(oldln);lm.update(wgrad_artifact='shared_z_tensor',wgrad_qualification_job=16662,priority_combined_job=16661)
lm['files']={f:sha(PKG/f) for f in oldln['files']}
write_json(PKG/'bias_manifest.json',bm);write_json(PKG/'ln_manifest.json',lm)
write_json(ROOT/'installation.json',{'bias_manifest.json':bm,'ln_manifest.json':lm,'wgrad_manifest.json':wm})
print('INSTALLED',biasname,wname,flush=True)
