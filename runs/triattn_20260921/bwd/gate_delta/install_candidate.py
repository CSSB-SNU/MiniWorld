"""Install a qualified gate/delta fusion and connect its shared autograd boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import torch

ap=argparse.ArgumentParser()
ap.add_argument('--artifact',required=True)
ap.add_argument('--qualification-job',type=int,required=True)
args=ap.parse_args()
root=Path(__file__).resolve().parent
src=root/args.artifact
for name,digest in json.loads((src/'build-ready.json').read_text()).items():
    assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest,name
engine=root.parents[2]/'trimul_sm90_parity_20260917/engine/src/miniworld_engine'
pkg=engine/'kernels/triangle_attention/cuda'
name=(src/'module-name.txt').read_text().strip()
binary=name+'.so'
shutil.copy2(src/'build/triattn_gate_delta.so',pkg/(binary+'.new'))
os.replace(pkg/(binary+'.new'),pkg/binary)
shutil.copy2(src/'fused.cu',pkg/'gate_delta.cu')
shutil.copy2(root/'package_api.py',pkg/'gate_backward.py')
shutil.copy2(root/'package_build.py',pkg/'build_gate.py')
api=pkg/'ln_backward.py'
s=api.read_text()
if '_fuse_gate_backward' not in s:
    old='''    out=model._kernel_triangle_attention(q,k,v,b,model._backend)
    out=out.permute(0,2,3,1,4).reshape(B,L,L,C)
    out=model._gate_out(g,out)'''
    new='''    from miniworld_engine.kernels.triangle_attention.cuda import gate_backward
    from miniworld_engine.kernels.bias_only_attention import dispatch as gate_dispatch
    fused_gate=(getattr(model, "_fuse_gate_backward", True)
        and getattr(model, "_fuse_bias_backward", True) and getattr(model, "_fuse_dq_backward", True)
        and gate_dispatch.gate_use_fused(C,model.to_out.weight.shape[0],B*L*L,pair.device,pair.dtype)
        and gate_backward.can_use(q,model.to_out.weight))
    if fused_gate:
        out=gate_backward.attention_gate(q,k,v,b,g,model.to_out.weight)
    else:
        out=model._kernel_triangle_attention(q,k,v,b,model._backend)
        out=out.permute(0,2,3,1,4).reshape(B,L,L,C)
        out=model._gate_out(g,out)'''
    assert old in s
    s=s.replace(old,new,1)
    api.write_text(s)
    (root.parent/'ln_residual/package_api.py').write_text(s)
module=engine/'modules/triangle_attention/module.py'
s=module.read_text()
if '_fuse_gate_backward' not in s:
    s=s.replace('        self._fuse_front_backward = True',
                '        self._fuse_front_backward = True\n        self._fuse_gate_backward = True',1)
    module.write_text(s)
s=module.read_text().replace('        self._fuse_gate_backward = False  # Re-enable after fresh-autograd-context qualification.','        self._fuse_gate_backward = True')
module.write_text(s)
data=dict(module_name=name,binary=binary,torch=str(torch.__version__),python_abi=sys.implementation.cache_tag,
          lengths=[384,768,1024],artifact=args.artifact,qualification_job=args.qualification_job,
          files={f:hashlib.sha256((pkg/f).read_bytes()).hexdigest() for f in
                 ('gate_delta.cu','gate_backward.py','build_gate.py',binary,'fa3_utils.h')})
(pkg/'gate_manifest.json.new').write_text(json.dumps(data,indent=2)+'\n')
os.replace(pkg/'gate_manifest.json.new',pkg/'gate_manifest.json')
ln=json.loads((pkg/'ln_manifest.json').read_text())
ln['files']['ln_backward.py']=hashlib.sha256(api.read_bytes()).hexdigest()
ln['gate_artifact']=args.artifact
(pkg/'ln_manifest.json.new').write_text(json.dumps(ln,indent=2)+'\n')
os.replace(pkg/'ln_manifest.json.new',pkg/'ln_manifest.json')
(root/'installation.json').write_text(json.dumps(dict(package_manifest=str(pkg/'gate_manifest.json'),
    module=str(module),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),**data),indent=2)+'\n')
print('INSTALLED gate/delta',args.artifact)
