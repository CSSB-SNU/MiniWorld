"""Install a separately qualified dQ binary into grouped backward dispatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--artifact', required=True)
ap.add_argument('--qualification-job', type=int, required=True)
args = ap.parse_args()
root = Path(__file__).resolve().parent
src = root / args.artifact
for name, digest in json.loads((src / 'build-ready.json').read_text()).items():
    assert hashlib.sha256((src / name).read_bytes()).hexdigest() == digest, name
engine = root.parents[2] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine'
pkg = engine / 'kernels/triangle_attention/cuda'
name = (src / 'module-name.txt').read_text().strip()
binary = name + '.so'
shutil.copy2(src / 'build/triattn_dq.so', pkg / (binary + '.new'))
os.replace(pkg / (binary + '.new'), pkg / binary)
shutil.copy2(src / 'fused.cu', pkg / 'dq.cu')
shutil.copy2(root / 'package_api.py', pkg / 'dq_backward.py')
shutil.copy2(root / 'package_build.py', pkg / 'build_dq.py')
api = pkg / 'bias_backward.py'
s = api.read_text()
if 'native_dq:bool' not in s:
    s = s.replace('Forward, delta and dQ retain the established implementation.',
                  'Forward and delta retain the established implementation. dQ uses qualified CUDA.')
    s = s.replace('def _fake(q,k,v,b,m,out,dy):', 'def _fake(q,k,v,b,m,out,dy,native_dq):')
    s = s.replace('dy:torch.Tensor)->tuple', 'dy:torch.Tensor,native_dq:bool)->tuple')
    start = s.index('    dq=torch.empty(')
    end = s.index('    return dq,dk,dv,db', start)
    stock = s[start:end]
    s = s[:start] + '''    from miniworld_engine.kernels.triangle_attention.cuda import dq_backward
    if native_dq and dq_backward.available():
        dq=dq_backward.backward(q,k,v,b,m,delta,dy)
    else:
''' + ''.join('    ' + line + '\n' for line in stock.splitlines()) + s[end:]
    s = s.replace('def forward(ctx,q,k,v,b):', 'def forward(ctx,q,k,v,b,native_dq):')
    s = s.replace('        ctx.save_for_backward(q,k,v,b,m,out)',
                  '        ctx.save_for_backward(q,k,v,b,m,out)\n        ctx.native_dq=native_dq')
    s = s.replace('def backward(ctx,dy):return _backward(*ctx.saved_tensors,dy)',
                  'def backward(ctx,dy):return (*_backward(*ctx.saved_tensors,dy,ctx.native_dq),None)')
    s = s.replace('def attention(q,k,v,b):return _Attention.apply(q,k,v,b)',
                  'def attention(q,k,v,b,native_dq=True):return _Attention.apply(q,k,v,b,native_dq)')
    api.write_text(s)
    # Preserve the install/rebuild recipe for the selected grouped artifact.
    (root.parent / 'bias_fusion/package_api.py').write_text(
        s.replace('SUPPORTED_LENGTHS=(384, 768, 1024)', 'SUPPORTED_LENGTHS=(768,1024)'))
module = engine / 'modules/triangle_attention/module.py'
s = module.read_text()
if '_fuse_dq_backward' not in s:
    s = s.replace('        self._fuse_bias_backward = True',
                  '        self._fuse_bias_backward = True\n        self._fuse_dq_backward = True', 1)
    s = s.replace('return attention(query, key, value, bias)',
                  'return attention(query, key, value, bias, native_dq=getattr(self, "_fuse_dq_backward", True))', 1)
    module.write_text(s)
data = dict(module_name=name, binary=binary, torch=str(torch.__version__),
            python_abi=sys.implementation.cache_tag, lengths=[384, 768, 1024],
            artifact=args.artifact, qualification_job=args.qualification_job,
            files={f: hashlib.sha256((pkg / f).read_bytes()).hexdigest() for f in
                   ('dq.cu', 'dq_backward.py', 'build_dq.py', binary, 'fa3_utils.h')})
(pkg / 'dq_manifest.json.new').write_text(json.dumps(data, indent=2) + '\n')
os.replace(pkg / 'dq_manifest.json.new', pkg / 'dq_manifest.json')
bias = json.loads((pkg / 'bias_manifest.json').read_text())
bias['files']['bias_backward.py'] = hashlib.sha256(api.read_bytes()).hexdigest()
bias['dq_artifact'] = args.artifact
(pkg / 'bias_manifest.json.new').write_text(json.dumps(bias, indent=2) + '\n')
os.replace(pkg / 'bias_manifest.json.new', pkg / 'bias_manifest.json')
(root / 'installation.json').write_text(json.dumps(dict(
    package_manifest=str(pkg / 'dq_manifest.json'), module=str(module),
    module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(), **data), indent=2) + '\n')
print('INSTALLED dQ', args.artifact)
