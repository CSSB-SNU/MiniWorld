"""Install the exact qualified manual_shared artifact and guarded module dispatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import torch

root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--artifact', default='manual_shared')
ap.add_argument('--qualification-job', type=int, default=16199)
args = ap.parse_args()
src = root / args.artifact
for name, digest in json.loads((src / 'build-ready.json').read_text()).items():
    assert hashlib.sha256((src / name).read_bytes()).hexdigest() == digest, name
engine = root.parents[2] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine'
pkg = engine / 'kernels/triangle_attention/cuda'
assert (pkg / 'fa3_utils.h').read_bytes() == (root.parents[1] / 'oc/opt_core/kernels/triattn_core_broadcast/csrc/fa3_utils.h').read_bytes()
name = (src / 'module-name.txt').read_text().strip()
binary = name + '.so'
shutil.copy2(src / 'build/triattn_ln_residual.so', pkg / (binary + '.new'))
os.replace(pkg / (binary + '.new'), pkg / binary)
for source, target in [('fused.cu', 'ln_residual.cu')]:
    shutil.copy2(src / source, pkg / target)
for source, target in [('package_api.py', 'ln_backward.py'), ('package_build.py', 'build_ln.py')]:
    shutil.copy2(root / source, pkg / target)
module = engine / 'modules/triangle_attention/module.py'
s = module.read_text()
if '_fuse_front_backward' not in s:
    s = s.replace('        self._fuse_bias_backward = True',
                  '        self._fuse_bias_backward = True\n        self._fuse_front_backward = True', 1)
    old = '        out = self._attention(pair, mask)'
    new = '''        if (
            getattr(self, "_fuse_front_backward", True)
            and getattr(self, "_fuse_projection_backward", True)
            and torch.is_grad_enabled() and self._backend == KernelBackend.TRITON
            and self.use_self_attention and not self.use_qk_norm and self.n_head == 4
            and _bo_dispatch.use_kernels(pair.shape[1])
        ):
            from miniworld_engine.kernels.triangle_attention.cuda import ln_backward
            weights = (self.to_query.weight, self.to_key.weight, self.to_value.weight,
                       self.to_gate.weight, self.to_bias.weight)
            if ln_backward.can_use(pair, weights, self.ln_pair.weight, self.ln_pair.bias):
                with _nvtx_range(self.nvtx_name, self.nvtx_enabled):
                    return ln_backward.forward(self, pair, mask)
        out = self._attention(pair, mask)'''
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    module.write_text(s)
data = dict(module_name=name, binary=binary, torch=str(torch.__version__),
            python_abi=sys.implementation.cache_tag, lengths=[384, 768, 1024],
            artifact=args.artifact, qualification_job=args.qualification_job,
            files={f: hashlib.sha256((pkg / f).read_bytes()).hexdigest() for f in
                   ('ln_residual.cu', 'ln_backward.py', 'build_ln.py', binary, 'fa3_utils.h')})
(pkg / 'ln_manifest.json.new').write_text(json.dumps(data, indent=2) + '\n')
os.replace(pkg / 'ln_manifest.json.new', pkg / 'ln_manifest.json')
(root / 'installation.json').write_text(json.dumps(dict(
    package_manifest=str(pkg / 'ln_manifest.json'), module=str(module),
    module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(), **data), indent=2) + '\n')
print('INSTALLED LN/residual', args.artifact, '[384, 768, 1024]')
