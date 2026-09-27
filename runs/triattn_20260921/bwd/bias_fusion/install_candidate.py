"""Install a separately qualified artifact. Run only after reviewing its evidence."""
import argparse,hashlib,json,shutil,sys,os
from pathlib import Path
import torch
root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--artifact',required=True);ap.add_argument('--lengths',nargs='+',type=int,required=True);ap.add_argument('--partial',choices=['fp32','bf16'],required=True);ap.add_argument('--qualification-job',type=int,required=True);a=ap.parse_args()
src=root/a.artifact
for name,digest in json.loads((src/'build-ready.json').read_text()).items():
    assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest,name
engine=root.parents[2]/'trimul_sm90_parity_20260917/engine/src/miniworld_engine'
pkg=engine/'kernels/triangle_attention/cuda'
assert (pkg/'fa3_utils.h').read_bytes()==(root.parents[1]/'oc/opt_core/kernels/triattn_core_broadcast/csrc/fa3_utils.h').read_bytes()
shutil.copy2(src/'grouped.cu',pkg/'bias_fusion.cu')
name_file=src/'module-name.txt'
module_name=name_file.read_text().strip() if name_file.exists() else 'triattn_bias_fusion'
binary=module_name+'.so'
shutil.copy2(src/'build/triattn_bias_fusion.so',pkg/(binary+'.new'))
os.replace(pkg/(binary+'.new'),pkg/binary)
shutil.copy2(root/'package_build.py',pkg/'build_bias.py')
api=(root/'package_api.py').read_text().replace('SUPPORTED_LENGTHS=(768,1024)','SUPPORTED_LENGTHS='+repr(tuple(a.lengths)))
(pkg/'bias_backward.py').write_text(api)
module=engine/'modules/triangle_attention/module.py';s=module.read_text()
if '_fuse_bias_backward' not in s:
    s=s.replace('# Qualified H100 training shapes fuse only the projection input gradient.\n        # The private switch is also used to measure the unchanged baseline.', '# Independent switches for qualified H100 projection and grouped-bias backward.\n        # Each switch also selects its corresponding benchmark baseline.')
    s=s.replace('        self._fuse_projection_backward = True','        self._fuse_projection_backward = True\n        self._fuse_bias_backward = True',1)
    old='''        if backend == KernelBackend.TRITON:
            return kernels.triton_triangle_attention_pair_bias('''
    new='''        if backend == KernelBackend.TRITON:
            if (getattr(self, "_fuse_bias_backward", True) and torch.is_grad_enabled()
                    and not self.use_qk_norm):
                from miniworld_engine.kernels.triangle_attention.cuda.bias_backward import can_use, attention
                if can_use(query, key, value, bias):
                    return attention(query, key, value, bias)
            return kernels.triton_triangle_attention_pair_bias('''
    assert old in s;s=s.replace(old,new,1);module.write_text(s)
data=dict(module_name=module_name,binary=binary,torch=str(torch.__version__),python_abi=sys.implementation.cache_tag,
    lengths=a.lengths,partial_dtype=a.partial,row_group=4,artifact=a.artifact,qualification_job=a.qualification_job,
    files={f:hashlib.sha256((pkg/f).read_bytes()).hexdigest() for f in ('bias_fusion.cu','bias_backward.py','build_bias.py',binary,'fa3_utils.h')})
(pkg/'bias_manifest.json.new').write_text(json.dumps(data,indent=2)+'\n')
os.replace(pkg/'bias_manifest.json.new',pkg/'bias_manifest.json')
(root/'installation.json').write_text(json.dumps(dict(package_manifest=str(pkg/'bias_manifest.json'),module=str(module),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),**data),indent=2)+'\n')
print('INSTALLED',a.artifact,a.lengths)
