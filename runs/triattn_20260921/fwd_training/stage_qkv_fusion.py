"""Prepare a reviewable package addition without changing installed dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

ap=argparse.ArgumentParser();ap.add_argument('--artifact',required=True)
ap.add_argument('--lengths',type=int,nargs='+',default=[1024]);a=ap.parse_args()
r=Path(__file__).resolve().parent
pkg=r.parent.parent/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda'
stage=r/('stage-'+a.artifact);stage.mkdir(exist_ok=True)
artifact=r/a.artifact;build=json.loads((artifact/'build-ready.json').read_text())
for name,sha in build['sha256'].items():assert hashlib.sha256((artifact/name).read_bytes()).hexdigest()==sha
s=(r/'stream_module.py').read_text().replace('from native import extension\n','')
s=s.replace('from miniworld_engine.kernels._compile import opaque','from miniworld_engine.kernels._compile import opaque,device_constant')
s=s.replace('_extension=None', '''import functools,hashlib,importlib.util,json,os,sys
from pathlib import Path
_ROOT=Path(__file__).resolve().parent
_EXT=None
SUPPORTED_LENGTHS='''+repr(tuple(a.lengths))+'''

@functools.lru_cache(maxsize=1)
def _artifact_ready():
    try:
        data=json.loads((_ROOT/'qkv_fwd_manifest.json').read_text())
        return (data['python_abi']==sys.implementation.cache_tag and data['torch']==str(torch.__version__)
            and all(hashlib.sha256((_ROOT/n).read_bytes()).hexdigest()==h for n,h in data['files'].items()))
    except (OSError,ValueError,KeyError):return False

@device_constant
def _available(device):
    return (_artifact_ready() and torch.cuda.get_device_capability(device)==(9,0)
            and 'H100' in torch.cuda.get_device_name(device))

def extension():
    global _EXT
    if _EXT is None:
        data=json.loads((_ROOT/'qkv_fwd_manifest.json').read_text())
        spec=importlib.util.spec_from_file_location(data['module_name'],_ROOT/data['binary'])
        _EXT=importlib.util.module_from_spec(spec);spec.loader.exec_module(_EXT)
    return _EXT

def can_use(model,pair,mask=None):
    if (os.environ.get('MINIWORLD_TRIATTN_QKV_FWD','1')=='0'
            or os.environ.get('MINIWORLD_TRIATTN_QG_FWD','1')=='0'
            or os.environ.get('MINIWORLD_TRIATTN_Q_FWD','1')=='0'
            or os.environ.get('MINIWORLD_TRIATTN_TRAINING_FWD','1')=='0'):
        return False
    if not (torch.is_grad_enabled() and pair.ndim==4 and pair.shape[1] in SUPPORTED_LENGTHS
            and model.use_self_attention and not model.use_qk_norm and model.n_head==4
            and all(getattr(model,n,True) for n in ('_fuse_front_backward','_fuse_projection_backward',
                '_fuse_gate_backward','_fuse_bias_backward','_fuse_dq_backward'))):
        return False
    weights=(model.to_query.weight,model.to_key.weight,model.to_value.weight,
             model.to_gate.weight,model.to_bias.weight)
    if not ln_backward.can_use(pair,weights,model.ln_pair.weight,model.ln_pair.bias):return False
    if mask is not None and not (mask.dtype==torch.bool and mask.device==pair.device
            and mask.shape==(1,pair.shape[1])):return False
    from miniworld_engine.kernels.bias_only_attention import dispatch as gate_dispatch
    return (_available(pair.device) and gate_backward.can_use(pair,model.to_out.weight)
        and gate_dispatch.gate_use_fused(128,128,pair.shape[1]**2,pair.device,pair.dtype))
''')
s=s.replace("name='triangle_qkv_stream_experiment'","name='triangle_qkv_projection_attention_cuda'")
s=s.replace('_extension.forward(', 'extension().forward(')
prefix=s[:s.index('def make_forward(artifact):')]
body=textwrap.dedent(s[s.index('    class FrontAttention'):])
assert body.endswith('return forward\n')
body=body[:-len('return forward\n')]
s=prefix+body
s=s.replace('"""Experimental combined front/attention autograd; existing native backward kernels."""',
'''"""H100 Q/K/V/gate projections plus attention; preserve native training backward.

Set MINIWORLD_TRIATTN_QKV_FWD=0 to restore Q+gate fusion.
Only supported BF16 shapes with every required native fusion enabled qualify.
"""''')
(stage/'qkv_projection_attention.py').write_text(s)
shutil.copy2(artifact/'fused.cu',stage/'qkv_projection_attention.cu')
binary=build['module']+'.so';shutil.copy2(artifact/build['binary'],stage/binary)
old=json.loads((pkg/'fwd_manifest.json').read_text())
names=['qkv_projection_attention.py','qkv_projection_attention.cu',binary]
manifest=dict(module_name=build['module'],binary=binary,artifact=a.artifact,lengths=a.lengths,
    torch=old['torch'],python_abi=old['python_abi'],baseline_promotion=18006,
    files={n:hashlib.sha256((stage/n).read_bytes()).hexdigest() for n in names})
(stage/'qkv_fwd_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(stage)
