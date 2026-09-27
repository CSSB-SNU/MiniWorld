import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
EXT={}


def extension(C):
    if C not in EXT:
        p=ROOT/f'projection_C{C}';m=json.loads((p/'ready.json').read_text())
        for name,digest in m['sha256'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest
        spec=importlib.util.spec_from_file_location(m['module'],p/m['binary'])
        ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext);EXT[C]=ext
    return EXT[C]


class Projections(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,*weights):
        ctx.save_for_backward(x,*weights)
        return tuple(F.linear(x,w) for w in weights)

    @staticmethod
    def backward(ctx,*grads):
        x,*weights=ctx.saved_tensors;C=x.shape[-1]
        dy=[g.reshape(-1,w.shape[0]).contiguous() for g,w in zip(grads,weights)]
        dx=extension(C).dgrad(dy,weights,64).view_as(x) if ctx.needs_input_grad[0] else None
        xx=x.reshape(-1,C)
        dw=[g.T@xx if needed else None for g,needed in zip(dy,ctx.needs_input_grad[1:])]
        return dx,*dw


if __name__=='__main__':
    from torch.utils.cpp_extension import load
    ap=argparse.ArgumentParser();ap.add_argument('--width',type=int,required=True);args=ap.parse_args();C=args.width
    p=ROOT/f'projection_C{C}';build=p/'build';build.mkdir(exist_ok=True)
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
    name=f'triattn_wide_projection_C{C}'
    include=ROOT.parent/'oc/opt_core/kernels/triattn_core_broadcast/csrc'
    ext=load(name=name,sources=[str(p/'projection.cu')],build_directory=str(build),
             extra_include_paths=[str(include),str(ROOT.parent.parent/'anthropic_adoption_20260919/cutlass-4.2/include')],
             extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--objdir-as-tempdir','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','-Xptxas=-v',f'-DPAIR_DIM={C}'],verbose=True)
    binary=str(Path(ext.__file__).relative_to(p))
    m=dict(module=name,binary=binary,sha256={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ('projection.cu',binary)})
    (p/'ready.json').write_text(json.dumps(m,indent=2)+'\n');print('BUILD_READY',m,flush=True)
