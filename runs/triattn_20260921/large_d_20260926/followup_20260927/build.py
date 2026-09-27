import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
CONFIGS={
    'pipe3':dict(TILE_N=128,TILE_K=64,STAGES=3,CONSUMERS=1,CONSUMER_REGS=128,MIN_BLOCKS=2,DYNAMIC_REG=0),
    'share2':dict(TILE_N=128,TILE_K=64,STAGES=2,CONSUMERS=2,CONSUMER_REGS=96,MIN_BLOCKS=1,DYNAMIC_REG=0),
    'n256':dict(TILE_N=256,TILE_K=64,STAGES=2,CONSUMERS=1,CONSUMER_REGS=192,MIN_BLOCKS=1,DYNAMIC_REG=0),
    'k128':dict(TILE_N=128,TILE_K=128,STAGES=2,CONSUMERS=1,CONSUMER_REGS=128,MIN_BLOCKS=2,DYNAMIC_REG=1),
}
EXT={}
def extension(kind,C):
    key=kind,C
    if key not in EXT:
        p=ROOT/f'{kind}_C{C}';m=json.loads((p/'ready.json').read_text())
        for n,h in m['sha256'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
        spec=importlib.util.spec_from_file_location(m['module'],p/m['binary'])
        ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext);EXT[key]=ext
    return EXT[key]
if __name__=='__main__':
    from torch.utils.cpp_extension import load
    ap=argparse.ArgumentParser();ap.add_argument('--kind',required=True);ap.add_argument('--width',type=int,required=True)
    args=ap.parse_args();kind,C=args.kind,args.width
    p=ROOT/f'{kind}_C{C}';p.mkdir(exist_ok=True);build=p/'build';build.mkdir(exist_ok=True)
    source=ROOT/('bias.cu' if kind=='bias16' else kind+'.cu' if kind.startswith('accumulate') else 'projection.cu')
    (p/'kernel.cu').write_bytes(source.read_bytes())
    defines=dict(HEAD_DIM=C//4) if kind=='bias16' else {} if kind.startswith('accumulate') else dict(PAIR_DIM=C,**CONFIGS[kind])
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
    name=f'triattn_wide_next_{kind}_C{C}'
    include=BASE.parent/'oc/opt_core/kernels/triattn_core_broadcast/csrc'
    ext=load(name=name,sources=[str(p/'kernel.cu')],build_directory=str(build),
             extra_include_paths=[str(include),str(BASE.parent.parent/'anthropic_adoption_20260919/cutlass-4.2/include')],
             extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--objdir-as-tempdir','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','-Xptxas=-v']+[f'-D{k}={v}' for k,v in defines.items()],verbose=True)
    binary=str(Path(ext.__file__).relative_to(p))
    m=dict(module=name,binary=binary,config=defines,smem=ext.smem(),sha256={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ('kernel.cu',binary)})
    (p/'ready.json').write_text(json.dumps(m,indent=2)+'\n');print('BUILD_READY',m,flush=True)
