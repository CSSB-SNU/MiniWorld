"""Build/load isolated wide native CUDA artifacts; never compile at import."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_EXTENSIONS = {}


def extension(head_dim, kind):
    key = head_dim,kind
    if key not in _EXTENSIONS:
        folder = ROOT / f'native_head{head_dim}'
        info = json.loads((folder / f'{kind}-ready.json').read_text())
        for name, digest in info['sha256'].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, name
        spec = importlib.util.spec_from_file_location(info['module'], folder/info['binary'])
        ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
        _EXTENSIONS[key] = ext
    return _EXTENSIONS[key]


if __name__ == '__main__':
    from torch.utils.cpp_extension import load
    ap = argparse.ArgumentParser(); ap.add_argument('--head-dim',type=int,required=True)
    ap.add_argument('--kind',choices=['fwd','dq','dkdv'],required=True)
    args = ap.parse_args()
    folder = ROOT/f'native_head{args.head_dim}'
    build = folder/f'build_{args.kind}'; build.mkdir(exist_ok=True)
    source = folder/f'{args.kind}.cu'
    manifests = json.loads((folder/'sources.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifests['source_sha256'][source.name]
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'; os.environ['MAX_JOBS']='2'
    name = f'triattn_wide_h{args.head_dim}_{args.kind}'
    include = ROOT.parent/'oc/opt_core/kernels/triattn_core_broadcast/csrc'
    ext = load(name=name,sources=[str(source)],build_directory=str(build),
               extra_include_paths=[str(include),str(ROOT.parent.parent/'anthropic_adoption_20260919/cutlass-4.2/include')],
               extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--objdir-as-tempdir','-std=c++17',
               '--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','-Xptxas=-v',
               f'-DHEAD_DIM={args.head_dim}'],verbose=True)
    binary = str(Path(ext.__file__).relative_to(folder))
    info = dict(module=name,binary=binary,smem=ext.smem(),head_dim=args.head_dim,
                sha256={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in (source.name,binary)},
                fa3_utils_sha256=hashlib.sha256((include/'fa3_utils.h').read_bytes()).hexdigest())
    (folder/f'{args.kind}-ready.json').write_text(json.dumps(info,indent=2)+'\n')
    print('BUILD_READY', info,flush=True)
