import hashlib
import importlib.util
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
_EXT=None

def extension():
    global _EXT
    if _EXT is None:
        manifest=json.loads((ROOT/'aux-ready.json').read_text())
        for name,digest in manifest['sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        spec=importlib.util.spec_from_file_location(manifest['module'],ROOT/manifest['binary'])
        _EXT=importlib.util.module_from_spec(spec);spec.loader.exec_module(_EXT)
    return _EXT

if __name__=='__main__':
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
    folder=ROOT/'build_aux';folder.mkdir(exist_ok=True)
    name='triattn_wide_aux'
    ext=load(name=name,sources=[str(ROOT/'aux.cu')],build_directory=str(folder),
             extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--objdir-as-tempdir','-lineinfo','-Xptxas=-v'],verbose=True)
    binary=str(Path(ext.__file__).relative_to(ROOT))
    info=dict(module=name,binary=binary,sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('aux.cu',binary)})
    (ROOT/'aux-ready.json').write_text(json.dumps(info,indent=2)+'\n')
    print('BUILD_READY',info,flush=True)
