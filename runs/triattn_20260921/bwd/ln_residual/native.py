from pathlib import Path
import os,hashlib,json,importlib.util,shutil
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
ARTIFACT_ROOT=ROOT/os.environ.get("LN_ARTIFACT",".")
os.environ['MAX_JOBS']='2'
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
EXT=None
def extension():
    global EXT
    if EXT is None:
        data=json.loads((ARTIFACT_ROOT/'build-ready.json').read_text())
        for filename,digest in data.items():
            assert hashlib.sha256((ARTIFACT_ROOT/filename).read_bytes()).hexdigest()==digest,('stale build',filename)
        name_file=ARTIFACT_ROOT/'module-name.txt'
        name=name_file.read_text().strip() if name_file.exists() else 'triattn_ln_residual'
        spec=importlib.util.spec_from_file_location(name,ARTIFACT_ROOT/'build/triattn_ln_residual.so')
        EXT=importlib.util.module_from_spec(spec);spec.loader.exec_module(EXT)
    return EXT
if __name__=='__main__':
    target=ROOT/os.environ.get("LN_BUILD_ROOT",".")
    build=target/'build';build.mkdir(exist_ok=True)
    name=os.environ.get('LN_MODULE_NAME','triattn_ln_residual')
    ext=load(name=name,sources=[str(target/'fused.cu')],build_directory=str(build),
        extra_include_paths=[str(ROOT.parents[1]/'oc/opt_core/kernels/triattn_core_broadcast/csrc'),str(ROOT.parents[2]/'anthropic_adoption_20260919/cutlass-4.2/include')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','-Xptxas=-v'],verbose=True)
    print(ext.smem())
    if name!='triattn_ln_residual':shutil.copy2(ext.__file__,build/'triattn_ln_residual.so')
    (target/'module-name.txt').write_text(name+'\n')
    (target/'build-ready.json').write_text(json.dumps({f:hashlib.sha256((target/f).read_bytes()).hexdigest()
        for f in ('fused.cu','build/triattn_ln_residual.so')},indent=2)+'\n')
