from pathlib import Path
import os,hashlib,json,importlib.util,shutil
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
ARTIFACT_ROOT=ROOT/os.environ.get('FUSION_ARTIFACT','archive-v3')
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
        name=name_file.read_text().strip() if name_file.exists() else 'triattn_bias_fusion'
        spec=importlib.util.spec_from_file_location(name,ARTIFACT_ROOT/'build/triattn_bias_fusion.so')
        EXT=importlib.util.module_from_spec(spec);spec.loader.exec_module(EXT)
    return EXT
if __name__=='__main__':
    target=ROOT/os.environ.get('FUSION_BUILD_ROOT','.')
    build=target/'build';build.mkdir(exist_ok=True)
    name=os.environ.get('FUSION_MODULE_NAME','triattn_bias_fusion')
    ext=load(name=name,sources=[str(target/'grouped.cu')],build_directory=str(build),
        extra_include_paths=[str(ROOT.parents[1]/'oc/opt_core/kernels/triattn_core_broadcast/csrc'),str(ROOT.parents[2]/'anthropic_adoption_20260919/cutlass-4.2/include')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--objdir-as-tempdir','-std=c++17','--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo','-Xptxas=-v'],verbose=True)
    print(ext.smem())
    if name!='triattn_bias_fusion':shutil.copy2(ext.__file__,build/'triattn_bias_fusion.so')
    (target/'module-name.txt').write_text(name+'\n')
    (target/'build-ready.json').write_text(json.dumps({f:hashlib.sha256((target/f).read_bytes()).hexdigest()
        for f in ('grouped.cu','build/triattn_bias_fusion.so')},indent=2)+'\n')
