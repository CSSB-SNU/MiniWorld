import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / os.environ.get('WGRAD_ARTIFACT', 'shared_z')
EXT = None


def extension():
    global EXT
    if EXT is None:
        for name, digest in json.loads((ARTIFACT / 'build-ready.json').read_text()).items():
            assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
        name = (ARTIFACT / 'module-name.txt').read_text().strip()
        spec = importlib.util.spec_from_file_location(name, ARTIFACT / 'build/triattn_wgrad.so')
        EXT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(EXT)
    return EXT


if __name__ == '__main__':
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    os.environ['MAX_JOBS'] = '2'
    target = ROOT / os.environ.get('WGRAD_BUILD_ROOT', 'shared_z')
    (target / 'build').mkdir(parents=True, exist_ok=True)
    name = os.environ.get('WGRAD_MODULE_NAME', 'triattn_wgrad_shared_z')
    ext = load(name=name, sources=[str(target / 'grouped.cu')], build_directory=str(target / 'build'),
        extra_include_paths=[str(ROOT.parents[1] / 'oc/opt_core/kernels/triattn_core_broadcast/csrc'),
                             str(ROOT.parents[2] / 'anthropic_adoption_20260919/cutlass-4.2/include')],
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr',
            '--expt-extended-lambda', '-lineinfo', '-Xptxas=-v'], verbose=True)
    shutil.copy2(ext.__file__, target / 'build/triattn_wgrad.so')
    (target / 'module-name.txt').write_text(name + '\n')
    (target / 'build-ready.json').write_text(json.dumps({f: hashlib.sha256((target / f).read_bytes()).hexdigest()
        for f in ['grouped.cu', 'build/triattn_wgrad.so']}, indent=2) + '\n')
    print('SMEM', ext.smem(), flush=True)
