"""Isolated native CUDA inference-forward artifacts; no automatic compilation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def extension(artifact):
    folder = ROOT / artifact
    info = json.loads((folder / 'build-ready.json').read_text())
    for filename, digest in info['sha256'].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(info['module'], folder / info['binary'])
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    return ext


if __name__ == '__main__':
    import argparse
    from torch.utils.cpp_extension import load
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', required=True)
    a = ap.parse_args()
    folder = ROOT / a.artifact
    build = folder / 'build'
    build.mkdir(exist_ok=True)
    os.environ['MAX_JOBS'] = '2'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    name = 'triattn_infer_' + a.artifact
    keep_flags = ['--keep', '--keep-dir', str(build)] if os.environ.get('KEEP_PTX') == '1' else []
    ext = load(name=name, sources=[str(folder / 'fused.cu')], build_directory=str(build),
               extra_include_paths=[str(ROOT.parent / 'oc/opt_core/kernels/triattn_core_broadcast/csrc'),
                                    str(ROOT.parent.parent / 'anthropic_adoption_20260919/cutlass-4.2/include')],
               extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--objdir-as-tempdir', '-std=c++17',
                   '--expt-relaxed-constexpr', '--expt-extended-lambda', '-lineinfo', '-Xptxas=-v',
                   *keep_flags], verbose=True)
    binary = str(Path(ext.__file__).relative_to(folder))
    info = dict(module=name, binary=binary, smem=ext.smem(), sha256={
        f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in ('fused.cu', binary)})
    (folder / 'build-ready.json').write_text(json.dumps(info, indent=2) + '\n')
    print('BUILD_READY', info, flush=True)
