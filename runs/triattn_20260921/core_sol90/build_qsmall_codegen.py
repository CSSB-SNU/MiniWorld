"""Isolated compiler-scheduling controls on the exact installed CUDA source."""
from pathlib import Path
import hashlib, importlib.util, os, sys

HERE = Path(__file__).resolve().parent
mode = sys.argv[1]
extra = {'o2': '-Xptxas=-O2', 'ru0': '-Xptxas=--register-usage-level=0'}[mode]
variant = 'baseqsmallcode' + mode
src = HERE.parent / 'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src / 'triattn_broadcast.so').read_bytes()).hexdigest() == '97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
dst = HERE / variant
dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
    if not p.is_file() or p.suffix not in ('.py', '.cu', '.cuh', '.h'):
        continue
    rel = p.relative_to(src)
    if '__pycache__' in rel.parts or str(rel) in ('__init__.py', 'build_native.py'):
        continue
    s = p.read_text().replace('ta_core_broadcast', 'ta_sol_' + variant).replace('triattn_broadcast', 'triattn_sol_' + variant)
    if str(rel) == 'build_source.py':
        rel = Path('triattn_m1.py')
        marker = '    csrc = os.path.join(_HERE, "csrc", "m1")'
        assert s.count(marker) == 1
        s = s.replace(marker, '    flags.append(' + repr(extra) + ')\n' + marker)
    target = dst / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR'] = str(HERE / 'build')
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
os.environ['MAX_JOBS'] = '2'
os.environ['CUTLASS_PATH'] = str(HERE.parents[1] / 'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS'] = 'image'
spec = importlib.util.spec_from_file_location('qsmall_codegen_builder', dst / 'triattn_m1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m._build(verbose=True)
print('BUILT', variant, flush=True)
