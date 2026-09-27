from pathlib import Path
import hashlib, importlib.util, os, sys
from lean_producer_q import transform
HERE = Path(__file__).resolve().parent
kind = sys.argv[1]
assert kind in ('plain', 'mixed')
variant = 'baseqthreelean' + kind
src = HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest() == '4be5b7cdd291150436b9b048318b6cac7977b0db82b9a4f748520d398cfb4c32'
dst = HERE/variant
dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
    if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'): continue
    rel = p.relative_to(src)
    if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'): continue
    s = p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
    if p.name == 'triattn_m1_sm90.cuh': s = transform(s, kind == 'mixed')
    if str(rel) == 'build_source.py':
        rel = Path('triattn_m1.py')
        marker = '    csrc = os.path.join(_HERE, "csrc", "m1")'
        assert s.count(marker) == 1
        keep = HERE/'build'/('triattn_sol_'+variant)
        s = s.replace(marker, '    flags.extend(["--keep", "--keep-dir='+str(keep)+'"])\n'+marker)
    target = dst/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(s)
if '--generate-only' in sys.argv:
    print('GENERATED',variant);sys.exit(0)
os.environ['TORCH_EXTENSIONS_DIR'] = str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
os.environ['MAX_JOBS'] = '2'
os.environ['CUTLASS_PATH'] = str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS'] = 'image'
spec = importlib.util.spec_from_file_location('lean_producer_builder',dst/'triattn_m1.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
