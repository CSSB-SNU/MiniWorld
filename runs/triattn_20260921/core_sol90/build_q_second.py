"""Keep the second K16 Q slice in registers, preserving QK accumulation order."""
from pathlib import Path
import hashlib, importlib.util, os
HERE = Path(__file__).resolve().parent
src = HERE.parent / 'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest() == '97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
variant = 'baseqsmallk1'
dst = HERE / variant
dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
    if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'): continue
    rel = p.relative_to(src)
    if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'): continue
    if str(rel) == 'build_source.py': rel = Path('triattn_m1.py')
    s = p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
    if p.name == 'triattn_m1_sm90.cuh':
        old = '''            cute::gemm(tiled_mma_qkr, tQr(_,_,0), tK(_,_,0,c,st), acc);
            cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);'''
        new = '''            cute::gemm(tiled_mma_qk, tQ(_,_,0,hh,cwg), tK(_,_,0,c,st), acc);
            cute::gemm(tiled_mma_qkr, tQr(_,_,1), tK(_,_,1,c,st), acc);'''
        assert s.count(old) == 1
        s = s.replace(old,new)
        old = 'auto a0=tQr0(_,_,Int<0>{});auto a1=tQr1(_,_,Int<0>{});'
        assert s.count(old) == 2
        s = s.replace(old,'auto a0=tQr0(_,_,Int<1>{});auto a1=tQr1(_,_,Int<1>{});')
    target = dst/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR'] = str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
os.environ['MAX_JOBS'] = '2'
os.environ['CUTLASS_PATH'] = str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS'] = 'image'
spec = importlib.util.spec_from_file_location('q_second_builder',dst/'triattn_m1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m._build(verbose=True)
print('BUILT',variant,flush=True)
