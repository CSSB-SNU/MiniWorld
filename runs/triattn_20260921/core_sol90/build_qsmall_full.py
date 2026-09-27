"""Full-Q register retries after constant descriptors and compact ones storage.

The mixed version gives one full WG 168 registers, two partial-Q WGs 160,
and the entire producer WG 24. Total remains 512 * 128 = 65536 registers.
"""
from pathlib import Path
import hashlib, importlib.util, os, sys
HERE = Path(__file__).resolve().parent
mixed = len(sys.argv) > 1 and sys.argv[1] == 'mixed'
variant = 'baseqsmallmixed168' if mixed else 'baseqsmallfull'
src = HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest() == '97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
dst = HERE/variant
dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
    if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'): continue
    rel = p.relative_to(src)
    if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'): continue
    if str(rel) == 'build_source.py': rel = Path('triattn_m1.py')
    s = p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
    if p.name == 'triattn_m1_sm90.cuh':
        if mixed:
            marker = '    {\n    // ================================================= CONSUMERS'
            assert s.count(marker) == 1
            s = s.replace(marker,'    auto consume = [&](auto full_tag, auto regs_tag) __attribute__((always_inline)) {\n    constexpr bool kFullQR=decltype(full_tag)::value;\n    // ================================================= CONSUMERS')
            assert s.count('warpgroup_reg_alloc<160>();') == 1
            s = s.replace('warpgroup_reg_alloc<160>();','warpgroup_reg_alloc<decltype(regs_tag)::value>();')
            old = 'cutlass::arch::warpgroup_reg_dealloc<32>();'
            assert s.count(old) == 1
            s = s.replace(old,'if constexpr(kFast)cutlass::arch::warpgroup_reg_dealloc<24>(); else cutlass::arch::warpgroup_reg_dealloc<32>();')
            old = '    }   // consumers\n'
            assert s.count(old) == 1
            s = s.replace(old,'''    };   // consumers
    if constexpr(kFast){
        if(wg_idx==1)consume(cute::true_type{},Int<168>{});
        else consume(cute::false_type{},Int<160>{});
    }else consume(cute::false_type{},Int<160>{});
''')
        else:
            marker = '    constexpr bool kFast ='
            assert s.count(marker) == 1
            s = s.replace(marker,'    constexpr bool kFullQR=true;\n'+marker)
        old = '            cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);'
        assert s.count(old) == 1
        s = s.replace(old,'''            if constexpr(kFullQR) {
                cute::gemm(tiled_mma_qkr, tQr(_,_,1), tK(_,_,1,c,st), acc);
            } else {
'''+old+'''
            }''')
        old = '''        if constexpr(kFast) {
            auto a0=tQr0(_,_,Int<0>{});auto a1=tQr1(_,_,Int<0>{});'''
        assert s.count(old) == 2
        s = s.replace(old,'''        if constexpr(kFast && !kFullQR) {
            auto a0=tQr0(_,_,Int<0>{});auto a1=tQr1(_,_,Int<0>{});''')
    target = dst/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR'] = str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
os.environ['MAX_JOBS'] = '2'
os.environ['CUTLASS_PATH'] = str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS'] = 'image'
spec = importlib.util.spec_from_file_location('qsmall_full_builder',dst/'triattn_m1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m._build(verbose=True)
print('BUILT',variant,flush=True)
