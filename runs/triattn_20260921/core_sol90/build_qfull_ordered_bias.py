"""Full Q registers on installed four-score pipeline, ordered bias LDS.

Form the zero dependency from all P registers before issuing any PV that
reads them. The later bias initializer consumes that dependency. No operand
of an in-flight WGMMA is read to construct it, and addresses stay unchanged.
"""
from pathlib import Path
import hashlib, importlib.util, os
HERE=Path(__file__).resolve().parent
variant='baseqfullorderedbias'
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
    if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
    rel=p.relative_to(src)
    if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
    if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
    s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
    if p.name=='triattn_m1_sm90.cuh':
        old='            cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);'
        assert s.count(old)==1
        s=s.replace(old,'            cute::gemm(tiled_mma_qkr, tQr(_,_,1), tK(_,_,1,c,st), acc);')
        old='''        if constexpr(kFast) {
            auto a0=tQr0(_,_,Int<0>{});auto a1=tQr1(_,_,Int<0>{});'''
        assert s.count(old)==2
        s=s.replace(old,'''        if constexpr(false) {
            auto a0=tQr0(_,_,Int<0>{});auto a1=tQr1(_,_,Int<0>{});''')
        a=s.index('    auto init_chunk =');b=s.index('    auto issue_qk =',a)
        block=s[a:b]
        old='auto cc, auto hc) __attribute__((always_inline))'
        assert block.count(old)==1
        block=block.replace(old,'auto cc, auto hc, uint32_t ordered_zero=0) __attribute__((always_inline))')
        old='bias_thread + c * T::kSlotElems'
        assert block.count(old)==1
        block=block.replace(old,'bias_thread + ordered_zero + c * T::kSlotElems')
        s=s[:a]+block+s[b:]
        old='        if constexpr (kFast) { pack_chunk(accC[bp], PCb[pb]); warpgroup_fence_operand(PCb[pb]); }'
        assert s.count(old)==1
        s=s.replace(old,'''        uint32_t pack_ordered_zero=0;
        if constexpr(kFast){
            pack_chunk(accC[bp],PCb[pb]);
            auto packed_words=recast<uint32_t>(PCb[pb]);
            uint32_t packed_sum=0;
            #pragma unroll
            for(int word=0;word<8;++word)packed_sum+=packed_words(word);
            pack_ordered_zero=packed_sum&uint32_t(params.zero);
            warpgroup_fence_operand(PCb[pb]);
        }''')
        old='        init_chunk(accC[bp], Int<c3>{}, Int<h3>{});'
        assert s.count(old)==1
        s=s.replace(old,'        init_chunk(accC[bp], Int<c3>{}, Int<h3>{},pack_ordered_zero);')
    target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('ordered_bias_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
