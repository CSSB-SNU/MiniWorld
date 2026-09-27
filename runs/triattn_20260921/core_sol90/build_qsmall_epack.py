"""Try full Q RS with current in-place exp and early packed pending E.

Packing occurs earlier than installed; a period rescale forces original SAFE
recomputation. This remains a numerical candidate until fully qualified.
"""
from pathlib import Path
import hashlib,importlib.util,os
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
variant='baseqsmallepackfull';dst=HERE/variant;dst.mkdir(exist_ok=True)
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
            auto a0=tQr0(_,_,Int<0>{});auto a1=tQr1(_,_,Int<0>{});
            warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);
        } else {warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);}'''
        assert s.count(old)==2
        s=s.replace(old,'        warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);')
        old='            if constexpr (kPrmtPack) {'
        assert s.count(old)==1
        s=s.replace(old,'            if constexpr(kFast){dst32(pr)=__float_as_uint(acc(pr));}\n            else if constexpr (kPrmtPack) {')
        a=s.index('    auto exp_chunk =');b=s.index('    // E-phase token ring',a)
        block=s[a:b]
        old='        warpgroup_fence_operand(acc);'
        assert block.count(old)==1
        block=block.replace(old,'''        if constexpr(kFast){
            #pragma unroll
            for(int pair=0;pair<8;++pair){
                auto packed=__floats2bfloat162_rn(acc(2*pair),acc(2*pair+1));
                acc(pair)=__uint_as_float(reinterpret_cast<uint32_t const&>(packed));
            }
            #pragma unroll
            for(int pair=8;pair<16;++pair)acc(pair)=0.f;
        }
'''+old)
        s=s[:a]+block+s[b:]
        old='''            if (__any_sync(0xffffffffu, out)) {
                Tensor p_rc'''
        assert s.count(old)==1
        s=s.replace(old,'''            if (__any_sync(0xffffffffu, out)) {
                // Early BF16 rounding can differ when a scale moves a
                // subnormal across its boundary. Recompute these CTAs.
                if constexpr(kFast)bad=true;
                Tensor p_rc''')
        old='''                        #pragma unroll
                        for (int ni = 0; ni < kNC; ++ni) { p_rc(mi, ni) *= f; }'''
        assert s.count(old)==1
        s=s.replace(old,'''                        if constexpr(kFast){
                            #pragma unroll
                            for(int pair=0;pair<4;++pair){
                                int ix=2*pair+mi;uint32_t bits=__float_as_uint(pend(ix));
                                auto bf=*reinterpret_cast<__nv_bfloat162 const*>(&bits);
                                float2 v=__bfloat1622float2(bf);
                                auto scaled=__floats2bfloat162_rn(v.x*f,v.y*f);
                                pend(ix)=__uint_as_float(reinterpret_cast<uint32_t const&>(scaled));
                            }
                        }else{
'''+old+'''
                        }''')
    target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('qsmall_epack_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
