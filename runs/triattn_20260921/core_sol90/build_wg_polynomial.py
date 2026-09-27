"""Spatial pipe balancing: one consumer WG evaluates a cubic, two use MUFU."""
from pathlib import Path
import hashlib,importlib.util,os
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
variant='basewgpoly3';dst=HERE/variant;dst.mkdir(exist_ok=True)
helper='''
// Same cubic/error domain as the previously checked scalar polynomial, but
// assigned to a whole WG instead of mixing SFU/FMA inside every warp.
__device__ __forceinline__ float ex2_wg_poly(float x){
    float rounded=__fadd_rn(x,12582912.f);
    float n=__fadd_rn(rounded,-12582912.f);
    float f=__fadd_rn(x,-n);
    float p=fmaf(fmaf(fmaf(0.05587550535773725f,f,0.24229446522590045f),f,0.6931272658129415f),f,0.9999482425444736f);
    unsigned exponent=__float_as_uint(rounded)<<23;
    float y=__uint_as_float(__float_as_uint(p)+exponent);
    return x < -126.f ? 0.f : (x >= 128.f ? INFINITY : y);
}
'''
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 rel=p.relative_to(src)
 if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
 if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
 s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
 if p.name=='triattn_m1_sm90.cuh':
  pos=s.index('template <class T>',s.index('__device__ __forceinline__ float ex2_approx'))
  s=s[:pos]+helper+s[pos:]
  old='''        #pragma unroll
        for (int mi = 0; mi < kNRows; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < kNC; ++ni) { s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi])); }
        }
        warpgroup_fence_operand(acc);''';assert s.count(old)==1
  s=s.replace(old,'''        if(kFast && cwg==0){
            #pragma unroll
            for(int mi=0;mi<kNRows;++mi){
                #pragma unroll
                for(int ni=0;ni<kNC;++ni)s_rc(mi,ni)=ex2_wg_poly(fmaf(s_rc(mi,ni),c_l2,nm[hh][mi]));
            }
        }else{
'''+old.split('        warpgroup_fence_operand(acc);')[0]+'''        }
        warpgroup_fence_operand(acc);''')
 target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build');os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2');os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('wg_poly_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
