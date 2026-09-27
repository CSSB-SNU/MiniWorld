"""Bounded exp2 polynomial without per-logit integer range reconstruction.

For x in [-70,-62], approximate 2**(x/4) by a relative-LS quartic and
square twice. The seed64 makes this a useful candidate domain, not a promise
about inputs. Masked -infinity maps to zero; all other inputs use native EX2.
Only one quarter/eighth of hot logits use this numerical experiment.
"""
from pathlib import Path
import hashlib,importlib.util,os,sys,json
import numpy as np
HERE=Path(__file__).resolve().parent
fraction=int(sys.argv[1]);assert fraction in (4,8)
variant='baserangeexp4f%d'%fraction
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
t=np.linspace(-1.5,.5,200001);y=np.exp2(t)
c=np.linalg.lstsq(np.polynomial.polynomial.polyvander(t,4)/y[:,None],np.ones_like(t),rcond=None)[0]
err=np.polynomial.polynomial.polyval(t,c)**4/np.exp2(4*t)-1
assert max(abs(err))<.000662
(HERE/(variant+'-polynomial.json')).write_text(json.dumps({'fraction':fraction,'domain_x':[-70,-62],'coefficients':list(c),'real_max_relative_error':float(max(abs(err))),'real_rms_relative_error':float(np.sqrt(np.mean(err**2))),'cuda_accuracy_qualified':False},indent=2))
# Scaling all quartic coefficients by 2^-16 folds the fixed exponent into
# the polynomial; two squares then produce the original 2^-64 scale.
poly=repr(float(np.float32(c[4]*2**-16)))+'f'
for i in reversed(range(4)):
    poly='fmaf('+poly+',t,'+repr(float(np.float32(c[i]*2**-16)))+'f)'
helper='''
__device__ __forceinline__ float ex2_fixed_range(float x){
    if(x==-INFINITY)return 0.f;
    float t=fmaf(x,0.25f,16.f);
    if(t>=-1.5f && t<=0.5f){
        float y=POLYNOMIAL;
        y=y*y;return y*y;
    }
    return ex2_approx(x);
}
'''.replace('POLYNOMIAL',poly)
dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
    if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
    rel=p.relative_to(src)
    if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
    if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
    s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
    if p.name=='triattn_m1_sm90.cuh':
        pos=s.index('template <class T>',s.index('__device__ __forceinline__ float ex2_approx'))
        s=s[:pos]+helper+s[pos:]
        a=s.index('        if constexpr(kFast){',s.index('    auto exp_chunk ='))
        b=s.index('        }else{',a)
        s=s[:a]+'''        if constexpr(kFast){
            #pragma unroll
            for(int mi=0;mi<kNRows;++mi){
                #pragma unroll
                for(int ni=0;ni<kNC;ni+=4){
                    asm volatile(
                        "fma.rn.ftz.f32 %0,%0,%4,%5;\\n"
                        "fma.rn.ftz.f32 %1,%1,%4,%5;\\n"
                        "fma.rn.ftz.f32 %2,%2,%4,%5;\\n"
                        "fma.rn.ftz.f32 %3,%3,%4,%5;\\n"
                        : "+f"(s_rc(mi,ni)),"+f"(s_rc(mi,ni+1)),"+f"(s_rc(mi,ni+2)),"+f"(s_rc(mi,ni+3))
                        : "f"(c_l2),"f"(nm[hh][mi]));
                    if(ni%FRACTION==0)s_rc(mi,ni)=ex2_fixed_range(s_rc(mi,ni));
                    else s_rc(mi,ni)=ex2_approx(s_rc(mi,ni));
                    asm volatile(
                        "ex2.approx.ftz.f32 %0,%0;\\n"
                        "ex2.approx.ftz.f32 %1,%1;\\n"
                        "ex2.approx.ftz.f32 %2,%2;\\n"
                        : "+f"(s_rc(mi,ni+1)),"+f"(s_rc(mi,ni+2)),"+f"(s_rc(mi,ni+3)));
                }
            }
'''.replace('FRACTION',str(fraction))+s[b:]
    target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('fixed_range_exp_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
