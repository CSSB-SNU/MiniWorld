"""Constant hot log2 offset plus full Q RS. Numerical candidate, not bitwise."""
from pathlib import Path
import hashlib,importlib.util,os
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
qr=os.environ.get('FIXED_OFFSET_QR','1')=='1'
variant='basefixedoffsetqr' if qr else 'basefixedoffsetss';dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 rel=p.relative_to(src)
 if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
 if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
 s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
 if p.name=='triattn_m1_sm90.cuh':
  if qr:
   old='static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0;';assert old in s
   s=s.replace(old,'static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0 || (!kSafe && (kFlags_ & 1073741824));')
  assert s.count('uint32_t seeded = 0;')==1 and s.count('bool need_seed = true;')==1
  s=s.replace('uint32_t seeded = 0;','uint32_t seeded = kFast ? 0xfu : 0u;')
  s=s.replace('bool need_seed = true;','bool need_seed = !kFast;')
  old='s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, nm[hh][mi]));';assert s.count(old)==1
  s=s.replace(old,'s_rc(mi, ni) = ex2_approx(fmaf(s_rc(mi, ni), c_l2, kFast ? -64.f : nm[hh][mi]));')
  start=s.index('    auto l_check =');end=s.index('    // dep:',start)
  block=s[start:end];assert block.count('if constexpr (!kSafe)')==1
  block=block.replace('if constexpr (!kSafe)','if constexpr (!kSafe && !kFast)')
  s=s[:start]+block+s[end:]
  old='if (qv < S && !uniform && !(chk > 0.f && chk < INFINITY)) { bad = true; }';assert s.count(old)==1
  s=s.replace(old,old+'''
                    // This variant never rescales hot P/O. Restrict its domain;
                    // exceptional sums are recomputed by the original SAFE path.
                    if constexpr(kFast){
                        if(qv<S && !(l>=0x1p-84f && l<=0x1p-44f))bad=true;
                    }''')
 target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build');os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2');os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('fixed_offset_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
