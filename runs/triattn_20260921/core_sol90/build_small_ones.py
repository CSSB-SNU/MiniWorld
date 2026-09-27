"""Re-anchor N40 denominator ones per K32 chunk; shrink shared ones8KB->2KB."""
from pathlib import Path
import hashlib,importlib.util,os
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
variant='basef40smallones';dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 rel=p.relative_to(src)
 if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
 if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
 s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
 if p.name=='triattn_m1_sm90.cuh':
  s=s.replace('cute::array_aligned<Element, kBlockN*kHeadDim, 1024> smem_ones;','cute::array_aligned<Element, CW*kHeadDim, 1024> smem_ones;')
  s=s.replace('idx < T::kBlockN*T::kHeadDim','idx < T::CW*T::kHeadDim')
  s=s.replace('auto const& operand, auto stage) __attribute__','auto const& operand, auto stage, auto column) __attribute__')
  old='constexpr int st=decltype(stage)::value;'
  assert s.count(old)==1
  s=s.replace(old,'constexpr int st=decltype(stage)::value, col=decltype(column)::value;')
  s=s.replace('uint32_t(st*T::kStageElemsV/8)<<16','uint32_t((st*T::kStageElemsV+col*CW*kHeadDim)/8)<<16')
  s=s.replace('fused_v_operand(tV(_, _, kb, c, st),Int<st>{})','fused_v_operand(tV(_, _, kb, c, st),Int<st>{},Int<c>{})')
  s=s.replace('fused_v_operand(tV(_, _, kb, cp, stp),Int<stp>{})','fused_v_operand(tV(_, _, kb, cp, stp),Int<stp>{},Int<cp>{})')
 target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('small_ones_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
