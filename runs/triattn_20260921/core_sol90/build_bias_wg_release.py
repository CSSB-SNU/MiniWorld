"""Aggregate bias-ring notifications after a complete consumer WG rendezvous."""
from pathlib import Path
import hashlib,importlib.util,os
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
variant='basebiaswgrelease';dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 rel=p.relative_to(src)
 if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
 if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
 s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
 if p.name=='triattn_m1_sm90.cuh':
  old='static constexpr int kArrivalsB = kNumMmaWG * 4 * CR;'
  assert old in s
  s=s.replace(old,'static constexpr int kArrivalsB = kNumMmaWG * (((kFlags & 1073741824) && !kSafe) ? 1 : 4) * CR;')
  a=s.index('        asm volatile("" ::: "memory"); __syncwarp();',s.index('auto bias_release_by ='))
  b=s.index('\n    };',a)
  old=s[a:b]
  s=s[:a]+'''        if constexpr(kFast){
            // All128 lanes rendezvous after their QK issue has consumed
            // each lane's shared-bias registers. Only then may one lane
            // notify the producer that this WG has freed the half slot.
            // User IDs5..7 map to hardware13..15 (ring8..11, SAFE reset12).
            cutlass::arch::NamedBarrier::sync(128,5+cwg);
            pipe_b.release(c,leader && wg_leader);
            (void)dep;
        }else{
'''+old+'''
        }'''+s[b:]
 target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('bias_wg_release_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
