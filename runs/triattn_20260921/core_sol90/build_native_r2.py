"""Installed four-score pipeline with two 224-register consumers, full Q RS."""
from pathlib import Path
import hashlib,importlib.util,os
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
qr=os.environ.get('NATIVE_R2_QR','1')=='1'
variant='basenativer2qr' if qr else 'basenativer2ss';dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 rel=p.relative_to(src)
 if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
 if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
 s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
 if p.name=='build_source.py':
  assert s.count('((N + 2) // 3 + 1)')==1
  s=s.replace('((N + 2) // 3 + 1)','((N + 1) // 2 + 1)')
 if p.name=='triattn_m1_sm90.cuh':
  assert s.count('CW = 32, R = 3,')==1
  s=s.replace('CW = 32, R = 3,','CW = 32, R = 2,')
  s=s.replace('warpgroup_reg_alloc<160>()','warpgroup_reg_alloc<224>()')
  s=s.replace('128*32 + 384*160 = 65536','128*32 + 256*224 = 61440 <= 384*168 = 64512 launch pool')
  for label in ('jb_','je_','fr_'):
   old='((cwg == 1) ? '+label+'[1] : '+label+'[2])'
   assert s.count(old)==1
   s=s.replace(old,label+'[1]')
  assert s.count('(jb_[0] | jb_[1] | jb_[2])')==1
  s=s.replace('(jb_[0] | jb_[1] | jb_[2])','(jb_[0] | jb_[1])')
  if qr:
   old='static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0;';assert old in s
   s=s.replace(old,'static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0 || (!kSafe && (kFlags_ & 1073741824));')
 target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build');os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2');os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('native_r2_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
