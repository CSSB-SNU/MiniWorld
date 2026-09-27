"""One QR176 and two SS152 consumers share the existing 65536-register pool."""
from pathlib import Path
import hashlib,importlib.util,os
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
variant='basemixedqr176';dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 rel=p.relative_to(src)
 if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
 if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
 s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
 if p.name=='triattn_m1_sm90.cuh':
  marker='    {\n    // ================================================= CONSUMERS';assert s.count(marker)==1
  s=s.replace(marker,'    auto consume = [&](auto qr_tag, auto regs_tag) __attribute__((always_inline)) {\n    // ================================================= CONSUMERS')
  assert s.count('warpgroup_reg_alloc<160>();')==1
  s=s.replace('warpgroup_reg_alloc<160>();','warpgroup_reg_alloc<decltype(regs_tag)::value>();')
  old='constexpr bool kQinRegs = T::kQinRegs;';assert s.count(old)==1
  s=s.replace(old,'constexpr bool kQinRegs = decltype(qr_tag)::value;')
  old='    }   // consumers\n';assert s.count(old)==1
  s=s.replace(old,'''    };   // consumers
    if constexpr(kFast){
        // Each allocation is uniform over a FULL warpgroup. Producer32,
        // QR176 + SS152 + SS152 total512 registers per lane ==65536/CTA.
        if(wg_idx==1)consume(cute::true_type{},Int<176>{});
        else consume(cute::false_type{},Int<152>{});
    }else consume(cute::bool_constant<T::kQinRegs>{},Int<160>{});
''')
 target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build');os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2');os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('mixed_qr_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
