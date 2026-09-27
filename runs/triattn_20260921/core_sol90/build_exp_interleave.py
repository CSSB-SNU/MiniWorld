"""Interleave PV and bias prefetch between two halves of hot exponentiation."""
from pathlib import Path
import hashlib,importlib.util,os
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
variant='baseexpsplit8';dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 rel=p.relative_to(src)
 if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
 if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
 s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
 if p.name=='triattn_m1_sm90.cuh':
  a=s.index('    auto body =');b=s.index('    // drained state:',a);body=s[a:b]
  line=next(x for x in body.splitlines(True) if '        exp_chunk(accC[bq]' in x)
  body=body.replace(line,'''        if constexpr(kFast){
            Tensor row=make_tensor(accC[bq].data(),flash::convert_layout_acc_rowcol(accC[bq].layout()));
            #pragma unroll
            for(int ni=0;ni<kNC;++ni)row(0,ni)=ex2_approx(fmaf(row(0,ni),c_l2,nm[hh][0]));
        }else{
'''+line+'''        }
''')
  line=next(x for x in body.splitlines(True) if '        init_chunk(accC[bp]' in x)
  body=body.replace(line,line+'''        if constexpr(kFast){
            Tensor row=make_tensor(accC[bq].data(),flash::convert_layout_acc_rowcol(accC[bq].layout()));
            #pragma unroll
            for(int ni=0;ni<kNC;++ni)row(1,ni)=ex2_approx(fmaf(row(1,ni),c_l2,nm[hh][1]));
            warpgroup_fence_operand(accC[bq]);
        }
''')
  s=s[:a]+body+s[b:]
 target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build');os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2');os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('exp_split_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
