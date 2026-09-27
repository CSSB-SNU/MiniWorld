"""Installed6d52 arithmetic/pipeline, parameterized CTA scheduling permutation."""
from pathlib import Path
import hashlib,importlib.util,os
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe'
variant='basegrid';dst=HERE/variant;dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 rel=p.relative_to(src)
 if '__pycache__' in rel.parts or str(rel) in ('__init__.py','build_native.py'):continue
 if str(rel)=='build_source.py':rel=Path('triattn_m1.py')
 s=p.read_text().replace('ta_core_broadcast','ta_sol_'+variant).replace('triattn_broadcast','triattn_sol_'+variant)
 if p.name=='triattn_m1_sm90.cuh':
  old='        int const* rowkc1;'
  assert s.count(old)==1
  s=s.replace(old,'        int grid_order;\n'+old)
  old='    int const g_qtile = int(blockIdx.x), g_rg = int(blockIdx.y), g_bh = int(blockIdx.z);'
  assert old in s
  s=s.replace(old,'''    int g_qtile = int(blockIdx.x), g_rg = int(blockIdx.y), g_bh = int(blockIdx.z);
    if constexpr(kFast){
        if(params.grid_order==9){
            int flat=g_qtile+6*(g_rg+256*(g_bh%4));
            g_qtile=flat/1024;g_rg=(flat/4)%256;g_bh=4*(g_bh/4)+flat%4;
        }else{
            int sh=params.grid_order, group=1<<sh;
            int flat=g_qtile+6*g_rg, chunk=flat/(6*group), inner=flat-chunk*(6*group);
            g_qtile=inner>>sh;g_rg=chunk*group+(inner&(group-1));
        }
    }''')
 if p.name=='launch_m1.cuh':
  s=s.replace('#include <torch/types.h>','#include <torch/types.h>\n#include <cstdlib>')
  old='    typename T::Params p{'
  assert s.count(old)==1
  s=s.replace(old,'''    char const* order_text=std::getenv("TRIATTN_SOL_GRID_ORDER");
    int grid_order=order_text?std::atoi(order_text):0;
    TORCH_CHECK(grid_order>=0 && grid_order<=9,"grid order must be 0..9");
'''+old)
  s=s.replace('n_kcol, a.rowkc0, a.rowkc1};','n_kcol, a.rowkc0, grid_order, a.rowkc1};')
 target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TRIATTN_PTXAS']='image'
spec=importlib.util.spec_from_file_location('grid_builder',dst/'triattn_m1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m._build(verbose=True)
print('BUILT',variant,flush=True)
