"""Build L1024 specialization in the final namespace without installing it."""
from pathlib import Path
import hashlib,json,os,runpy
from specialize_1024_q import transform
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='4be5b7cdd291150436b9b048318b6cac7977b0db82b9a4f748520d398cfb4c32'
record=json.loads((src/'manifest.json').read_text())
for name,sha in record['sha256'].items():
 assert hashlib.sha256((src/name).read_bytes()).hexdigest()==sha,name
dst=HERE/'q1024_package';dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 rel=p.relative_to(src)
 if '__pycache__' in rel.parts or 'build' in rel.parts:continue
 s=transform(p.name,p.read_text())
 target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'q1024_package_build')
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2';os.environ['TRIATTN_PTXAS']='image'
runpy.run_path(str(dst/'build_native.py'),run_name='__main__')
print('BUILT baseq1024packaged',flush=True)
