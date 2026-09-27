"""Build three-quarter Q cache in final namespace without publishing it."""
from pathlib import Path
import hashlib,json,os,runpy
from three_quarter_q import transform
HERE=Path(__file__).resolve().parent
src=HERE.parent/'oc/opt_core/kernels/triattn_core_broadcast'
assert hashlib.sha256((src/'triattn_broadcast.so').read_bytes()).hexdigest()=='97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b'
record=json.loads((src/'manifest.json').read_text())
for name,sha in record['sha256'].items():
 assert hashlib.sha256((src/name).read_bytes()).hexdigest()==sha,name
dst=HERE/'qthree_package';dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 rel=p.relative_to(src)
 if '__pycache__' in rel.parts or 'build' in rel.parts:continue
 s=p.read_text()
 if p.name=='triattn_m1_sm90.cuh':s=transform(s,0)
 target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'qthree_package_build')
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2';os.environ['TRIATTN_PTXAS']='image'
runpy.run_path(str(dst/'build_native.py'),run_name='__main__')
print('BUILT baseqthreepackaged',flush=True)
