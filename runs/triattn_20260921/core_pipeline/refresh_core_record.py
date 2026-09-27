"""Record the optional mask hook's Python digest; the core CUDA binary is unchanged."""
from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).resolve().parents[1]
PKG=R/'oc/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg'
record=PKG/'prebuilt/torch2.10.0+cu128-cpython-310-x86_64-linux-gnu/triattn_m1_ext.json'
data=json.loads(record.read_text())
source=PKG/'cuda_b/triattn_m1.py'
assert hashlib.sha256(record.with_suffix('.so').read_bytes()).hexdigest()==data['so_sha256']
data['module_sha256']['triattn_m1.py']=hashlib.sha256(source.read_bytes()).hexdigest()
record.write_text(json.dumps(data,indent=1,sort_keys=True)+'\n')
subprocess.run([sys.executable,str(R/'refresh_sums.py'),str(R/'oc')],check=True)
