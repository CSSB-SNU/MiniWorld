"""Install the qualified CUDA core plus its optional native-wrapper dispatch."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
import torch
HERE=Path(__file__).resolve().parent
R=HERE.parent
DST=R/'oc/opt_core/kernels/triattn_core_broadcast'
DST.mkdir(exist_ok=True)
shutil.copy2(HERE/'package/__init__.py',DST/'__init__.py')
shutil.copy2(HERE/'package/build_native.py',DST/'build_native.py')
shutil.copy2(HERE/'build/triattn_broadcast/triattn_broadcast.so',DST/'triattn_broadcast.so')
shutil.copytree(HERE/'broadcast/csrc',DST/'csrc',dirs_exist_ok=True)
shutil.copy2(HERE/'broadcast/triattn_m1.py',DST/'build_source.py')
names=['__init__.py','build_source.py','build_native.py','triattn_broadcast.so']+[str(p.relative_to(DST)) for p in sorted((DST/'csrc').rglob('*')) if p.is_file()]
manifest=dict(torch=str(torch.__version__),python_abi=sys.implementation.cache_tag,arch='sm_90a',
             sha256={n:hashlib.sha256((DST/n).read_bytes()).hexdigest() for n in names})
(DST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
wrapper=R/'oc/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/cuda_b/triattn_m1.py'
old=wrapper.read_text()
backup=HERE/'triattn_m1_before.py'
if not backup.exists():backup.write_text(old)
needle='    ext.fwd(qkv[0], qkv[1], qkv[2], bias_staged,'
if 'from opt_core.kernels.triattn_core_broadcast import select_core' not in old:
    hook="""    # This specialization changes the attention loop, while reusing the same
    # mask/bias staging, numerical contract and SAFE recomputation interface.
    try:
        from opt_core.kernels.triattn_core_broadcast import select_core
    except ImportError:
        selected_core = None
    else:
        selected_core = select_core(q, mask, flags, trace)
    compute_ext = selected_core if selected_core is not None else ext
"""
    assert old.count(needle)==1
    wrapper.write_text(old.replace(needle,hook+needle.replace('ext.fwd','compute_ext.fwd')))
subprocess.run([sys.executable,str(R/'core_pipeline/refresh_core_record.py')],check=True)
print(json.dumps(manifest,indent=2))
