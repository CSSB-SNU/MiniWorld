"""Install the already-built CUDA/TMA artifact and record its source/binary hashes."""
from pathlib import Path
import hashlib,json,shutil,sys
import torch
HERE=Path(__file__).resolve().parent
DST=HERE.parent/'oc/opt_core/kernels/triattn_surround_tma'
shutil.copy2(HERE/'build_native/triattn_native_surround.so',DST/'triattn_native_surround.so')
names=['mask_stage.cu','prologue_pipeline.cu','prologue.cu','epilogue.cu','checks.h','bindings.cpp','triattn_native_surround.so']
manifest=dict(torch=str(torch.__version__),python_abi=sys.implementation.cache_tag,arch='sm_90a',
 sha256={n:hashlib.sha256((DST/n).read_bytes()).hexdigest() for n in names})
(DST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
