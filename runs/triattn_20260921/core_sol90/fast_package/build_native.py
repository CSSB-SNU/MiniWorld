"""Rebuild this installed CUDA core and refresh its local artifact manifest."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import torch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "build"))
os.environ.setdefault("MAX_JOBS", "2")
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
os.environ.setdefault("CUTLASS_PATH", str(ROOT.parents[4] / "anthropic_adoption_20260919/cutlass-4.2"))
spec = importlib.util.spec_from_file_location("triattn_broadcast_build_source", ROOT / "build_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
module = source._build(verbose=True)
shutil.copy2(module.__file__, ROOT / "triattn_broadcast.so")
names = ["__init__.py", "build_source.py", "build_native.py", "triattn_broadcast.so"]
names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "csrc").rglob("*")) if p.is_file()]
manifest = dict(torch=str(torch.__version__), python_abi=sys.implementation.cache_tag, arch="sm_90a",
                sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names})
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
