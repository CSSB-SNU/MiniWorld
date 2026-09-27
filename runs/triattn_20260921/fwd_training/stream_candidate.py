"""Explicit frozen18006 baseline and isolated full-QKV candidates."""
import importlib.util
from pathlib import Path
import sys

def module_at(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

def baseline():
    path=Path(__file__).parent/'checkpoint18006/cuda/qg_projection_attention.py'
    return module_at('triattn_frozen_qg18006',path)

def load(artifact,stage=False,installed=False):
    if installed:
        from miniworld_engine.kernels.triangle_attention.cuda import qkv_projection_attention as module,ln_backward
        return ln_backward.forward,module
    if stage:
        path=Path(__file__).parent/('stage-'+artifact)/'qkv_projection_attention.py'
        module=module_at('triattn_staged_qkv',path)
        return module.forward,module
    import stream_module as module
    return module.make_forward(artifact),module
