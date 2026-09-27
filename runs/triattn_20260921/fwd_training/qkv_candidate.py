"""Load a frozen experimental, staged, or installed Q fusion for verification."""
import importlib.util
from pathlib import Path
import sys
from miniworld_engine.kernels.triangle_attention.cuda import ln_backward

def load(artifact,stage=False,installed=False):
    if installed:
        from miniworld_engine.kernels.triangle_attention.cuda import q_projection_attention as module
        return ln_backward.forward,module
    if stage:
        path=Path(__file__).parent/('stage-'+artifact)/'q_projection_attention.py'
        spec=importlib.util.spec_from_file_location('triattn_staged_q_fusion',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        return module.forward,module
    import qkv_module as module
    return module.make_forward(artifact),module
