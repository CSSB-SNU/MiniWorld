"""Original committed Triton module/core, isolated from installed CUDA dispatch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
METADATA = json.loads((ROOT / 'original_triton_source.json').read_text())


def load(name):
    path = ROOT / (name + '.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == METADATA['files'][path.name]['loaded_sha256']
    qualified = 'comparison_' + name
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


core = load('original_triton_core')
module = load('original_triton_module')
# The frozen module calls kernels.triton_triangle_attention_pair_bias. Give only
# its own module globals a proxy targeting the frozen core; serving globals stay
# unchanged. All original surrounding LN/gate/projection behavior is retained.
installed_kernels = module.kernels


class KernelProxy:
    def __getattr__(self, name):
        if name == 'triton_triangle_attention_pair_bias':
            return core.triton_triangle_attention_pair_bias
        # Keep the package's lazy __getattr__ imports for original LN/gate ops.
        return getattr(installed_kernels, name)


module.kernels = KernelProxy()
TriangleAttention = module.TriangleAttention
