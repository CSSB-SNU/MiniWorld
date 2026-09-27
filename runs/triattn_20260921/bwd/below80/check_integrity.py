import hashlib
import json
from pathlib import Path

from miniworld_engine.kernels.triangle_attention import cuda
from miniworld_engine.modules import TriangleAttention

root = Path(cuda.__file__).parent
expected = json.loads((Path(__file__).parent / 'installation.json').read_text())
for name in ['manifest.json', 'bias_manifest.json', 'dq_manifest.json', 'ln_manifest.json', 'gate_manifest.json']:
    data = json.loads((root / name).read_text())
    hashes = data['sha256'] if 'sha256' in data else data['files']
    for filename, digest in hashes.items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest, (name, filename)
    if name in expected:
        assert data == expected[name]
    print('MANIFEST_PASS', name, data.get('artifact'), flush=True)
model = TriangleAttention(128, n_head=4, d_hidden=128, implementation='triton')
for flag in ['_fuse_projection_backward', '_fuse_bias_backward', '_fuse_dq_backward', '_fuse_front_backward', '_fuse_gate_backward']:
    assert getattr(model, flag) is True, flag
print('DEFAULT_FLAGS_PASS', flush=True)
