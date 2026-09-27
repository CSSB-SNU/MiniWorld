"""Install explicitly qualified CUDA artifacts without changing Python dispatch.

Versioned binaries keep already-loaded extensions valid; manifests are replaced
last. The saved pre-turn manifests detect unrelated edits before installation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parents[2] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda'
ap = argparse.ArgumentParser()
ap.add_argument('--bias', required=True)
ap.add_argument('--dq', required=True)
ap.add_argument('--ln', required=True)
ap.add_argument('--bias-job', type=int, required=True)
ap.add_argument('--dq-job', type=int, required=True)
ap.add_argument('--ln-job', type=int, required=True)
ap.add_argument('--combined-job', type=int, required=True)
a = ap.parse_args()
plans = []
for kind, folder, lib, source, target in [
    ('bias', 'bias_fusion', 'triattn_bias_fusion', 'grouped.cu', 'bias_fusion.cu'),
    ('dq', 'dq', 'triattn_dq', 'fused.cu', 'dq.cu'),
    ('ln', 'ln_residual', 'triattn_ln_residual', 'fused.cu', 'ln_residual.cu'),
]:
    artifact = getattr(a, kind)
    job = getattr(a, kind + '_job')
    manifest_name = kind + '_manifest.json'
    assert (PKG / manifest_name).read_bytes() == (ROOT / 'baseline' / manifest_name).read_bytes(), manifest_name
    manifest = json.loads((PKG / manifest_name).read_text())
    src = ROOT.parent / folder / artifact
    for filename, digest in json.loads((src / 'build-ready.json').read_text()).items():
        assert hashlib.sha256((src / filename).read_bytes()).hexdigest() == digest
    qualification = (ROOT / f'qualify-{job}.log').read_text()
    assert 'API_PASS' in qualification
    assert qualification.count('ERROR SUMMARY: 0 errors') >= 2
    assert '0 hazards displayed (0 errors, 0 warnings)' in qualification
    records = json.loads((ROOT / f'module-{kind}-{artifact}-{job}.json').read_text())
    assert len(records) == 10
    for length in (384, 768, 1024):
        combined = json.loads((ROOT / f'combined-{a.combined_job}-L{length}.json').read_text())
        assert combined['complete'] and len(combined['measurements']) == 4
        assert str(src) in combined['candidate_binaries'][kind]
    name = (src / 'module-name.txt').read_text().strip()
    binary = name + '.so'
    manifest.update(module_name=name, binary=binary, artifact=artifact,
                    qualification_job=job, below80_combined_job=a.combined_job)
    if kind == 'bias':
        manifest['dq_artifact'] = a.dq
    plans.append((src, lib, source, target, manifest_name, manifest))

for src, lib, source, target, manifest_name, manifest in plans:
    binary = manifest['binary']
    shutil.copy2(src / 'build' / (lib + '.so'), PKG / (binary + '.new'))
    os.replace(PKG / (binary + '.new'), PKG / binary)
    shutil.copy2(src / source, PKG / (target + '.new'))
    os.replace(PKG / (target + '.new'), PKG / target)
    # Retain all existing API/build/header integrity entries, replace the binary.
    old = json.loads((ROOT / 'baseline' / manifest_name).read_text())
    files = set(old['files']) - {old['binary']}
    files.update((binary, target))
    manifest['files'] = {f: hashlib.sha256((PKG / f).read_bytes()).hexdigest() for f in sorted(files)}
    (PKG / (manifest_name + '.new')).write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(PKG / (manifest_name + '.new'), PKG / manifest_name)
(ROOT / 'installation.json').write_text(json.dumps({p[4]: p[5] for p in plans}, indent=2) + '\n')
print('INSTALLED', {p[4]: p[5]['artifact'] for p in plans})
