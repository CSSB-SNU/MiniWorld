"""Install qualified follow-ups only against the frozen installed job17025 state."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
pkg = root.parents[2] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda'
ap = argparse.ArgumentParser()
for kind in ('bias', 'dq'):
    ap.add_argument('--' + kind, required=True)
    ap.add_argument('--' + kind + '-job', type=int, required=True)
ap.add_argument('--fp64-job', type=int, required=True)
ap.add_argument('--combined-job', type=int, required=True)
a = ap.parse_args()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, data):
    temp = path.with_name(path.name + '.new')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temp, path)

def copy(source, dest):
    temp = dest.with_name(dest.name + '.new')
    shutil.copy2(source, temp)
    os.replace(temp, dest)

checkpoint = root / 'checkpoint17025'
snapshot = json.loads((checkpoint / 'snapshot.json').read_text())
for filename, digest in snapshot.items():
    assert sha(pkg / filename) == digest, ('intervening change', filename)
manifests = json.loads((root / 'installation.json').read_text())
sources = {}
for kind, folder, source, lib in (
    ('bias', 'bias_fusion', 'grouped.cu', 'triattn_bias_fusion.so'),
    ('dq', 'dq', 'fused.cu', 'triattn_dq.so'),
):
    artifact = getattr(a, kind)
    job = getattr(a, kind + '_job')
    path = root.parent / folder / artifact
    digests = json.loads((path / 'build-ready.json').read_text())
    for filename, digest in digests.items():
        assert sha(path / filename) == digest
    assert '--objdir-as-tempdir' in (path / 'build.log').read_text()
    log = (root.parent / 'below80' / ('qualify-%d.log' % job)).read_text()
    assert 'API_PASS 10' in log
    assert log.count('ERROR SUMMARY: 0 errors') >= 4
    assert log.count('0 hazards displayed (0 errors, 0 warnings)') >= 2
    for length in (64, 256):
        for tool in ('memcheck', 'racecheck', 'synccheck'):
            report = json.loads((root.parent / 'below80' / ('%s-%d-L%d.json' % (tool, job, length))).read_text())
            assert report['complete'] and report['source_digests'] == digests
    name = (path / 'module-name.txt').read_text().strip()
    sources[kind] = (path, source, lib, name, digests)

assert len(json.loads((root.parent / 'below80' / ('fp64-%d.json' % a.bias_job)).read_text())) == 24
cancellation = json.loads((root.parent / 'bias_fusion' / ('cancellation-%d.json' % a.bias_job)).read_text())
assert cancellation['artifact'].endswith('/' + a.bias)
assert cancellation['passed'] and cancellation['candidate_nonzero'] == 0
for length in (64, 128):
    for mask in ('none', 'mixed', 'one_key', 'all_masked'):
        d = json.loads((root / ('fp64-%d-L%d-%s.json' % (a.fp64_job, length, mask))).read_text())
        assert d['length'] == length and d['mask'] == mask
        assert d['native_build'] == sources['dq'][4]
        assert 'fp64_rms' in d or mask == 'all_masked'

for length in (384, 768, 1024):
    d = json.loads((root / ('combined-%d-L%d.json' % (a.combined_job, length))).read_text())
    assert d['complete'] and d['baseline_from_package'] and len(d['measurements']) == 4
    assert d['benchmark_rounds'] >= 24
    for kind in ('bias', 'dq'):
        manifest = manifests[kind + '_manifest.json']
        assert d['binary_digests']['baseline'][kind] == snapshot[manifest['binary']]
        assert d['binary_digests']['candidate'][kind] == sources[kind][4]['build/' + sources[kind][2]]
    for value in d['measurements'].values():
        assert value['speedup'] > 1
        assert all(value['gradient_bitwise']), 'These scheduling-only candidates must be bitwise equal'

for kind, (path, source, lib, name, _) in sources.items():
    data = manifests[kind + '_manifest.json']
    old_binary = data['binary']
    copy(path / 'build' / lib, pkg / (name + '.so'))
    copy(path / source, pkg / ('bias_fusion.cu' if kind == 'bias' else 'dq.cu'))
    data.update(module_name=name, binary=name + '.so', artifact=getattr(a, kind),
                qualification_job=getattr(a, kind + '_job'), core90_combined_job=a.combined_job,
                core90_followup_baseline='checkpoint17025')
    data['files'] = {filename: sha(pkg / filename) for filename in
                     sorted((set(data['files']) - {old_binary}) | {name + '.so'})}
    if kind == 'bias':
        data['dq_artifact'] = a.dq
    else:
        data['fp64_job'] = a.fp64_job
for kind in ('bias', 'dq'):
    write_json(pkg / (kind + '_manifest.json'), manifests[kind + '_manifest.json'])
write_json(root / 'installation.json', manifests)
write_json(root / 'followup-installation.json', dict(arguments=vars(a), baseline_snapshot=snapshot))
print('INSTALLED_FOLLOWUP', a.bias, a.dq, flush=True)
