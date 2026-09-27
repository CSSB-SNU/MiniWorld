"""Install only measured, qualified changes against an intact checkpoint.

Unchanged kernel artifacts retain their manifests and binaries. Both changed
and unchanged arms of the all-gradient comparison must match exact digests.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
pkg = root.parents[2] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda'
ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--combined-job', type=int, required=True)
for kind in ('bias', 'dq'):
    ap.add_argument('--' + kind)
    ap.add_argument('--' + kind + '-job', type=int)
ap.add_argument('--fp64-job', type=int)
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()
assert a.bias or a.dq
assert Path(a.checkpoint).name == a.checkpoint

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

checkpoint = root / a.checkpoint
snapshot = json.loads((checkpoint / 'snapshot.json').read_text())
for filename, digest in snapshot.items():
    assert sha(checkpoint / filename) == digest, ('corrupt checkpoint', filename)
    assert sha(pkg / filename) == digest, ('intervening change', filename)
manifests = json.loads((root / 'installation.json').read_text())
for filename, data in manifests.items():
    assert json.loads((pkg / filename).read_text()) == data
sources = {}
for kind, folder, source, lib in (
    ('bias', 'bias_fusion', 'grouped.cu', 'triattn_bias_fusion.so'),
    ('dq', 'dq', 'fused.cu', 'triattn_dq.so'),
):
    artifact = getattr(a, kind)
    if not artifact:
        continue
    job = getattr(a, kind + '_job')
    assert job is not None
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
            assert report['complete'] and report['artifact'] == artifact
            assert report['source_digests'] == digests
    name = (path / 'module-name.txt').read_text().strip()
    assert name != manifests[kind + '_manifest.json']['module_name']
    sources[kind] = (path, source, lib, name, digests)

if a.bias:
    fp64 = json.loads((root.parent / 'below80' / ('fp64-%d.json' % a.bias_job)).read_text())
    assert len(fp64) == 24
    cancellation = json.loads((root.parent / 'bias_fusion' / ('cancellation-%d.json' % a.bias_job)).read_text())
    assert cancellation['artifact'].endswith('/' + a.bias)
    assert cancellation['passed'] and cancellation['candidate_nonzero'] == 0
if a.dq:
    assert a.fp64_job is not None
    # Query-group4 dispatch is inactive below L256; require the independent
    # reference to cover the actual new branch, not just its smaller fallback.
    dq_source = (sources['dq'][0] / sources['dq'][1]).read_text()
    fp64_lengths = (64, 128, 256) if 'launch_query_group<4>(p)' in dq_source else (64, 128)
    for length in fp64_lengths:
        for mask in ('none', 'mixed', 'one_key', 'all_masked'):
            d = json.loads((root / ('fp64-%d-L%d-%s.json' % (a.fp64_job, length, mask))).read_text())
            assert d['length'] == length and d['mask'] == mask
            assert d['native_build'] == sources['dq'][4]
            assert 'fp64_rms' in d or mask == 'all_masked'

for length in (384, 768, 1024):
    d = json.loads((root / ('combined-%d-L%d.json' % (a.combined_job, length))).read_text())
    assert d['complete'] and d['baseline_from_package'] and len(d['measurements']) == 4
    assert d['benchmark_rounds'] >= 24 and d['benchmark_replays'] >= 20
    for kind in ('bias', 'dq'):
        manifest = manifests[kind + '_manifest.json']
        baseline = snapshot[manifest['binary']]
        candidate = sources[kind][4]['build/' + sources[kind][2]] if kind in sources else baseline
        assert d['binary_digests']['baseline'][kind] == baseline
        assert d['binary_digests']['candidate'][kind] == candidate
    for value in d['measurements'].values():
        assert value['speedup'] > 1
        assert all(value['gradient_bitwise']), 'Scheduling-only changes must be bitwise equal'

print('VALIDATED_INCREMENTAL', sorted(sources), a.checkpoint, flush=True)
if a.dry_run:
    raise SystemExit(0)
for kind, (path, source, lib, name, _) in sources.items():
    data = manifests[kind + '_manifest.json']
    old_binary = data['binary']
    copy(path / 'build' / lib, pkg / (name + '.so'))
    copy(path / source, pkg / ('bias_fusion.cu' if kind == 'bias' else 'dq.cu'))
    data.update(module_name=name, binary=name + '.so', artifact=getattr(a, kind),
                qualification_job=getattr(a, kind + '_job'), core90_combined_job=a.combined_job,
                core90_followup_baseline=a.checkpoint)
    data['files'] = {filename: sha(pkg / filename) for filename in
                     sorted((set(data['files']) - {old_binary}) | {name + '.so'})}
    if kind == 'bias' and a.dq:
        data['dq_artifact'] = a.dq
    if kind == 'dq':
        data['fp64_job'] = a.fp64_job
for kind in sources:
    write_json(pkg / (kind + '_manifest.json'), manifests[kind + '_manifest.json'])
write_json(root / 'installation.json', manifests)
write_json(root / ('incremental-installation-%d.json' % a.combined_job), dict(arguments=vars(a), baseline_snapshot=snapshot))
print('INSTALLED_INCREMENTAL', sorted(sources), flush=True)
