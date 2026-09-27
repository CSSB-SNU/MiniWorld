"""Replace the installed CUDA forward only after paired incremental gates pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from miniworld_engine.kernels.triangle_attention import cuda
from miniworld_engine.kernels.triangle_attention.triton import main as core

ap = argparse.ArgumentParser()
ap.add_argument('--artifact', required=True)
ap.add_argument('--baseline-artifact', default='cooperative_q2')
ap.add_argument('--baseline-promotion', type=int, default=17310)
ap.add_argument('--threads', type=int, default=128)
ap.add_argument('--query-tile', type=int, default=64)
ap.add_argument('--key-tile', type=int, default=64)
ap.add_argument('--stages', type=int, default=2)
ap.add_argument('--min-forward-speedup', type=float, default=1.015)
ap.add_argument('--require-forward-ci', action='store_true')
ap.add_argument('--installed-rounds', type=int, default=32)
for name in ('qualify', 'bench', 'profile', 'sanitize'):
    ap.add_argument('--' + name, required=True, type=int)
a = ap.parse_args()
assert a.min_forward_speedup >= 1.0
assert a.min_forward_speedup >= 1.015 or a.require_forward_ci
assert a.installed_rounds >= 32
root = Path(__file__).resolve().parent
pkg = Path(cuda.__file__).parent
dispatcher = Path(core.__file__)
job = os.environ['SLURM_JOB_ID']
artifact = root / a.artifact
baseline = root / a.baseline_artifact
report = dict(job=job, artifact=a.artifact, baseline_artifact=a.baseline_artifact,
              baseline_promotion=a.baseline_promotion, state='checking', evidence=vars(a))
result = root / ('promotion-' + job + '.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save():
    result.write_text(json.dumps(report, indent=2) + '\n')


def run(script, *args):
    subprocess.run([sys.executable, '-u', '-B', str(root / script), *map(str, args)], check=True)


def paired_ci(row):
    import numpy as np
    ratios = np.asarray(row['paired_ratios'], dtype=np.float64)
    assert len(ratios) >= 32 and np.isfinite(ratios).all()
    rng = np.random.default_rng(17345)
    medians = np.median(ratios[rng.integers(0, len(ratios), (10000, len(ratios)))], axis=1)
    return np.quantile(medians, [.025, .975]).tolist()


def check_bench(data):
    assert data['complete'] and data['candidate_build'] == build
    assert len(data['records']) == 6
    assert data['baseline_artifact'] == a.baseline_artifact
    assert data['baseline_build'] == baseline_build
    for row in data['records']:
        if row['kind'] == 'forward':
            assert row['speedup'] > a.min_forward_speedup, row
            if a.require_forward_ci:
                assert paired_ci(row)[0] > 1., row
        elif row['kind'] == 'forward_backward':
            assert row['speedup'] > 1., row


build = read(artifact / 'build-ready.json')
baseline_build = read(baseline / 'build-ready.json')
for directory, info in ((artifact, build), (baseline, baseline_build)):
    for filename, sha in info['sha256'].items():
        assert digest(directory / filename) == sha
qual = read(root / f'qualify-{a.qualify}.json')
assert qual['complete'] and qual['candidate_build'] == build
qlog = (root / f'qualify-{a.qualify}.log').read_text()
assert qlog.count('ERROR SUMMARY: 0 errors') == 4
assert qlog.count('0 hazards displayed (0 errors, 0 warnings)') == 2
flog = (root / f'full-sanitize-{a.sanitize}.log').read_text()
assert flog.count('ERROR SUMMARY: 0 errors') == 2
assert flog.count('0 hazards displayed (0 errors, 0 warnings)') == 1
for tool in ('memcheck', 'racecheck', 'synccheck'):
    data = read(root / f'{tool}-{a.sanitize}-L768.json')
    assert data['complete'] and data['artifact'] == a.artifact
for length in (384, 768, 1024):
    check_bench(read(root / f'bench-{a.bench}-L{length}.json'))
profile = read(root / f'source-profile-{a.profile}.json')
assert profile['complete'] and profile['build'] == build

old_manifest = read(pkg / 'fwd_manifest.json')
assert old_manifest['artifact'] == a.baseline_artifact
baseline_promotion = read(root / f'promotion-{a.baseline_promotion}.json')
assert baseline_promotion['state'] == 'complete' and baseline_promotion['manifest'] == old_manifest
assert digest(pkg / 'training_forward.cu') == baseline_build['sha256']['fused.cu']
for name, info in qual['manifests'].items():
    assert read(pkg / name) == info
    for filename, sha in info.get('files', info.get('sha256', {})).items():
        assert digest(pkg / filename) == sha
assert digest(dispatcher) == old_manifest['dispatcher']['sha256']

backup = root / ('before-' + job)
backup.mkdir()
names = ['training_forward.cu', 'fwd_manifest.json', build['module'] + '.so']
existing = [name for name in names if (pkg / name).exists()]
for name in existing:
    shutil.copy2(pkg / name, backup / name)
shutil.copy2(dispatcher, backup / 'main.py')
report.update(state='installing', backup=str(backup), previous_manifest=old_manifest)
save()

try:
    shutil.copy2(artifact / 'fused.cu', pkg / 'training_forward.cu')
    binary = build['module'] + '.so'
    shutil.copy2(artifact / build['binary'], pkg / binary)
    manifest = dict(old_manifest)
    manifest.update(module_name=build['module'], binary=binary, artifact=a.artifact,
                    qualification_job=a.qualify, benchmark_job=a.bench,
                    profile_job=a.profile, full_sanitizer_job=a.sanitize,
                    baseline_promotion=a.baseline_promotion, kv_stages=a.stages,
                    query_tile=a.query_tile, key_tile=a.key_tile, threads=a.threads)
    manifest['files'] = {name: digest(pkg / name) for name in
                         ('training_forward.py', 'training_forward.cu', binary, 'fa3_utils.h')}
    staged = pkg / ('fwd_manifest-' + job + '.tmp')
    staged.write_text(json.dumps(manifest, indent=2) + '\n')
    staged.replace(pkg / 'fwd_manifest.json')
    report.update(state='verifying', manifest=manifest)
    save()
    run('verify_installed.py', '--output', root / f'installed-{job}.json')
    run('verify_amp.py', '--output', root / f'amp-{job}.json')
    for length in (384, 768, 1024):
        path = root / f'installed-bench-{job}-L{length}.json'
        run('module_check.py', '--artifact', a.artifact, '--baseline-artifact', a.baseline_artifact,
            '--length', length, '--mode', 'bench', '--installed', '--rounds', a.installed_rounds,
            '--replays', 40, '--output', path)
        data = read(path)
        assert data['installed_dispatch']
        check_bench(data)
    for filename, sha in manifest['files'].items():
        assert digest(pkg / filename) == sha
    assert digest(dispatcher) == old_manifest['dispatcher']['sha256']
    for name, info in qual['manifests'].items():
        if name != 'fwd_manifest.json':
            assert read(pkg / name) == info
            for filename, sha in info.get('files', info.get('sha256', {})).items():
                assert digest(pkg / filename) == sha
    snapshot = root / ('checkpoint' + job)
    (snapshot / 'cuda').mkdir(parents=True)
    (snapshot / 'triton').mkdir()
    frozen_files = set()
    for path in pkg.glob('*manifest.json'):
        info = read(path)
        frozen_files.add(path.name)
        frozen_files.update(info.get('files', info.get('sha256', {})))
    for name in sorted(frozen_files):
        shutil.copy2(pkg / name, snapshot / 'cuda' / name)
    shutil.copy2(dispatcher, snapshot / 'triton/main.py')
    hashes = {str(path.relative_to(snapshot)): digest(path) for path in snapshot.rglob('*') if path.is_file()}
    (snapshot / 'snapshot.json').write_text(json.dumps(dict(promotion=job, sha256=hashes), indent=2) + '\n')
    report.update(state='complete', snapshot=str(snapshot), snapshot_files=len(hashes))
    save()
    print('PROMOTION_COMPLETE_FWD', job, flush=True)
except BaseException as error:
    for name in names:
        if name in existing:
            shutil.copy2(backup / name, pkg / name)
        else:
            (pkg / name).unlink(missing_ok=True)
    report.update(state='rolled_back', error=repr(error))
    save()
    raise
