"""Promote the qualified async candidate only inside the requested GPU allocation.

Preserve the currently installed package while queued. Recheck every adoption
gate at launch, then validate the actual package and publish measured reports.
Restore the changed files if installed-path verification fails.
"""
import hashlib
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', default='checkpoint17069')
ap.add_argument('--bias', default='rs8_async_bias64')
ap.add_argument('--bias-job', type=int, default=17162)
ap.add_argument('--combined-job', type=int, default=17163)
ap.add_argument('--bias-ncu', type=int, default=17164)
ap.add_argument('--dq-ncu', type=int, default=17063)
config = ap.parse_args()
job = int(os.environ['SLURM_JOB_ID'])
node = os.environ.get('SLURMD_NODENAME')
assert node == 'node02', ('requested node02', node)
schedule = subprocess.check_output(['scontrol', 'show', 'job', str(job), '-o'], text=True)
assert 'QOS=normal_h100 ' in schedule
pkg = root.parents[2] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda'
status = root / ('promotion-%d.json' % job)
args = [sys.executable, '-u', '-B', str(root / 'install_incremental.py'),
        '--checkpoint', config.checkpoint, '--bias', config.bias,
        '--bias-job', str(config.bias_job), '--combined-job', str(config.combined_job)]
def write_status(state, **kwargs):
    status.write_text(json.dumps(dict(job=job, state=state, node=node, qos='normal_h100',
                                     candidate=config.bias, **kwargs), indent=2) + '\n')

subprocess.run(args + ['--dry-run'], check=True)
paths = [pkg / 'bias_fusion.cu', pkg / 'bias_manifest.json', root / 'installation.json',
         root / 'README.md', root / 'installed_results.json', root / 'ATTRIBUTION.md',
         root.parents[1] / 'HANDOFF.md',
         root.parent / 'README.md', root.parent / 'SOL90_PROGRESS.md', root.parent / 'FUSION_PLAN.md']
backups = {p: p.read_bytes() for p in paths}
published = False
def terminated(signum, frame):
    raise RuntimeError('Promotion interrupted by signal %d' % signum)
signal.signal(signal.SIGTERM, terminated)
try:
    write_status('installing')
    subprocess.run(args, check=True)
    installed_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths[:3]}
    write_status('validating_installed', installed_hashes=installed_hashes)
    subprocess.run(['bash', str(root / 'final.sbatch')], check=True)
    with (root / ('attribution-%d.log' % job)).open('w') as output:
        subprocess.run(['bash', str(root / 'attribution.sbatch')], stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    # Both summaries recheck source/binary provenance and complete measurements.
    subprocess.run([sys.executable, '-B', str(root / 'summarize.py'), '--final-job', str(job),
                    '--bias-ncu', str(config.bias_ncu), '--dq-ncu', str(config.dq_ncu), '--node', node], check=True)
    subprocess.run([sys.executable, '-B', str(root / 'summarize_attribution.py'), '--job', str(job)], check=True)
    subprocess.run([sys.executable, '-B', str(root / 'publish_current.py'), '--job', str(job)], check=True)
    published = True
    write_status('complete', installed_hashes=installed_hashes)
    print('PROMOTION_COMPLETE', job, flush=True)
except BaseException as error:
    # No other package files are changed by this installer. An unused copied
    # binary is harmless; restoring manifests restores exact prior dispatch.
    if not published:
        for path, data in backups.items():
            path.write_bytes(data)
    write_status('failed_rolled_back', error=repr(error))
    raise
