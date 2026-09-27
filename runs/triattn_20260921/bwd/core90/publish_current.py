"""Refresh current pointers only after installed-path reports are complete."""
import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--job', type=int, required=True)
a = ap.parse_args()
d = json.loads((root / 'installed_results.json').read_text())
assert d['final_job'] == a.job
assert len(d['cases']) == 6 and not d['sol90_achieved']
for length in (384, 768, 1024):
    assert json.loads((root / ('attribution-%d-L%d.json' % (a.job, length))).read_text())['complete']
b = [100 * (1 - 1 / r['backward']['speedup']) for r in d['cases']]
fb = [100 * (1 - 1 / r['forward_backward']['speedup']) for r in d['cases']]
bias = d['installation']['bias_manifest.json']['artifact']
dq = d['installation']['dq_manifest.json']['artifact']
detail = ('Qualified `%s` dK/dV + bias and `%s` dQ pass installed-path job%d, '
          'including all input/parameter gradients. Complete backward improves%.2f–%.2f%% '
          'and F+B%.2f–%.2f%% versus job16663. Attribution%d accounts for every16/17 '
          'backward kernels. SOL90 remains unmet. All new experiments use '
          '`%s / normal_h100`.') % (bias, dq, a.job, min(b), max(b), min(fb), max(fb), a.job, d.get('node','node02'))
files = [
    (root.parents[1] / 'HANDOFF.md', 'bwd/core90/README.md'),
    (root.parent / 'README.md', 'core90/README.md'),
    (root.parent / 'SOL90_PROGRESS.md', 'core90/README.md'),
    (root.parent / 'FUSION_PLAN.md', 'core90/README.md'),
]
for path, link in files:
    lines = [line for line in path.read_text().splitlines() if not line.startswith('> **Queued backward follow-up')]
    first = next(i for i, line in enumerate(lines) if line.startswith('> **'))
    lines[first] = '> **Current backward installation, 2026-09-25:** [%s](%s). %s' % ('Installed report', link, detail)
    path.write_text('\n'.join(lines) + '\n')
continuation = root / 'CONTINUATION17127.md'
if continuation.exists():
    lines = continuation.read_text().splitlines()
    lines = [('> **Status:** Installed and verified by job%d; see [current installed report](README.md). ' % a.job +
              'The earlier queue snapshot below is historical.') if line.startswith('> **Status:**') else line for line in lines]
    continuation.write_text('\n'.join(lines) + '\n')
print('PUBLISHED_CURRENT', a.job)
