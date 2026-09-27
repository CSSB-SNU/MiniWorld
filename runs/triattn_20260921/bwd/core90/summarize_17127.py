"""Record this continuation separately from the qualified job17069 installation."""
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
trials = [
    (17127, 'bias', 'rs8_q16_two'),
    (17128, 'bias', 'rs8_q16_four'),
    (17129, 'bias', 'rs8_q16_coop'),
    (17133, 'bias', 'rs8_staged_parallel'),
    (17138, 'dq', 'rs_query_group2'),
    (17139, 'dq', 'rs_query_group4'),
    (17146, 'bias', 'rs8_three'),
    (17147, 'bias', 'rs8_three_staged'),
    (17148, 'dq', 'rs_query_coop2'),
    (17149, 'dq', 'rs_query_coop4'),
    (17151, 'bias', 'rs8_async_bias64'),
    (17160, 'bias', 'rs8_three_p56c152'),
    (17167, 'bias', 'rs8_async_q4'),
    (17168, 'bias', 'rs8_async_q8'),
]
results = []
for job, kind, artifact in trials:
    path = root.parent / ('bias_fusion' if kind == 'bias' else 'dq') / artifact
    ready = path / 'build-ready.json'
    digests = json.loads(ready.read_text()) if ready.exists() else {}
    for filename, digest in digests.items():
        assert hashlib.sha256((path / filename).read_bytes()).hexdigest() == digest
    row = dict(job=job, kind=kind, artifact=artifact, source_digests=digests, lengths={})
    log = root / ('build-pilot-%d.log' % job)
    if log.exists():
        row['ptxas'] = re.findall(r'(?:\d+ bytes stack frame[^\n]*|Used \d+ registers[^\n]*)', log.read_text())
    for length in (64, 384, 768, 1024):
        report = root / ('pilot-%d-L%d.json' % (job, length))
        if not report.exists():
            continue
        d = json.loads(report.read_text())
        assert d['source_digests'] == digests
        assert d['complete']
        row['baseline_binary'] = d['baseline']
        row['baseline_binary_sha256'] = hashlib.sha256(Path(d['baseline']).read_bytes()).hexdigest()
        r = d['records'][0]
        row['lengths'][str(length)] = {k: r[k] for k in ('bitwise', 'relative_l2', 'baseline_ms', 'candidate_ms', 'speedup')}
        row['lengths'][str(length)]['latency_change_pct'] = (1 / r['speedup'] - 1) * 100
    row['pilot_complete'] = len(row['lengths']) == 4
    row['qualified'] = False
    results.append(row)
(root / 'continuation17127-results.json').write_text(json.dumps(results, indent=2) + '\n')
for r in results:
    print(r['job'], r['artifact'], ' '.join('%s:%+.2f%%' % (length, d['latency_change_pct']) for length, d in sorted(r['lengths'].items(), key=lambda x: int(x[0]))))
