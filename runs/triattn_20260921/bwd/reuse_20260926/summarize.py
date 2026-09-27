"""Collect the bounded backward pass; never infer a promotion from a fast pilot."""
from pathlib import Path
import hashlib
import json
import random
import statistics

root = Path(__file__).resolve().parent
pkg = root.parent.parents[1] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention'
baseline = json.loads((root / 'baseline.json').read_text())
for name, digest in baseline.items():
    assert hashlib.sha256((pkg / name).read_bytes()).hexdigest() == digest, name


def summarize_timing(row):
    result = {k: v for k, v in row.items() if k not in ('paired_ratios', 'rounds_ms', 'rounds_us')}
    result['time_reduction_percent'] = 100 * (1 - 1 / row['speedup'])
    ratios = row['paired_ratios']
    rng = random.Random(92626)
    boots = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000))
    result['time_reduction_95ci_percent'] = [100 * (1 - 1 / boots[i]) for i in (125, 4874)]
    return result


report = dict(scope='Training backward against installed checkpoint18246; inference FWD unchanged',
              unchanged_package_files=len(baseline), native=[], full=[], attribution=[], qualification={})
for artifact, kind, job in [('reuse_q', 'dq', 19243), ('reuse_qdo', 'dq', 19243),
                            ('rs8_prob_overlap', 'bias_fusion', 19244)]:
    folder = root.parent / kind / artifact
    build = json.loads((folder / 'build-ready.json').read_text())
    for name, digest in build.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    entry = dict(artifact=artifact, job=job, build=build,
                 resources=(folder / 'resources.txt').read_text(), records=[])
    for length in (64, 384, 768, 1024):
        source = f'native-{artifact}-{length}-{job}.json'
        data = json.loads((root / source).read_text())
        assert data['complete'] and all(data['records'][0]['bitwise'])
        entry['records'].append(dict(source=source, length=length,
                                     **summarize_timing(data['records'][0])))
    report['native'].append(entry)
for artifact, job in [('rs8_prob_overlap', 19251), ('reuse_q', 19253),
                       ('reuse_qdo', 19257), ('reuse_qdo', 19268)]:
    for length in (384, 768, 1024):
        source = f'combined-{length}-{job}.json'
        path = root / source
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows = []
        for name, row in data['measurements'].items():
            assert all(row['gradient_bitwise'])
            rows.append(dict(regime=name, **summarize_timing(row)))
        report['full'].append(dict(source=source, artifact=artifact, job=job, length=length,
                                   complete=data.get('complete', False), records=rows))
for length in (384, 768, 1024):
    source = f'attribution-{length}-19245.json'
    data = json.loads((root / source).read_text())
    assert data['complete']
    for row in data['records']:
        report['attribution'].append(dict(source=source, length=length, ending=row['ending'],
            graph_us=row['graph_us'], profile_sum_us=row['kernel_sum_us'],
            stages=[dict(name=x['name'], us=x['us_per_backward'], percent=x['kernel_time_percent'])
                    for x in row['kernels']]))
log = (root / 'qualify-19266.log').read_text()
report['qualification'] = dict(job=19266,
    zero_mem_sync_summaries=log.count('ERROR SUMMARY: 0 errors'),
    zero_race_summaries=log.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'),
    fp64_files=len(list(root.glob('fp64-reuse_qdo-*-19266.json'))),
    module_files=len(list(root.glob('module-dq-reuse_qdo-*-19266.json'))),
    graph_files=len(list(root.glob('graph-reuse_qdo-*-19266.json'))))
for path in root.glob('graph-reuse_qdo-*-19266.json'):
    data = json.loads(path.read_text())
    assert data['complete']
    assert sum(row.get('bitwise', False) for row in data['records']) == 6
q = report['qualification']
report['complete'] = (q['zero_mem_sync_summaries'] == 16 and q['zero_race_summaries'] == 8
                      and q['fp64_files'] == 8 and q['module_files'] == q['graph_files'] == 3
                      and len(report['full']) == 12 and all(x['complete'] for x in report['full']))
report['production_changed'] = False
report['explicit_entry'] = []
for length in (384, 768, 1024):
    path = root / f'entry-{length}-19277.json'
    if path.exists():
        data = json.loads(path.read_text())
        assert data['complete'] and data['explicit_entry']
        assert sum(row.get('bitwise', False) for row in data['records']) == 6
        report['explicit_entry'].append(dict(source=path.name, length=length,
            changed_graph_bitwise_cases=6, kernel_attribution=True))
report['complete'] = report['complete'] and len(report['explicit_entry']) == 3
(root / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print('SUMMARY', report['complete'], q)
for entry in report['full']:
    if entry['job'] == 19268:
        for row in entry['records']:
            print(entry['length'], row['regime'], row['baseline_us']/1000,
                  row['candidate_us']/1000, row['time_reduction_percent'], row['time_reduction_95ci_percent'])
