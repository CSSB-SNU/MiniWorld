"""Collect the immutable core pilots and qualified selected-entry measurements."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

PILOTS = {
    19031: ['pipeend32_c4'],
    19032: ['pipeend64_c2', 'pipeend64_c4'],
    19062: ['n40_hot4t', 'n40_hot6t'],
    19075: ['n40_qhalf4', 'qhalf4'],
    19076: ['n40_qhalf6'],
    19088: ['pack4', 'pack6'],
    19089: ['n40_pack4', 'n40_pack6'],
    19100: ['split6q', 'split4q'],
    19111: ['qfull4v2'],
    19122: ['qhalf6'],
}
QUALIFICATIONS = {
    19098: ('n40_qhalf4', [128, 1024], [1024]),
    19109: ('qhalf4', [128, 1024], [1024]),
    19134: ('qfull4v2', [128, 1024], [1024]),
    19135: ('qfull4v2', [768], [768]),
}
parser = argparse.ArgumentParser()
parser.add_argument('--selected-job', type=int)
args = parser.parse_args()
root = Path(__file__).resolve().parent


def read(name):
    return json.loads((root / name).read_text())


def timing(row, ci=False):
    result = {k: v for k, v in row.items() if k != 'paired_ratios'}
    result['time_reduction_percent'] = 100 * (1 - 1 / row['speedup'])
    if ci:
        ratios = row['paired_ratios']
        rng = random.Random(92626)
        boots = sorted(statistics.median(rng.choices(ratios, k=len(ratios)))
                       for _ in range(5000))
        result['speedup_median_bootstrap_95ci'] = [boots[125], boots[4874]]
        result['time_reduction_95ci_percent'] = [100 * (1 - 1 / v)
                                                for v in (boots[125], boots[4874])]
    return result


report = dict(scope='Explicit inference experiment; no production or training promotion',
              previous_selection={384: 'h4kv_local1', 768: 'hot6t', 1024: 'hot4t'},
              pilots=[], qualifications=[], finalists=[], profiles=[], selected=[])
report['qfull_native_exact'] = []
for length in (64, 128, 384, 768, 1024):
    source = f'native-pair-qfull4v2-{length}-19134.json'
    data = read(source)
    assert data['complete'] and data['changed_graph_bitwise']
    assert len(data['records']) == 8 and all(row['bitwise'] for row in data['records'])
    report['qfull_native_exact'].append(dict(source=source, length=length,
        bitwise_fixtures=8, changed_graph_bitwise=True))
for job, artifacts in PILOTS.items():
    for artifact in artifacts:
        build = read(f'{artifact}/build-ready.json')
        for name, digest in build['sha256'].items():
            assert hashlib.sha256((root / artifact / name).read_bytes()).hexdigest() == digest
        entry = dict(artifact=artifact, job=job, build=build, measurements=[])
        entry['resources'] = (root / artifact / 'resources.txt').read_text()
        log = (root / f'endwait-pilot-{job}.log').read_text()
        entry['serialization_warnings'] = [line for line in log.splitlines()
                                           if 'C751' in line and f'_{artifact}I' in line]
        for length in (384, 768, 1024):
            source = f'anthropic-{artifact}-{length}-{job}.json'
            data = read(source)
            assert data['complete']
            assert data['out_artifact'] == data['previous_out_artifact'] == 'outproj_c1'
            entry['measurements'].extend(dict(source=source, length=length, **timing(row))
                                         for row in data['records']
                                         if row['kind'] == 'paired_previous')
        report['pilots'].append(entry)

for job, (artifact, san_lengths, module_lengths) in QUALIFICATIONS.items():
    log = (root / f'core-followup-qualify-{job}.log').read_text()
    entry = dict(job=job, artifact=artifact, sanitizer_lengths=san_lengths,
                 zero_error_summaries=log.count('ERROR SUMMARY: 0 errors'),
                 zero_race_summaries=log.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'),
                 module=[])
    entry['sanitizer_complete'] = (entry['zero_error_summaries'] == 6 * len(san_lengths)
                                   and entry['zero_race_summaries'] == 3 * len(san_lengths))
    for length in module_lengths:
        path = root / f'qualified-{artifact}-{length}-{job}.json'
        if path.exists():
            data = json.loads(path.read_text())
            entry['module'].append(dict(length=length, complete=data['complete'],
                records_by_kind=dict(collections.Counter(x['kind'] for x in data['records']))))
    report['qualifications'].append(entry)

for source in ('finalist-qhalf4-n40_qhalf4-19132.json',
               'finalist-qhalf4-qfull4v2-19132.json',
               'finalist-hot6t-qfull4v2-768-19137.json'):
    data = read(source)
    assert data['complete']
    report['finalists'].append(dict(source=source, base=data['base'], artifact=data['artifact'],
        length=data['length'], records=[timing(row, ci=True) for row in data['records']]))
native_source = 'native-pair-qfull4v2-1024-19132.json'
data = read(native_source)
report['native_1024'] = dict(source=native_source, **timing(data['timing'], ci=True))
for artifact in ('hot4t', 'qfull4v2'):
    source = f'coreprofile-{artifact}-1024-19136.csv'
    metrics = ['gpu__time_duration.sum', 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
               'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
               'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
               'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed',
               'dram__bytes_read.sum', 'dram__bytes_write.sum']
    with (root / source).open() as handle:
        rows = list(csv.DictReader(handle))
    units = rows[0]
    report['profiles'].extend(dict(source=source, artifact=artifact,
        metrics={key: dict(value=float(row[key]), unit=units[key]) for key in metrics})
        for row in rows if row['ID'].isdigit())

if args.selected_job:
    report['selected_job'] = args.selected_job
    report['selection'] = {384: 'h4kv_local1', 768: 'hot6t', 1024: 'qfull4v2'}
    for length in (384, 768, 1024):
        source = f'core-selected-anthropic-{length}-{args.selected_job}.json'
        data = read(source)
        assert data['complete'] and data['actual_default_entry'] and data['selected_entry']
        assert data['out_artifact'] == data['previous_out_artifact'] == 'outproj_c1'
        assert report['selection'][length] in data['candidate_builds']
        for ending in (False, True):
            rows = [row for row in data['records'] if row.get('ending') == ending]
            best = min((row for row in rows if row['kind'] == 'paired_module'),
                       key=lambda row: row['baseline_us'])
            previous = next((row for row in rows if row['kind'] == 'paired_previous'), None)
            assert previous is not None or length in (384, 768)
            report['selected'].append(dict(length=length, ending=ending, source=source,
                anthropic=timing(best, ci=True),
                previous=timing(previous, ci=True) if previous else None))
        module = read(f'core-selected-module-{length}-{args.selected_job}.json')
        assert module['complete'] and module['actual_default_entry']
        counts = dict(collections.Counter(x['kind'] for x in module['records']))
        assert counts == dict(correctness=10, fp64_full_module_sample=6, paired_module=2,
                             changed_input_weight_mask_graph=2, fullgraph_compile=2, no_grad=2)
        report.setdefault('selected_module', []).append(dict(length=length,
            records_by_kind=counts))

snapshot = json.loads((root.parent / 'fwd_training/checkpoint18246/snapshot.json').read_text())['sha256']
package = root.parents[1] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention'
assert all(hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
           for name, digest in snapshot.items())
report['unchanged_training_files'] = len(snapshot)
report['candidate_sha256'] = hashlib.sha256((root / 'candidate.py').read_bytes()).hexdigest()
report['complete'] = bool(args.selected_job) and all(x['sanitizer_complete'] for x in report['qualifications'])
(root / 'core-followup-results.json').write_text(json.dumps(report, indent=2) + '\n')
print('CORE_FOLLOWUP_SUMMARY', len(report['pilots']), report['complete'])
for row in report['selected']:
    print(row['length'], row['ending'], row['anthropic']['candidate_us'],
          row['anthropic']['time_reduction_percent'],
          row['previous']['time_reduction_percent'] if row['previous'] else 'unchanged',
          row['previous']['time_reduction_95ci_percent'] if row['previous'] else None)
