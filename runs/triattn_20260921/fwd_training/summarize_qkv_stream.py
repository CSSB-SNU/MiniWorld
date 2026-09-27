"""Collect immutable full-QKV experiment evidence; never changes dispatch."""
import csv
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
ENGINE = R.parents[1] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention'
CONTROLS = [
    ('qkv_stream_c4_static', 18063, 18071),
    ('qkv_stream_c2_late88', 18065, 18072),
    ('qkv_stream_wide2b', 18089, 18095),
    ('qkv_stream_wide4b', 18090, 18096),
    ('qkv_stream_wide6b', 18091, 18097),
    ('qkv_compact', 18094, 18101),
    ('qkv_compact_async4', 18105, 18111),
    ('qkv_stream_slots3_c2', 18106, 18112),
    ('qkv_stream_slots3_c4', 18107, 18113),
    ('qkv_compact_qresident', 18119, 18123),
    ('qkv_compact_serial', 18120, 18124),
    ('qkv_compact_remat', 18140, 18155),
]


def read(name):
    return json.loads((R / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing(record):
    return dict(
        baseline_us=record['baseline_us'], candidate_us=record['candidate_us'],
        paired_speedup=record['speedup'],
        time_increase_pct=100 * (1 / record['speedup'] - 1),
        paired_ratios=record['paired_ratios'],
    )


def profile(job, artifact):
    name = 'stream-profile-%s-%s.csv' % (job, artifact)
    rows = list(csv.DictReader((R / name).open()))
    units, values = rows[0], rows[1:]
    metrics = ['dram__bytes_read.sum', 'dram__bytes_write.sum',
               'gpu__time_duration.sum',
               'sm__throughput.avg.pct_of_peak_sustained_elapsed',
               'lts__throughput.avg.pct_of_peak_sustained_elapsed',
               'sm__warps_active.avg.pct_of_peak_sustained_active',
               'launch__shared_mem_per_block_dynamic']
    assert units[metrics[0]] == units[metrics[1]] == 'Mbyte'
    assert units[metrics[2]] == 'ms'
    return dict(
        artifact=artifact, job=job, evidence=name,
        read_mb=sum(float(x[metrics[0]]) for x in values),
        write_mb=sum(float(x[metrics[1]]) for x in values),
        duration_ms=sum(float(x[metrics[2]]) for x in values),
        units={k: units[k] for k in metrics},
        kernels=[dict(name=x['Kernel Name'], metrics={k: float(x[k]) for k in metrics})
                 for x in values],
    )


report = dict(
    baseline='checkpoint18006 / qg_scoped', selected='checkpoint18006 / qg_scoped',
    production_changed=False, sol90_achieved=False,
    shape='B1 H4 D32 C128 BF16; H100; node02 / normal_h100',
    native_boundary='all four Q/K/V/gate projections plus attention; six backward-save outputs',
    timing_method='median of paired baseline/candidate ratios; 12x20 native, 16x20 module; AB/BA order',
    controls=[], full_module=[], qualification=[], profiles=[],
)
for artifact, build_job, bench_job in CONTROLS:
    build = read(artifact + '/build-ready.json')
    for name, expected in build['sha256'].items():
        assert digest(R / artifact / name) == expected, (artifact, name)
    control = dict(artifact=artifact, build_job=build_job, bench_job=bench_job,
                   build=build, native_fixtures=0, timings={})
    for length in [64, 128, 384, 768, 1024]:
        name = 'stream-core-%s-L%s.json' % (bench_job, length)
        data = read(name)
        assert data['complete'] and data['artifact'] == artifact
        assert len(data['records']) == 6
        for rec in data['records']:
            assert max(rec['projection_rel']) == 0
            assert rec['combined_o_rel'] == rec['combined_lse_max'] == 0
            assert rec['attention_only_o_rel'] == rec['attention_only_lse_max'] == 0
        control['native_fixtures'] += 6
        control['timings'][str(length)] = dict(evidence=name, **timing(data['records'][0]))
    report['controls'].append(control)

for artifact, bench, qualify, sanitize in [
    ('qkv_compact_async4', 18125, 18126, 18127),
    ('qkv_compact_remat', 18159, 18160, 18161),
]:
    for length in [384, 768, 1024]:
        name = 'stream-module-%s-L%s.json' % (bench, length)
        data = read(name)
        assert data['complete'] and data['artifact'] == artifact
        for rec in data['records']:
            assert max(rec['errors']) == 0
            report['full_module'].append(dict(
                artifact=artifact, job=bench, length=length,
                direction='ending' if rec['ending'] else 'starting',
                kind=rec['kind'], evidence=name, **timing(rec)))
    name = 'stream-module-%s-L384.json' % qualify
    data = read(name)
    assert data['complete'] and data['artifact'] == artifact and len(data['records']) == 26
    logname = 'stream-sanitize-%s.log' % sanitize
    log = (R / logname).read_text()
    assert log.count('ERROR SUMMARY: 0 errors') == 4
    assert log.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)') == 2
    for length in [128, 768]:
        for tool in ['memcheck', 'racecheck', 'synccheck']:
            data = read('stream-%s-%s-L%s.json' % (tool, sanitize, length))
            assert data['complete'] and data['artifact'] == artifact
    report['qualification'].append(dict(
        artifact=artifact, qualify_job=qualify, evidence=name, records=26,
        sanitizer_job=sanitize, sanitizer_log=logname, sanitizer_lengths=[128, 768],
        sanitizer_errors=0, sanitizer_hazards=0,
    ))

for job, artifact in [(18076, 'qkv_stream_c4_static'),
                      (18128, 'qkv_compact_async4'), (18162, 'qkv_compact_remat')]:
    report['profiles'].append(dict(baseline=profile(job, 'baseline'),
                                   candidate=profile(job, artifact)))
report['phase_probe'] = dict(evidence='stream-phase-18129.json', **read('stream-phase-18129.json'))
snapshot = read('checkpoint18006/snapshot.json')['sha256']
differences = [name for name, expected in snapshot.items() if digest(ENGINE / name) != expected]
assert not differences, differences
report['installed_verification'] = dict(snapshot='checkpoint18006/snapshot.json',
                                        files=len(snapshot), differing_files=differences)
report['rejected_before_benchmark'] = [
    dict(artifacts=['qkv_stream_c2', 'qkv_stream_c4', 'qkv_stream_c2_late'],
         reason='C2 redistribution requests 32768 registers from an allocated 30720 pool; invalid',
         cancelled_jobs=[18060, 18061]),
    dict(artifacts=['qkv_stream_wide2', 'qkv_stream_wide4', 'qkv_stream_wide6'],
         reason='incorrect N64/K128 weight swizzle; projection correctness failed',
         failed_pilots=[18079, 18080, 18085], cancelled_dependencies=[18086, 18087]),
    dict(reason='generator failed before sources existed; cancelled and regenerated',
         cancelled_jobs=[18116, 18117]),
]
report['complete'] = True
(R / 'qkv-stream-results.json').write_text(json.dumps(report, indent=2) + '\n')

lines = ['# Full Q/K/V/gate fusion follow-up', '',
         '2026-09-26. **No new candidate selected. Installed checkpoint18006 (`qg_scoped`) remains unchanged. SOL90 is unmet.**', '',
         'Native CUDA/TMA, B1/H4/D32/C128 BF16 on node02 / normal_h100. '
         'All twelve valid candidates fuse Q/K/V/gate projections and attention into one kernel, '
         'preserving all six outputs needed by the existing backward. '
         'The comparison baseline already fuses Q and gate with attention; K/V use two GEMMs.', '',
         '## Four projections plus attention', '',
         'Speedup is the median paired baseline/candidate time ratio; greater than1 is faster. '
         'Each measurement uses12 AB/BA rounds with20 CUDA Graph replays. '
         'These are the combined projection/attention boundary, not attention-only or full-module FWD.', '',
         '| Candidate | Build / bench jobs | L384 speedup | L768 speedup | L1024 speedup |',
         '|---|---|---:|---:|---:|']
for c in report['controls']:
    lines.append('| %s | %s / %s | %s |' % (
        c['artifact'], c['build_job'], c['bench_job'],
        ' | '.join('%.4fx' % c['timings'][str(length)]['paired_speedup'] for length in [384, 768, 1024])))
lines += ['', 'L64/128 also pass and sometimes improve; they are not the target lengths used for selection. '
          'Separate arm medians and every paired ratio are retained in [machine-readable results](qkv-stream-results.json). '
          'The median of ratios need not equal the ratio of arm medians.', '',
          '## Final candidate: complete module', '',
          '`qkv_compact_remat` recomputes epilogue coordinates after attention. '
          'PTXAS reports zero spill stores/loads for all compiled configurations, down from28B/28B '
          'for the large compact_async4 configurations. It still loses at the target lengths.', '',
          'Module job18159 uses16 paired rounds x20 replays, both directions, '
          'with bitwise output and all input/parameter gradient comparisons. '
          'Positive percentages below mean more time than installed18006.', '',
          '| L | Direction | Full FWD time change | BWD time change | F+B time change |',
          '|---|---|---:|---:|---:|']
for length in [384, 768, 1024]:
    for direction in ['starting', 'ending']:
        records = [x for x in report['full_module'] if x['artifact'] == 'qkv_compact_remat'
                   and x['length'] == length and x['direction'] == direction]
        # Preserve the harness order: forward, backward, forward_backward.
        assert len(records) == 3
        lines.append('| %s | %s | %s |' % (length, direction,
            ' | '.join('%+.2f%%' % x['time_increase_pct'] for x in records)))
lines += ['', 'The earlier compact_async4 module result (job18125) is also retained in the JSON. '
          'No backward algorithm changed. Small BWD/F+B differences do not establish a consistent gain.', '',
          '## L768 HBM evidence', '',
          'Each row sums the same four-projection-plus-attention region. NCU timing is profiler timing, '
          'not the paired graph timing above. Baselines are freshly measured within each profile job.', '',
          '| Job / candidate | HBM read MB, baseline -> candidate | HBM write MB, baseline -> candidate | Duration ms, baseline -> candidate |',
          '|---|---:|---:|---:|']
for p in report['profiles']:
    b, c = p['baseline'], p['candidate']
    lines.append('| %s / %s | %.2f -> %.2f | %.2f -> %.2f | %.5f -> %.5f |' %
                 (c['job'], c['artifact'], b['read_mb'], c['read_mb'], b['write_mb'], c['write_mb'],
                  b['duration_ms'], c['duration_ms']))
lines += ['', 'The158MB streaming read result and the faster compact timings belong to different candidates. '
          'They must not be combined into a fictitious result. Training still needs globally saved Q/K/V/gate; '
          'fusion removes their forward rereads, not those required backward-save writes.', '',
          'Final remat reduces reads66.10% and total HBM traffic32.32% in job18162, but '
          'the profiled region takes5.63% longer. Its SM throughput is59.77%, L2 throughput35.81%, '
          'and active-warps metric36.61%; none establishes SOL90. The NCU raw export emits a '
          'nonfatal embedded-Python site/encoding traceback after each completed report. '
          'Both native runs complete, the .ncu-rep files exist, and CSV exports contain the expected '
          'three baseline kernels and one candidate kernel with valid metrics.', '',
          '## Implementation and bottleneck evidence', '',
          '- Streaming: one producer warpgroup projects64-key K/V tiles;2/4 consumers retain Q and share a '
          'two- or three-slot K/V ring. Only query group0 writes K/V saves; other groups recompute them. '
          'QK completion retires the preceding PV before consumers release its slot. '
          'The N64 producer computes K and V together with double-buffered Z loads.',
          '- Compact: K/V stay in shared memory. A configurable subset of six warpgroups projects '
          'Q+gate then K+V; all six perform attention. Weight/Z scratch is reused, and projection-save waits '
          'are deferred where lifetimes permit. AtL768, compact_async4 uses230400B dynamic shared memory '
          'and one CTA per SM, versus37888B and a five-CTA resource ceiling for installed QG. '
          'Residency and projection phase costs remain despite reduced HBM traffic.',
          '- Q-resident and serial-weight controls trade additional shared memory against Q rereads '
          'and repeated Z loads; neither wins. The rematerialized epilogue eliminates the local-memory '
          'coordinate spills without changing projection/softmax order.',
          '- PTXAS still emits C7515 WGMMA serialization warnings in the compact builds, including '
          '[18140](stream-build-18140.log). Zero spill does not imply a fully overlapped WGMMA pipeline. '
          'The current evidence does not isolate the exact cost of compiler serialization.',
          '- Job18129 instruments CTA cycles around initialization, projection completion and attention '
          'completion. Mean projection fractions are36.26%,20.89%,20.29% forL384/768/1024. '
          'The instrumented binary has extra register/spill cost. These are CTA-cycle fractions, '
          'not exact whole-GPU time attribution or SOL evidence.', '',
          'Asynchronous save scheduling distinguishes shared-source retirement from global-destination '
          'completion: `cp.async.bulk.wait_group.read` covers source reads; default wait covers destination '
          'writes as well. See [NVIDIA PTX semantics](https://docs.nvidia.com/cuda/archive/12.6.2/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group). '
          'Q reloads occur only after destination completion and the required CTA synchronization.', '',
          '## Validation and rejected experiments', '',
          '- All12 valid native candidates:30 fixtures each (L64/128/384/768/1024, mixed/dense/one-key/'
          'all-masked/late-live/large-logits). Q/K/V/gate, O and LSE match the frozen baseline bitwise; '
          'independent sampled FP64 references and changed-input CUDA Graph replay pass.',
          '- Compact_async4: full-module bench18125, qualification18126 (18 module fixtures plus8 '
          'independent FP64 gradient fixtures), sanitizer18127.',
          '- Compact_remat: full-module bench18159, qualification18160 (same26 fixtures), sanitizer18161. '
          'Both candidates pass memcheck/racecheck/synccheck atL128/L768, mixed/all-masked, '
          'with zero errors or hazards. No production installation or fresh-installed qualification is claimed.',
          '- The first C2 register redistribution required32768 registers but had30720 allocated, '
          'and could deadlock. Jobs18060/18061 were cancelled. Later88-register consumers plus64-register '
          'producer fit the pool. Every valid streaming benchmark runs the SASS register-pool audit first.',
          '- Initial combined-N64 weight loading used the wrong swizzle (18079/18080/18085). '
          'It failed correctness and was replaced by fourM32/K64 TMA boxes. Dependent18086/18087 '
          'were cancelled. None of those measurements is accepted as performance evidence.',
          '- Source-generation failures18116/18117 were cancelled before the corrected builds. '
          '18056/18057 built binaries but their initial wrapper failed to find barecuobjdump; '
          'all later wrappers use the project environment. Unrelated training jobs were preserved.',
          '- All44 installed files still match checkpoint18006 SHA256 values. '
          'The experimental six-output wrapper uses the existing backward and remains outside production dispatch.', '',
          '[Design and ownership](STREAMING_QKV.md), [final CUDA source](qkv_compact_remat/fused.cu), '
          '[all hashes, timings and evidence paths](qkv-stream-results.json), '
          '[installed QG results](QG_FUSION_RESULTS.md).', '']
(R / 'QKV_STREAM_RESULTS.md').write_text('\n'.join(lines))
print('Wrote QKV_STREAM_RESULTS.md and qkv-stream-results.json; installed files verified:', len(snapshot))
