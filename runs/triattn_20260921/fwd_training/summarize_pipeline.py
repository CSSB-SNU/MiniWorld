"""Read-only experiment audit against frozen CUDA promotion17345.

Writes a separate experiment report, never the installed manifest or results.
Fails closed if a new candidate improves every full-FWD cell: such a candidate
requires further qualification instead of a no-promotion report.
"""
import hashlib
import csv
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent
ENGINE = R.parents[1] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention'
BASE = 'cooperative_q1_s2'
CANDIDATES = [
    ('producer_warp_s2', 17480, 17486, '32-thread producer, 160-thread CTA, register cap'),
    ('producer_wg_s2', 17481, 17487, 'Full producer warpgroup, 32/96 register redistribution'),
    ('cooperative_s2_late', 17482, 17488, 'Delay cooperative TMA refill until after softmax'),
    ('cooperative_s2_retire', 17499, 17504, 'Retire PV before the next iteration'),
    ('producer_warp_relaxed', 17501, 17505, '32-thread producer with relaxed register cap'),
    ('cooperative_n32', 17508, 17511, '32-key tiles, cooperative CTA'),
    ('producer_warp_n32', 17509, 17512, '32-key tiles, dedicated producer warp'),
    ('cooperative_pshared', 17514, 17521, 'Reuse consumed bias shared storage for P, SS PV'),
    ('producer_warp_pshared', 17515, 17522, 'Bias-to-P shared reuse with a producer warp'),
    ('cooperative_cluster2_v2', 17530, 17534, 'Bias TMA multicast across two outer-row CTAs'),
    ('cooperative_cluster4_v2', 17531, 17535, 'Bias TMA multicast across four outer-row CTAs'),
    ('cooperative_cluster2_early', 17544, 17546, 'Release bias before softmax with four warp arrivals per CTA'),
]


def read(name):
    return json.loads((R / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = read(BASE + '/build-ready.json')
    for f, sha in baseline['sha256'].items():
        assert digest(R / BASE / f) == sha
    snapshot = read('checkpoint17345/snapshot.json')
    for f, sha in snapshot['sha256'].items():
        assert digest(ENGINE / f) == sha, ('installed file changed', f)
    records = []
    for name, build_job, bench_job, design in CANDIDATES:
        info = read(name + '/build-ready.json')
        for f, sha in info['sha256'].items():
            assert digest(R / name / f) == sha
        log = (R / ('build-%s.log' % build_job)).read_text()
        core, module = [], []
        for length in (384, 768, 1024):
            for prefix, dest in (('core', core), ('bench', module)):
                filename = '%s-%s-L%s.json' % (prefix, bench_job, length)
                data = read(filename)
                assert data['complete'] and data['artifact'] == name
                assert data['baseline_artifact'] == BASE
                assert data['baseline_build'] == baseline
                dest.append(dict(path=filename, length=length, records=data['records']))
        fw = [x for d in module for x in d['records'] if x['kind'] == 'forward']
        assert len(fw) == 6
        assert not all(x['speedup'] > 1.015 for x in fw), (name, 'needs further qualification')
        records.append(dict(
            artifact=name, build_job=build_job, benchmark_job=bench_job, design=design,
            build=info, compiler_registers=re.findall(r'Used (\d+) registers', log),
            compiler_spills=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads', log),
            compiler_warnings=sorted(set(re.findall(r'\(C75\d+\)[^\n]+', log))),
            core=core, module=module, promoted=False, full_qualification=False,
        ))
    failures = [
        dict(artifact='producer_wg_r72', jobs=[17493, 17506],
             reason='PTXAS C7602 at first QK WGMMA, initial register target48 below required58'),
        dict(artifact='producer_wg_kstash', jobs=[17494],
             reason='Same first-QK resource failure before the output-to-K stash can help'),
        dict(artifact='cooperative_cluster2', jobs=[17525],
             reason='Ambiguous CUTLASS arrive(0) overload; fixed in immutable v2 artifact'),
        dict(artifact='cooperative_cluster4', jobs=[17526],
             reason='Ambiguous CUTLASS arrive(0) overload; fixed in immutable v2 artifact'),
    ]
    traffic = []
    for name in (BASE, 'cooperative_cluster2_v2', 'cooperative_cluster2_early'):
        prefix = 'traffic-17547-' + name
        info = read(prefix + '.json')
        assert info['complete'] and info['artifact'] == name
        units, values = list(csv.DictReader((R / (prefix + '.csv')).open()))
        metrics = {key: dict(value=values[key], unit=units[key]) for key in values
                   if key.startswith(('lts__', 'dram__', 'gpu__time_duration.sum'))}
        assert all(key in metrics for key in ('lts__t_sectors.sum', 'lts__t_sectors_op_read.sum', 'dram__bytes_read.sum'))
        traffic.append(dict(artifact=name, path=prefix + '.csv', build=info['build'], metrics=metrics))
    result = dict(
        baseline='CUDA checkpoint17345', baseline_build=baseline,
        node='node02', qos='normal_h100', records=records, build_failures=failures,
        cancelled_dependencies=[17502, 17503], installed_snapshot_files_verified=len(snapshot['sha256']),
        traffic_profiles=traffic, promoted=None, sol90=False, complete=True,
    )
    (R / 'pipeline-experiments.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = [
        '# Training forward producer and CTA follow-up', '',
        '2026-09-25. No candidate is promoted. The native CUDA/TMA installation',
        '`cooperative_q1_s2` (promotion17345) is retained; all %d frozen package' % len(snapshot['sha256']),
        'files and the dispatcher match checkpoint17345. SOL90 remains unmet.', '',
        'Experiments run on node02 / normal_h100 against the frozen current CUDA',
        'kernel, not the earlier Triton path or CUDA17310. Core measurements use',
        '12 alternating AB/BA rounds × 20 graph replays; complete module measurements',
        'use 16 rounds × 20 replays, both directions, nonzero output weights, and',
        'all input/parameter gradients. Percentages below are time increases from',
        '`100 * (1 / median(paired baseline/candidate ratios) - 1)`.', '',
        '| Candidate | L384 core | L768 core | L1024 core | Full FWD increase, six cells |',
        '|---|---:|---:|---:|---:|',
    ]
    for d in records:
        increases = [(1 / x['records'][0]['speedup'] - 1) * 100 for x in d['core']]
        fw = [(1 / x['speedup'] - 1) * 100 for m in d['module'] for x in m['records'] if x['kind'] == 'forward']
        lines.append('| `%s` | %+.2f%% | %+.2f%% | %+.2f%% | %+.2f–%+.2f%% |' %
                     (d['artifact'], *increases, min(fw), max(fw)))
    lines += ['', '## Resource and lifetime findings', '',
        '- The spill-free full producer warpgroup has initial64 registers/thread,',
        '  producer32/consumer96 redistribution, and four CTAs/SM in profile17495.',
        '  It removes WGMMA serialization warnings but loses to the six-CTA baseline.',
        '- A compact producer warp with a tight register cap spills. Relaxing the',
        '  cap eliminates spills at90 registers but still regresses measured speed.',
        '- Eager PV retirement removes serialization warnings, but95 registers',
        '  reduce CTA residency and the complete workload is slower.',
        '- Smaller key tiles reduce fragment size but increase iterations and',
        '  synchronization. They pass the sampled FP64 forward cases; changed online',
        '  softmax grouping produces small BF16 differences, not bitwise equality.',
        '- Shared P reuses dead bias storage and adds no HBM intermediate. Extra',
        '  shared-memory operations and synchronization outweigh the register saving.',
        '- Multicast recipients arm their local full barrier before leader TMA.',
        '  The existing all-reader barrier precedes remote empty arrivals, so the',
        '  leader cannot overwrite a bias slot before every CTA has finished it.',
        '  Cluster entry/exit synchronization protects remote shared storage lifetime.',
        '- Warp-level early bias release moves the handoff before softmax. It stays',
        '  spill-free but grows to96 registers/thread and is slower than late release.',
        '', '## Multicast traffic experiment', '',
        'NCU17547 directly counts L768 traffic in the same allocation:',
        'two-CTA multicast reduces L2 read sectors from228,449,858 to185,836,385',
        '(18.65%) and total L2 tag sectors from236,021,731 to216,094,411 (8.44%).',
        'HBM reads stay essentially unchanged at910.85 versus910.74MB. The bias',
        'was already cached in L2; multicast does not remove a material HBM pass.',
        'Profile17541 retains six CTAs/SM and80 registers but increases the barrier',
        'stall-per-issue-active ratio from0.488 to0.820 versus profile17337.',
        'The traffic reduction is real, but complete-workload measurements are slower.',
        'The early-release variant has185–188 million L2 read sectors,96 registers,',
        'and still slower measured execution. NCU export emits a Python site/encoding',
        'diagnostic; all profile jobs exit0 and the report/CSV required metrics exist.',
        '', '## Evidence and limits', '',
        'All completed performance candidates pass six sampled independent FP64',
        'forward fixtures at each production length and changed-input graph replay.',
        'Complete-module benchmarks check gradients and actual native dispatch.',
        'Performance rejection does not imply full sanitizer or strict independent',
        'FP64 gradient qualification; these experimental candidates are not approved',
        'for dispatch/autograd. Failed builds are not performance measurements.', '',
        'The lower-register producer and K-stash controls fail PTXAS C7602 at the',
        'first QK WGMMA (jobs17493/17494; retained PTX probe17506). Dependencies17502/',
        '17503 were cancelled. The first cluster builds17525/17526 had a C++ overload',
        'error, fixed in separate v2 artifacts before benchmarking.', '',
        'See [pipeline-experiments.json](pipeline-experiments.json) for artifact hashes,',
        'all paired core/full-workload rounds, compiler diagnostics and build failures.',
        'The unchanged installed result remains in [README.md](README.md).', '',
        'Harness changes: `check.py` accepts an explicit frozen baseline; native builds',
        'can retain PTX with `KEEP_PTX=1`; `bench_17345.sbatch` fixes the baseline;',
        '`bench_cluster.sbatch` checks two-stage wraparound before production shapes.',
        '`promote_incremental.py` accepts explicit baseline/launch metadata but was',
        'not used for a promotion in this follow-up.', '',
    ]
    (R / 'PIPELINE_FOLLOWUP.md').write_text('\n'.join(lines))
    print('REPORT_COMPLETE', len(records), 'candidates;', len(snapshot['sha256']), 'installed hashes match')


if __name__ == '__main__':
    main()
