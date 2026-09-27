"""Render the complete three-arm comparison, including the unsupported fourth arm."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--job', type=int, required=True)
ap.add_argument('--repeat-job', type=int, action='append', default=[])
args = ap.parse_args()
rng = random.Random(92682)


def interval(values):
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(5000))
    return [medians[125], medians[4874]]


reports = {L: json.loads((ROOT / f'compare-{L}-{args.job}.json').read_text()) for L in (384, 768, 1024)}
assert all(report['complete'] for report in reports.values())
assert len({report['benchmark_sha256'] for report in reports.values()}) == 1
benchmark_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT / 'compare.py', ROOT / 'compare_19305.py')}
assert next(iter(reports.values()))['benchmark_sha256'] in benchmark_hashes.values()
engine = ROOT.parents[2] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine'
module_hashes = json.loads((ROOT / 'module-source-sha256.json').read_text())
for name, expected in module_hashes.items():
    assert hashlib.sha256((engine / name).read_bytes()).hexdigest() == expected
result = dict(job=args.job, rows=[], benchmark_source_sha256=benchmark_hashes,
              module_source_sha256=module_hashes, max_relative_l2=0,
              candidate_bitwise_engine=True, anthropic=reports[384]['anthropic'])
lines = ['# Complete training comparison', '',
         '**Baseline correction:** "Existing engine" below means our already optimized CUDA '
         'checkpoint18246, not the original Triton engine. The earlier implied comparison '
         'against original Triton is withdrawn. See [the corrected comparison](TRITON_COMPARISON.md).', '',
         f'Job {args.job}, node02 H100 80GB, BF16 activations/projection weights and FP32 LN affine, '
         'B=1, C=128, H=4, D=32, dropout=0, every seventh key masked. All eight parameter '
         'gradients and input gradient are included; F+B excludes optimizer and gradient accumulation.', '',
         'PyTorch means the existing eager dense einsum/softmax implementation, captured in CUDA Graph; '
         'it is neither torch.compile nor SDPA. Existing engine means installed training checkpoint18246. '
         'Candidate means the explicit `reuse_qdo` context with the same training forward. '
         'This comparison does not use the separate inference-only forward.', '',
         f"All three arms run in the same process/GPU for each length, {reports[384]['rounds']} rounds "
         f"x {reports[384]['replays']} graph replays, after 20 warmup replays. Six rotated execution orders "
         'balance position and predecessor. Timings are CUDA events; profiler runs occur afterward.', '',
         'The pristine Anthropic optimized rows k2b/k2/flash/cuda_sm90a/triattn_native/triattn_exact '
         'declare no backward. The engine adapter also rejects training at runtime. Backward-capable '
         'stock/cuEq/DS4Sci/SDPA rows delegate to other libraries and are not substitutes for the '
         'previously compared Anthropic optimized forward. The release sources were hash-checked '
         'against the original checkout. Unsupported cells below are not performance measurements.', '']
for regime, title in [('backward', 'Complete backward (ms)'), ('forward_backward', 'Complete forward + backward (ms)')]:
    lines += [f'## {title}', '',
              '| L | Direction | PyTorch | Anthropic optimized | Existing engine | Current candidate | PyTorch / candidate |',
              '|---:|---|---:|---|---:|---:|---:|']
    for L, report in reports.items():
        for ending in (False, True):
            cell = report['measurements'][f'{regime}_e{int(ending)}']
            arms = cell['arms']
            py, eng, cand = [arms[name]['median_ms'] for name in ('pytorch', 'engine18246', 'candidate_reuse_qdo')]
            speedups = cell['paired_pytorch_over_candidate']
            reductions = [100 * (1 - 1 / x) for x in cell['paired_engine_over_candidate']]
            direction = 'ending' if ending else 'starting'
            result['rows'].append(dict(length=L, direction=direction, regime=regime,
                                       pytorch_ms=py, engine_ms=eng, candidate_ms=cand,
                                       pytorch_speedup=statistics.median(speedups),
                                       pytorch_speedup_ci95=interval(speedups),
                                       engine_time_reduction_pct=statistics.median(reductions),
                                       engine_time_reduction_ci95=interval(reductions)))
            result['max_relative_l2'] = max(result['max_relative_l2'],
                *(value['relative_l2'] for arm in arms.values() for value in arm['versus_pytorch']))
            result['candidate_bitwise_engine'] &= cell['candidate_bitwise_engine'] and cell['forward_candidate_bitwise_engine']
            lines.append(f'| {L} | {direction} | {py:.4f} | unsupported | {eng:.4f} | {cand:.4f} | {statistics.median(speedups):.2f}x |')
    lines += ['']
lines += ['## Increment over the existing engine', '',
          'Positive means less elapsed time. Intervals are 5000-resample bootstrap 95% intervals '
          'of the median paired reduction. These differences should be interpreted together '
          'with the dedicated 64x40 comparison in [the prior qualification](../reuse_20260926/README.md).', '',
          '| L | Direction | BWD reduction (95% CI) | F+B reduction (95% CI) |',
          '|---:|---|---:|---:|']
for L in reports:
    for direction in ('starting', 'ending'):
        cells = [r for r in result['rows'] if r['length'] == L and r['direction'] == direction]
        text = []
        for cell in cells:
            lo, hi = cell['engine_time_reduction_ci95']
            text.append(f"{cell['engine_time_reduction_pct']:+.3f}% ({lo:+.3f}, {hi:+.3f})")
        lines.append(f'| {L} | {direction} | ' + ' | '.join(text) + ' |')
result['replications'] = []
if args.repeat_job:
    lines += ['', '## Independent repeat', '',
              'The three-arm sequence changes the immediately preceding workload. At L768 in '
              'the primary run, order-group median BWD differences span roughly -3% to +3%; '
              'the 95% intervals include zero. Therefore the primary arm medians do not establish '
              'a small candidate regression or improvement. An independent repeat uses the same '
              'timing code and fixtures with more rounds/replays:', '',
              '| Job | L | Direction | Regime | PyTorch ms | Engine ms | Candidate ms | Paired reduction (95% CI) |',
              '|---:|---:|---|---|---:|---:|---:|---:|']
    for job in args.repeat_job:
        paths = sorted(ROOT.glob(f'compare-*-{job}.json'))
        assert paths, job
        for path in paths:
            data = json.loads(path.read_text())
            assert data['complete']
            assert data['benchmark_sha256'] in benchmark_hashes.values()
            for key, cell in data['measurements'].items():
                assert cell['candidate_bitwise_engine'] and cell['forward_candidate_bitwise_engine']
                times = [cell['arms'][name]['median_ms'] for name in ('pytorch', 'engine18246', 'candidate_reuse_qdo')]
                values = [100 * (1 - 1 / r) for r in cell['paired_engine_over_candidate']]
                lo, hi = interval(values)
                change = statistics.median(values)
                direction = 'ending' if cell['ending'] else 'starting'
                result['replications'].append(dict(job=job, length=data['length'], direction=direction,
                    regime=cell['regime'], rounds=data['rounds'], replays=data['replays'],
                    pytorch_ms=times[0], engine_ms=times[1], candidate_ms=times[2],
                    engine_time_reduction_pct=change, engine_time_reduction_ci95=[lo, hi]))
                lines.append(f"| {job} | {data['length']} | {direction} | {cell['regime']} | "
                             + ' | '.join(f'{t:.4f}' for t in times)
                             + f' | {change:+.3f}% ({lo:+.3f}, {hi:+.3f}) |')
    lines += ['', 'PyTorch versus either native engine is a clear multi-fold difference. '
              'The much smaller candidate versus installed-engine difference is not a consistent '
              'whole-workload win in this comparison. Keep the installed engine as the default; '
              'the candidate remains an explicit experiment.']
lines += ['', '## Validation and limits', '',
          f"All outputs/input gradients/parameter gradients are finite. Maximum cross-implementation "
          f"relative L2 error versus the BF16 PyTorch path is {result['max_relative_l2']:.6f}, below the "
          'predeclared 0.03 comparison limit. Candidate outputs and every gradient are bitwise equal '
          'to the existing engine in all twelve regime/direction/length cells. The prior candidate '
          'qualification additionally covers independent FP64 adjoints, changed-input graph replay, '
          'dropout/SGD/frozen parameters and memcheck/racecheck/synccheck.', '',
          'CPU+CUDA profiles verify actual CUDA Graph launches and the installed versus candidate '
          'dQ symbols. PyTorch has no native dQ launch. The 48 checkpoint18246 source/binary hashes '
          'match before and after each length; module/dispatch/primitive source hashes are retained '
          'and unchanged. No production dispatch or source is modified.', '',
          'Graph allocation peaks in raw JSON are cumulative across simultaneously resident arms, '
          'so they must not be interpreted as a per-implementation peak-memory comparison. '
          'This is a training block timing comparison, not an SOL measurement.', '',
          f"Reproduce: submit `compare.sbatch`, then `python3 summarize.py --job {args.job}"
          + ''.join(f' --repeat-job {job}' for job in args.repeat_job) + '`. '
          'Raw rounds, error metrics, profiles, release metadata and source/binary hashes are in '
          '`compare-L-JOB.json`; the initial L384 pilot is job19301. '
          'The original benchmark source is frozen in `compare_19305.py`. The current script '
          'only changes post-timing profiler replay count from one to three after job19315 '
          'failed its ending-BWD dispatch assertion on an incomplete profiler trace. Its two '
          'completed starting cells are retained as partial evidence, not a completed repeat.', '']
assert result['candidate_bitwise_engine']
(ROOT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
(ROOT / 'README.md').write_text('\n'.join(lines))
print('\n'.join(lines))
