"""Correct the baseline identity and report the actual original Triton comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--job', type=int, required=True)
parser.add_argument('--large-job', type=int, required=True)
args = parser.parse_args()
jobs = {384: args.job, 768: args.job, 1024: args.large_job}
reports = {L: json.loads((ROOT / f'triton-compare-{L}-{job}.json').read_text()) for L, job in jobs.items()}
assert all(r['complete'] for r in reports.values())
benchmark_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ('compare_triton.py', 'compare_triton_three.py')}
loader_hash = hashlib.sha256((ROOT / 'original_triton.py').read_bytes()).hexdigest()
assert all(r['benchmark_sha256'] in benchmark_hashes.values() and r['original_loader_sha256'] == loader_hash for r in reports.values())
result = dict(jobs=jobs, original_triton=reports[384]['original_triton'], rows=[],
              benchmark_source_sha256=benchmark_hashes, original_loader_sha256=loader_hash,
              max_relative_l2=0, candidate_bitwise_installed_cuda_at_384_768=True)
result['shared_helper_audit'] = json.loads((ROOT / 'shared_helper_audit.json').read_text())
lines = ['# Corrected comparison against the original Triton engine', '',
         '**Correction:** The earlier report called our already optimized CUDA checkpoint18246 '
         'the "existing engine". The user meant the original Triton engine. That baseline label '
         'and the implied comparison against original Triton were wrong. The earlier data only '
         'compared two versions of our CUDA implementation.', '',
         f'Jobs{args.job} at384/768 and{args.large_job} at1024. Implementations share the same '
         'process/device per length, node02 '
         'H100 80GB, BF16 input/projection weights, FP32 LN affine, B=1, C=128, H=4, D=32, '
         'dropout=0, every seventh key masked. Same nonzero weights/input/dy and all input plus '
         'eight parameter gradients; no optimizer or gradient accumulation. CUDA Graph timings '
         'use24 rounds x10 replays and20 initial warmups. The384/768 run includes the installed '
         'CUDA checkpoint as a fourth control and rotates all24 orders;1024 uses three arms '
         'and all six orders to keep simultaneously resident CUDA Graphs within80GB. '
         'Compilation, autotuning and post-timing profiling are excluded.', '',
         '- **PyTorch:** original dense einsum/softmax module path; no torch.compile or SDPA.',
         '- **Original Triton engine:** module and attention source from engine git commit '
         f"`{reports[384]['original_triton']['engine_commit']}`. The module is byte-identical; "
         'only two custom-op registration names in the core are prefixed to avoid reusing the '
         'installed CUDA dispatcher. A process-local proxy routes the frozen module to the '
         'frozen core while retaining lazy imports for the unchanged original surroundings.',
         '- **Our installed CUDA:** cumulative optimized training checkpoint18246; the API option '
         '`implementation="triton"` is still used but actually dispatches our native CUDA kernels.',
         '- **Our latest CUDA candidate:** the same installed training forward and backward '
         'except dQ uses the separately qualified `reuse_qdo` extension.',
         '- **Anthropic optimized release:** no backward for the previously compared optimized '
         'forward rows, so BWD and F+B remain unsupported; stock-library fallback is not substituted.', '']
for regime, title in [('backward', 'Complete BWD (ms)'), ('forward_backward', 'Complete F+B (ms)')]:
    lines += [f'## {title}', '',
              '| L | Direction | PyTorch | Original Triton engine | Our latest CUDA | Triton / latest CUDA |',
              '|---:|---|---:|---:|---:|---:|']
    for L, report in reports.items():
        for ending in (False, True):
            cell = report['measurements'][f'{regime}_e{int(ending)}']
            names = ('pytorch', 'original_triton', 'candidate_reuse_qdo')
            times = {name: cell['arms'][name]['median_ms'] for name in names}
            speedup = statistics.median(cell['paired_original_triton_over_candidate'])
            direction = 'ending' if ending else 'starting'
            result['rows'].append(dict(length=L, direction=direction, regime=regime, job=jobs[L], times_ms=times,
                                       triton_over_latest_cuda=speedup,
                                       pytorch_over_latest_cuda=statistics.median(cell['paired_pytorch_over_candidate'])))
            result['max_relative_l2'] = max(result['max_relative_l2'],
                *(v['relative_l2'] for arm in cell['arms'].values() for v in arm['versus_pytorch']))
            if 'installed_cuda18246' in cell['arms']:
                assert cell['candidate_bitwise_engine'] and cell['forward_candidate_bitwise_engine']
            else:
                assert cell['candidate_bitwise_engine'] is None
                assert max(v['relative_l2'] for v in cell['original_triton_vs_candidate']) < .015
            original_kernels = cell['arms']['original_triton']['profile']['kernels']
            assert any('layer_norm_bwd_dx_fused' in name for name in original_kernels)
            lines.append(f'| {L} | {direction} | ' + ' | '.join(f'{times[name]:.4f}' for name in names)
                         + f' | {speedup:.2f}x |')
    lines += ['']
lines += ['## Verification', '',
          'All12 length/direction/regime cells require finite output/gradients and a predeclared '
          f"cross-implementation relative L2 limit0.03; observed maximum is{result['max_relative_l2']:.6f}. "
          'The two CUDA controls at384/768 are bitwise equal for output and every gradient. '
          'At1024, original Triton versus latest CUDA also passes the direct relative L2 limit0.015. Triton '
          'gradients are checked numerically rather than assumed bitwise equal to CUDA.', '',
          'Profiles confirm actual original `_attn_fwd`, `_attn_bwd_preprocess`, `_attn_bwd_dkdv`, '
          'and `_attn_bwd_dq` launches. The original arm must have no native dQ, dK/dV, '
          'projection/LN-residual fusion or shared-input weight-gradient launch from our CUDA '
          'campaign. Its LN/gate profiles are also retained; ordinary cuBLAS projection GEMMs '
          'remain part of the original module.', '',
          'The shared primitives, bias-only dispatch and gate helpers are unchanged against '
          'the git baseline. The changed standalone CUDA LN files are inactive for BF16 '
          'activations with FP32 affine parameters; original LN uses the Triton path, verified '
          'by actual kernel names. All48 installed attention source/binary files are verified '
          'before/after each length. No serving source/dispatch is changed.', '',
          'The first restoration attempt19327 failed before timing because a copied namespace '
          'omitted lazy kernel exports. The loader now forwards lazy attribute lookup. No '
          'failed-cell timing is included. Job19328 completed384/768 and1024 starting, then ran '
          'out of memory holding four complete graph arms at1024 ending. The complete1024 '
          'three-arm rerun replaces both directions in this table; partial19328 data is retained '
          'separately. This is a block performance comparison, not SOL.', '',
          f'Reproduce: `sbatch compare_triton.sbatch` for384/768; use '
          '`--export=ALL,LENGTHS=1024,BENCH_SCRIPT=compare_triton_three.py` for1024. Then '
          f'`python3 summarize_triton.py --job {args.job} --large-job {args.large_job}`. '
          'Original commit/source hashes, adapted source hashes and actual profiles are preserved '
          'in `original_triton_source.json` and `triton-compare-L-JOB.json`.', '']
(ROOT / 'triton-results.json').write_text(json.dumps(result, indent=2) + '\n')
(ROOT / 'TRITON_COMPARISON.md').write_text('\n'.join(lines))
print('\n'.join(lines))
