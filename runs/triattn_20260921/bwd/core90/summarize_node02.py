"""Summarize the complete node02 continuation against its actual starting state."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
installed = json.loads((root / 'installation.json').read_text())
snapshot = json.loads((root / 'checkpoint17069/snapshot.json').read_text())
promotion = json.loads((root / 'promotion-17211.json').read_text())
assert promotion['state'] == 'complete' and promotion['node'] == 'node02'
assert installed['bias_manifest.json']['artifact'] == 'rs8_async_q4'
report = dict(node='node02', qos='normal_h100', installed_job=17211,
              baseline='checkpoint17069', comparison_job=17212, cases=[], pilots=[])
lines = ['# Completed node02 backward continuation', '',
         'Installed `rs8_async_q4` dK/dV + bias; dQ remains `rs_softmax_overlap`.',
         'All builds, measurements and validation in this continuation used node02',
         'with normal_h100 QoS. Previously queued17175/17167/17168 were moved in',
         'place from node01, preserving their dependencies.', '',
         '## Gain since the start of this continuation', '',
         'Direct comparison17212 against frozen checkpoint17069, all input and',
         'parameter gradients, both directions.24 balanced AB/BA rounds and20',
         'CUDA graph replays. No sum of separate improvements. Times are arm',
         'medians; reductions use median paired ratios.', '',
         '| L | Direction | Before BWD ms | Final BWD ms | BWD reduction | F+B reduction |',
         '|---:|---|---:|---:|---:|---:|']
for length in (384, 768, 1024):
    d = json.loads((root / ('combined-17212-L%d.json' % length)).read_text())
    assert d['complete'] and not d['baseline_from_package']
    assert d['baseline_checkpoint'] == 'checkpoint17069'
    assert d['benchmark_rounds'] == 24 and d['benchmark_replays'] == 20
    for kind in ('bias', 'dq'):
        before = json.loads((root / 'checkpoint17069' / (kind + '_manifest.json')).read_text())
        current = installed[kind + '_manifest.json']
        assert d['binary_digests']['baseline'][kind] == snapshot[before['binary']]
        assert d['binary_digests']['candidate'][kind] == current['files'][current['binary']]
    for ending in (0, 1):
        b = d['measurements']['backward_e%d' % ending]
        f = d['measurements']['forward_backward_e%d' % ending]
        assert all(b['gradient_bitwise']) and all(f['gradient_bitwise'])
        assert b['speedup'] > 1 and f['speedup'] > 1
        br, fr = (100 * (1 - 1 / x['speedup']) for x in (b, f))
        report['cases'].append(dict(length=length, ending=bool(ending), backward=b,
                                    forward_backward=f, bwd_reduction_pct=br, fb_reduction_pct=fr))
        lines.append('| %d | %s | %.4f | %.4f | %.2f%% | %.2f%% |' %
                     (length, 'ending' if ending else 'starting', b['baseline_us']/1000,
                      b['candidate_us']/1000, br, fr))
current = json.loads((root / 'installed_results.json').read_text())
assert current['final_job'] == 17211
report['profiles'] = current['profiles']
report['sol90_achieved'] = current['sol90_achieved']
assert not report['sol90_achieved']
lines += ['', '## Current L768 SOL', '',
          '| Kernel | Time ms | Compute SOL | Memory SOL |', '|---|---:|---:|---:|']
for p in report['profiles']:
    lines.append('| %s | %.4f | %.2f%% | %.2f%% |' %
                 (p['kernel'], p['ms'], p['compute_sol'], p['memory_sol']))
lines += ['', '**Neither target core reaches SOL90.** Bias final reduction was',
          'already above the80% cutoff and remains unchanged.', '',
          '## Implementation and qualification', '',
          '- Two previously idle producer warps reduce the eight dS tiles while',
          '  consumers calculate the next query tile. FP32 association is preserved.',
          '- Four shared Q/dO/stat stages keep TMA farther ahead. Shared memory is',
          ' 175104 bytes; PTXAS reports168 initial registers and zero local spills.',
          '  No additional HBM tensor is introduced; R8 scratch is unchanged.',
          '- Explicit full-reader and ready/empty barriers remain on every reused',
          '  shared stage. Qualification17203 passes independent FP64/masks, exact',
          '  bias cancellation, module10 cases and all six sanitizer runs at64/256.',
          '- Installed job17211 passes all manifests, wgrad opcheck, ten opt-outs,',
          '  cold fullgraph, full all-gradient timing, BF16 AMP/FP16 refusal, and',
          '  exhaustive16/17-kernel attribution. Every gradient in comparison17212',
          '  is bitwise equal to the starting installation.', '',
          '## Rejected controls', '',
          '| Pilot | Candidate | L384 time change | L768 time change | L1024 time change |',
          '|---:|---|---:|---:|---:|']
for job in (17168, 17193, 17194, 17208):
    measurements = []
    for length in (384, 768, 1024):
        d = json.loads((root / ('pilot-%d-L%d.json' % (job, length))).read_text())
        value = d['records'][0]
        assert d['complete'] and all(value['bitwise'])
        measurements.append((1 / value['speedup'] - 1) * 100)
    report['pilots'].append(dict(job=job, artifact=d['artifact'], latency_change_pct=measurements))
    lines.append('| %d | %s | %+.2f%% | %+.2f%% | %+.2f%% |' %
                 tuple([job, d['artifact']] + measurements))
lines += ['', 'These controls compare with installed17175. Depth8 is faster than',
          'that parent, but slower than selected depth4. All three dQ half-P',
          'controls lose at768/1024 despite zero spills and exact pilot outputs;',
          'they are not sanitizer-qualified or installed.', '',
          'The original campaign comparison against16663 is in [README.md](README.md):',
          'full backward21.92-23.23% shorter, F+B14.63-16.00% shorter.',
          '[ATTRIBUTION.md](ATTRIBUTION.md) records every stage of the final installed path.']
(root / 'node02-results.json').write_text(json.dumps(report, indent=2) + '\n')
(root / 'NODE02_CONTINUATION.md').write_text('\n'.join(lines) + '\n')
print('NODE02_REPORT_COMPLETE', min(r['bwd_reduction_pct'] for r in report['cases']),
      max(r['bwd_reduction_pct'] for r in report['cases']))
