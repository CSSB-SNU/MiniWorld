"""Group measured graph nodes, retaining every kernel in the accounting."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--job', type=int, required=True)
a = ap.parse_args()
root = Path(__file__).resolve().parent
labels = ['dK/dV + bias partial', 'dQ', 'Bias final reduction', 'Weight gradients (six)',
          'Projection/LN/residual', 'Gate/output/delta', 'Ending dY layout copy',
          'Bias mask/layout', 'LN parameter reductions']
weight_labels = ['W_out', 'W_q/k/v/gate (shared input)', 'W_bias']
summary = dict(job=a.job, cases=[])
for length in [384, 768, 1024]:
    source = root / f'attribution-{a.job}-L{length}.json'
    data = json.loads(source.read_text())
    assert data['complete']
    for record in data['records']:
        count = record['kernels_per_backward']
        replays = record['profile_replays']
        sequence = record['kernel_sequence']
        assert len(sequence) == count * replays
        first_names = [v['name'] for v in sequence[:count]]
        groups = {label: 0. for label in labels}
        weights = {label: 0. for label in weight_labels}
        nodes = []
        for replay in range(replays):
            seq = sequence[replay * count:(replay + 1) * count]
            assert [v['name'] for v in seq] == first_names
            gemm = -1
            for idx, node in enumerate(seq):
                name = node['name']
                if 'grouped_dkdv' in name:
                    label = labels[0]
                elif 'dq_tma' in name:
                    label = labels[1]
                elif 'reduce_bias' in name:
                    label = labels[2]
                elif name.startswith('nvjet_') or ('cutlass::Kernel2' in name and 'gemm' in name):
                    label = labels[3]
                    gemm += 1
                    assert gemm < 2
                    # Source executes output dW before Front.backward, then
                    # Q/K/V/gate/bias in that order. Verify native boundaries.
                    front_before = any('projection_ln_residual_tma' in n for n in first_names[:idx])
                    assert front_before == (gemm > 0)
                    weights[weight_labels[gemm*2]] += node['us'] / replays
                elif 'cublasLt::splitKreduce_kernel' in name:
                    label = labels[3]
                    assert gemm >= 0
                    weights[weight_labels[gemm*2]] += node['us'] / replays
                elif 'grouped_wgrad' in name or 'reduce_wgrad' in name:
                    label = labels[3]
                    weights[weight_labels[1]] += node['us'] / replays
                elif 'projection_ln_residual_tma' in name:
                    label = labels[4]
                elif 'gate_delta_tma' in name:
                    label = labels[5]
                elif record['ending'] and idx == 0 and 'direct_copy_kernel_cuda' in name:
                    label = labels[6]
                elif any(s in name for s in ['masked_fill_kernel', 'direct_copy_kernel_cuda', 'memcpy128', 'Memcpy DtoD']):
                    label = labels[7]
                elif 'reduce_norm' in name or 'finish_norm' in name:
                    label = labels[8]
                else:
                    raise AssertionError(('unattributed kernel', name))
                groups[label] += node['us'] / replays
                if replay == 0:
                    nodes.append(dict(index=idx, category=label, kernel=name))
            assert gemm == 1
        total = sum(groups.values())
        assert abs(total - record['kernel_sum_us']) < 1e-6
        assert abs(sum(weights.values()) - groups[labels[3]]) < 1e-6
        case = dict(length=length, ending=record['ending'], kernel_sum_us=total,
                    graph_us=record['graph_us'], profiled_graph_us=record['profiled_graph_us'],
                    kernels_per_backward=count, groups_us=groups, weights_us=weights,
                    groups_percent={k: 100*v/total for k, v in groups.items()},
                    ordered_kernel_mapping=nodes, source=str(source))
        summary['cases'].append(case)

(root / f'attribution-summary-{a.job}.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = [f'# Installed backward time attribution — job{a.job}', '',
    'H100 80GB, B1 BF16 C128/H4/D32, all input and parameter gradients, dropout0,',
    'every seventh key masked. All five fusion flags are enabled; installed',
    f"{data['manifests']['bias_manifest.json']['artifact']} dK/dV, {data['manifests']['dq_manifest.json']['artifact']} dQ, warp_packed projection/LN,",
    'n64_staged gate, shared_z_tensor Q/K/V/gate weight gradients.', '',
    'All component times come from complete backward CUDA graph replays under',
    'CUPTI, with ten profiler-attached warmup replays discarded and fifteen',
    'measured replays. Percentages use the measured kernel-time sum for that',
    'same profile. No scaling to the independent CUDA-event timing is applied.',
    'CUDA-event totals outside the profiler use the median of nine rounds of',
    'fifteen graph replays. Small differences reflect profiling, clocks and',
    'inter-kernel gaps; this is backward without the forward or optimizer step.', '',
    '| L | Direction | Event total ms | Profile kernel sum ms | Kernels |',
    '|---:|---|---:|---:|---:|']
for c in summary['cases']:
    lines.append(f"| {c['length']} | {'ending' if c['ending'] else 'starting'} | {c['graph_us']/1000:.3f} | {c['kernel_sum_us']/1000:.3f} | {c['kernels_per_backward']} |")
for length in [384, 768, 1024]:
    c0, c1 = [c for c in summary['cases'] if c['length'] == length]
    lines += ['', f'## L{length}', '', '| Stage | Starting ms | Share | Ending ms | Share |',
              '|---|---:|---:|---:|---:|']
    for label in labels:
        lines.append(f"| {label} | {c0['groups_us'][label]/1000:.4f} | {c0['groups_percent'][label]:.2f}% | {c1['groups_us'][label]/1000:.4f} | {c1['groups_percent'][label]:.2f}% |")
    lines += ['', 'Weight-gradient detail, including each GEMM split-K reduction:', '',
              '| Weight | Starting ms | Ending ms |', '|---|---:|---:|']
    for label in weight_labels:
        lines.append(f"| {label} | {c0['weights_us'][label]/1000:.4f} | {c1['weights_us'][label]/1000:.4f} |")
lines += ['', 'The six weight gradients now launch six kernels: shared-input Q/K/V/gate',
          'GEMM plus FP32 split reduction, output GEMM plus reduction, bias GEMM',
          'plus reduction. The complete backward has 16/17 kernels (starting/ending),',
          'unchanged from the baseline of this core pass. The complete sequence and all six',
          'parameter gradients remain accounted for. Ending retains its dY copy.', '',
          f'Raw data: `attribution-{a.job}-L*.json`; Chrome traces:',
          f'`attribution-{a.job}-L*-e*-trace.json`. The summary JSON asserts that',
          'the category totals equal every kernel in the profile.']
(root / 'ATTRIBUTION.md').write_text('\n'.join(lines) + '\n')
for case in summary['cases']:
    print(case['length'], 'ending' if case['ending'] else 'starting',
          round(case['graph_us']/1000, 3), round(case['kernel_sum_us']/1000, 3),
          {k: round(v, 2) for k, v in case['groups_percent'].items()})
