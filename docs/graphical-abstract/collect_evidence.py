"""Refresh the draft's small evidence bundle. No imports or execution of kernels."""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path, default=ROOT.parent / 'miniworld-engine')
    args = parser.parse_args()
    sources, records = {}, []

    def source(name, path, kind, description):
        raw = path.read_bytes()
        sources[name] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                             kind=kind, description=description)
        return raw.decode()

    def add(op, shape, length, phase, base, ours, src, baseline, status='historical', note=''):
        assert base > 0 and ours > 0
        records.append(dict(op=op, shape=shape, length=length, phase=phase,
                            baseline_ms=base, ours_ms=ours, speedup=base / ours,
                            sources=src, baseline=baseline, status=status, note=note))

    trimul = json.loads(source('T', ROOT / 'runs/trimul_d256_bwd_sol90_stage2_20260923/qualified_speedups.json',
                              'timing JSON', 'Historical Triton versus our selected TriMul checkpoints'))
    for row in trimul['rows']:
        for phase, timing in row['timings'].items():
            add('TriMul', f'D{row["D"]}', row['L'],
                {'forward': 'train_fwd', 'backward': 'bwd', 'full': 'fb'}[phase],
                timing['triton_us'] / 1000, timing['ours_us'] / 1000, ['T'],
                'Existing Triton', 'research checkpoint' if row['D'] != 128 else 'historical checkpoint',
                f"job {row['job']}; {row['source']}; wide BWD is not promoted to shared dispatch")
    for length in (384, 768):
        key = f'X{length}'
        row = json.loads(source(key, args.engine / f'experiments/transition_fused/records/wired-L{length}.json',
                                'timing JSON', 'Wired D128 residual Transition module, forward + backward'))
        add('Transition', 'D128; expansion 4', length, 'fb',
            row['time_us']['triton_residual']['median'] / 1000,
            row['time_us']['fused_sm90a']['median'] / 1000, [key], 'Triton residual path')

    norm = []
    for key, filename in [('N0', 'bench-baseline.json'), ('N1', 'bench-candidate.json')]:
        norm.append(json.loads(source(key, ROOT / 'runs/norm_cuda_20260926/triton_fix' / filename,
                                     'timing JSON', '2026-09-26 portable Triton normalization comparison')))
    index = {(r['op'], r['M'], r['D'], r['N']): r for r in norm[1]}
    for old in norm[0]:
        new = index[(old['op'], old['M'], old['D'], old['N'])]
        name = {'rms': 'RMSNorm', 'ln': 'LayerNorm', 'linear': 'LNLinear'}[old['op']]
        length = {384**2: 384, 768**2: 768}.get(old['M'])
        shape = f'D{old["D"]}' + (f' to {old["N"]}' if name == 'LNLinear' else '')
        for phase, field in [('train_fwd', 'train_fwd'), ('fb', 'train'), ('inference', 'inference')]:
            add(name, shape, length, phase, old[field]['median_ms'], new[field]['median_ms'], ['N0', 'N1'],
                'Pre-change portable Triton', note=f'M={old["M"]}; not a native CUDA speedup; no BWD inferred by subtraction')

    text = source('O', ROOT / 'runs/msa_bench_20260921/engopmtime-15640.log', 'rounded timing log',
                  'Engine OPM module timing, S1024 L384; both save/recompute variants retained')
    for base, save, ours in re.findall(r'own ([\d.]+) ms .*?fused\[save_o=(\d)\] ([\d.]+) ms', text):
        add('OPM', 'S1024; save O' if save == '1' else 'S1024; recompute O', 384, 'fb',
            float(base), float(ours), ['O'], 'Own engine path (job 15640)',
            note='Input log rounded to 0.001 ms; save/recompute is a time-memory tradeoff')
    text = source('P', ROOT / 'runs/msa_bench_20260921/engtime-15737.log', 'rounded timing log',
                  'Engine PWA module timing, S1024 L384')
    base, ours = re.search(r'own ([\d.]+) ms .*?fused training path ([\d.]+) ms', text).groups()
    add('PWA', 'S1024; D_msa64 / D_pair128', 384, 'fb', float(base), float(ours), ['P'],
        'Own engine path (job 15737)', note='Input log rounded to 0.001 ms')

    code = {
        'C1': 'src/miniworld_engine/kernels/trimul_inproj/cuda/h100_inference.py',
        'C2': 'src/miniworld_engine/kernels/trimul_inproj/cuda/h100_training.py',
        'C3': 'src/miniworld_engine/kernels/trimul_inproj/cuda/h100_sources/inference/tmn_kernels.cuh',
        'C4': 'src/miniworld_engine/kernels/transition/cuda/fused_sm90a.py',
        'C5': 'src/miniworld_engine/integrations/opm_train.py',
        'C6': 'src/miniworld_engine/integrations/pwa_train.py',
        'C7': 'src/miniworld_engine/kernels/layernorm_linear/cute/gemm_layernorm_linear_fused.py',
        'C8': 'src/miniworld_engine/modules/triangle_attention/module.py',
        'C9': 'src/miniworld_engine/kernels/triangle_attention/cuda/training_forward.py',
    }
    for key, rel in code.items():
        source(key, args.engine / rel, 'implementation', rel)
    bundle = dict(title='MiniWorld: fuse around data reuse', date='2026-09-27',
                  engine_commit='afd54410a0bf59a204b6ca00af38909a684cf50f', gpu_run_performed=False,
                  scope='Historical H100 records, not a matched v2.1 release sweep. Baselines differ by family.',
                  sources=sources, records=records,
                  missing=['TriangleAttention: no matched operator-level comparison selected for this draft',
                           'Whole-model speedup: not inferred from operator ratios',
                           'Inference coverage: not substituted with training forward measurements'])
    (HERE / 'evidence.json').write_text(json.dumps(bundle, indent=2) + '\n')
    print(f'Collected {len(records)} measurements and {len(sources)} source hashes')


if __name__ == '__main__':
    main()
