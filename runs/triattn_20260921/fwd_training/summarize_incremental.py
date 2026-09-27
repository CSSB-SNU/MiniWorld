"""Document a completed incremental promotion without replacing the old evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics

ap = argparse.ArgumentParser()
ap.add_argument('--job', required=True, type=int)
a = ap.parse_args()
r = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
promotion = read(r / f'promotion-{a.job}.json')
assert promotion['state'] == 'complete'
pkg = r.parents[1] / 'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda'
manifest = read(pkg / 'fwd_manifest.json')
assert manifest == promotion['manifest']
for name, sha in manifest['files'].items():
    assert hashlib.sha256((pkg / name).read_bytes()).hexdigest() == sha
assert hashlib.sha256((pkg / manifest['dispatcher']['path']).read_bytes()).hexdigest() == manifest['dispatcher']['sha256']
snapshot = r / f'checkpoint{a.job}'
for name, sha in read(snapshot / 'snapshot.json')['sha256'].items():
    assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == sha


def profile(job):
    units, values = list(csv.DictReader((r / f'source-profile-{job}.csv').open()))
    keys = ['gpu__time_duration.sum', 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
            'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
            'lts__throughput.avg.pct_of_peak_sustained_elapsed',
            'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
            'sm__warps_active.avg.pct_of_peak_sustained_active', 'launch__registers_per_thread',
            'launch__shared_mem_per_block', 'launch__occupancy_limit_shared_mem']
    result = {k: dict(value=float(values[k]), unit=units[k]) for k in keys}
    t = result['gpu__time_duration.sum']
    result['time_ms'] = t['value'] * {'usecond': .001, 'msecond': 1., 'nsecond': .000001,
                                    'us': .001, 'µs': .001, 'μs': .001, 'ms': 1., 'ns': .000001}[t['unit']]
    return result


def interval(ratios):
    rng = random.Random(92517345)
    samples = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    return [100 * (1 - 1 / samples[250]), 100 * (1 - 1 / samples[9749])]


measurements = []
table = []
core_table = []
for length in (384, 768, 1024):
    data = read(r / f'installed-bench-{a.job}-L{length}.json')
    assert data['complete'] and data['installed_dispatch']
    assert data['baseline_artifact'] == 'cooperative_q2'
    for row in data['records']:
        row['reduction_pct'] = 100 * (1 - 1 / row['speedup'])
        row['paired_bootstrap_median_reduction_95ci_pct'] = interval(row['paired_ratios'])
    measurements.append(data)
    for ending in (False, True):
        f, fb = [next(v for v in data['records'] if v['ending'] == ending and v['kind'] == kind)
                 for kind in ('forward', 'forward_backward')]
        table.append('| %d | %s | %.4f → %.4f | %.2f%% | %.4f → %.4f | %.2f%% |' %
                     (length, 'ending' if ending else 'starting', f['baseline_us']/1000,
                      f['candidate_us']/1000, f['reduction_pct'], fb['baseline_us']/1000,
                      fb['candidate_us']/1000, fb['reduction_pct']))
    c = read(r / f'core-{promotion["evidence"]["bench"]}-L{length}.json')['records'][0]
    core_table.append('| %d | %.4f → %.4f | %.2f%% |' %
                      (length, c['baseline_us']/1000, c['candidate_us']/1000, 100*(1-1/c['speedup'])))

before = profile(17321)
after = profile(promotion['evidence']['profile'])
checks = read(r / f'installed-{a.job}.json')
amp = read(r / f'amp-{a.job}.json')
assert checks['complete'] and amp['complete']
result = dict(promotion=promotion, measurements=measurements, installed_checks=checks, amp=amp,
              before_profile=before, after_profile=after, sol90_achieved=False,
              node='node02', qos='normal_h100')
for source, archive in (('README.md', 'INITIAL_17310.md'),
                        ('installed-results.json', 'installed-results-17310.json')):
    if not (r / archive).exists():
        shutil.copy2(r / source, r / archive)
(r / 'installed-results.json').write_text(json.dumps(result, indent=2) + '\n')
text = f'''# Installed training attention forward

2026-09-25. **Native CUDA/TMA `cooperative_q1_s2` is installed and verified**,
promotion {a.job}. This is an additional improvement over the already installed
CUDA `cooperative_q2` from 17310. H100, B1/H4/D32 BF16 projection layout,
L384/768/1024, both directions. All experiments used node02 / normal_h100.
**SOL90 remains unmet.**

## Actual installed full workload

Frozen CUDA 17310 versus the newly installed path, same process and inputs,
32 alternating AB/BA rounds × 40 CUDA graph replays per arm. Nonzero parameters,
all input/parameter gradients. Times are arm medians; reductions use paired
median ratios. Every recorded full-workload comparison has zero output/gradient
error against CUDA 17310. This table is not a comparison with the old Triton path.

| L | Direction | Full FWD ms | Reduction | F+B ms | Reduction |
|---|---|---:|---:|---:|---:|
''' + '\n'.join(table) + f'''

Full backward is unchanged. Raw rounds and paired bootstrap median intervals
are in `installed-results.json`; intervals describe sampling uncertainty within
these runs, not variation across nodes or workloads. The original 17310-versus-
Triton results remain in [INITIAL_17310.md](INITIAL_17310.md).

## Selected change

One 128-thread CTA computes a 64×64 attention tile stream. A single FP32 score
fragment replaces the two-fragment pipeline. K/V/bias TMA buffering falls from
three stages to two. QK completion retires the previous PV, then all four warps
synchronize before thread zero reuses the prior stage for the next TMA loads.
The cooperating warpgroup consumes those tiles. Resident Q is reused for the
final TMA output store. No preparation kernel or score/probability HBM buffer.

Registers fall from 116 to 80 per thread, kernel shared storage from 54,272 to
37,888 bytes, with zero spills. Six CTAs fit per SM instead of four. Stable
online softmax, unrounded FP32 probability sums, BF16 PV operands/output and
FP32 base-2 LSE retain the original arithmetic. Production O/LSE match CUDA 17310
bitwise in all six core fixtures at each production length.

The selected build still has PTXAS C7515 serialization warnings. Removing a
warning alone did not improve speed: the three-stage single-score control
removed C7514 but was slightly slower. This selection is based on measured
complete-workload gains, not a claim of fully overlapping WGMMA operations.

## Core and profile

Core-only job {promotion['evidence']['bench']}, randomized inputs and mixed mask,
12 AB/BA rounds × 20 replays, separate from the complete-module timings:

| L | CUDA 17310 → selected ms | Paired reduction |
|---|---:|---:|
''' + '\n'.join(core_table) + f'''

NCU source profiles 17321 → {promotion['evidence']['profile']}, L768:
{before['time_ms']:.6f} → {after['time_ms']:.6f} ms. The selected kernel reaches
SM {after['sm__throughput.avg.pct_of_peak_sustained_elapsed']['value']:.2f}%,
L2 {after['lts__throughput.avg.pct_of_peak_sustained_elapsed']['value']:.2f}%,
HBM {after['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed']['value']:.2f}%,
tensor pipe {after['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']['value']:.2f}%,
and active-warp occupancy {after['sm__warps_active.avg.pct_of_peak_sustained_active']['value']:.2f}%.
These single-kernel profiles are not paired end-to-end benchmarks or SOL90.

## Validation and rejected experiments

- Qualification 17336: eight independent FP64 core gradient fixtures and 18
  complete-module checks covering both directions, masks, dropout, optimizer
  updates, fullgraph and frozen parameters. All pass.
- Memcheck/racecheck/synccheck pass at L64/256, all six fixtures; full-shape
  L768 mixed and fully masked cases pass in 17338. No errors or hazards.
- Installed {a.job}: fresh-process cold fullgraph F+B, schema/fake/AOT opcheck,
  actual six module dispatches, layout/dtype/opt-out guards, supported BF16 AMP,
  and the installed paired benchmarks above pass.
- The six backward manifests and every dependency remain unchanged from 17211.
  Dispatcher and Python guard logic also remain unchanged from 17310.

| Candidate | Result |
|---|---|
| `cooperative_earlyqk`, 17325 | Earlier QK launch retained C7514; all production shapes slower. |
| `cooperative_q1`, 17328 | No serialization warning; 96 registers, four CTAs; slightly slower. |
| `cooperative_unroll`, 17329 | Removes warning but spills; L768 core 1.108 → 1.402 ms. |
| `cooperative_q1_s2`, 17333/17344 | Selected; six CTAs, all complete workload cases improve. |
| `cooperative_q1_m128`, 17334 | Two query warpgroups share K/V; no consistent complete-workload gain. |
| `cooperative_bounds`, 17340 | Range-certified softmax passes O/LSE fixtures; including preparation, slower. Not fully qualified. |
| `cooperative_s2_fence`, 17341 | Extra accumulator fence compiles to identical 1,952 SASS instruction lines; no distinct improvement. |
| `cooperative_m128_s2`, 17343 | K/V sharing plus two stages improves over 17310, less than selected in measured overall results. Not promoted. |

## Reproduction and rollback evidence

- Source: [cooperative_q1_s2/fused.cu](cooperative_q1_s2/fused.cu).
- Promotion: [promotion-{a.job}.json](promotion-{a.job}.json).
- Complete frozen native package: [checkpoint{a.job}/snapshot.json](checkpoint{a.job}/snapshot.json).
- Original CUDA baseline remains in `cooperative_q2` and `checkpoint17310`.
- `module_check.py --baseline-artifact cooperative_q2 --installed` compares
  a new installed path with the frozen CUDA baseline; plain `--installed`
  still compares with the old Triton opt-out path. No implicit baseline mixing.
- `MINIWORLD_TRIATTN_TRAINING_FWD=0` retains the established opt-out behavior.
'''
(r / 'README.md').write_text(text)
handoff = r.parent / 'HANDOFF.md'
original = handoff.read_text()
entry = f'''> **Further training FWD improvement, 2026-09-25:** [Current CUDA results](fwd_training/README.md) records installed `cooperative_q1_s2`, promotion{a.job}. One score fragment and two TMA stages permit six resident CTAs/SM, with 80 registers and zero spills. Additional full FWD and F+B gains are measured against frozen CUDA17310, with every output/gradient bitwise equal. FP64/graph/sanitizer/fresh-installed/BF16 AMP gates pass. Backward17211 and the dispatcher are unchanged. L768 NCU: {after['time_ms']:.6f}ms, SM73.77%, L277.58%; SOL90 remains unmet. Full evidence and all rejected candidates are recorded in the report.\n\n'''
anchor = '# TriangleAttention kernel optimisation — handoff\n\n'
assert original.startswith(anchor)
if f'promotion{a.job}' not in original:
    handoff.write_text(anchor + entry + original[len(anchor):])
print('SUMMARY_COMPLETE', a.job)
