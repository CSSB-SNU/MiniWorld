"""Record all controls; publish the selected package only after promotion finishes."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--promotion', type=int, required=True)
a = ap.parse_args()
r = Path(__file__).resolve().parent

def read(name):
    return json.loads((r / name).read_text())

def paired(row):
    values = np.asarray(row['paired_ratios'])
    rng = np.random.default_rng(17989)
    bounds = np.quantile(np.median(values[rng.integers(0, len(values), (10000, len(values)))], axis=1), [.025, .975])
    return dict(row, reduction_pct=100 * (1 - 1 / row['speedup']), paired_ratio_ci95=bounds.tolist())

def profile(artifact, job=17990):
    source = 'qg-profile-%d-%s.csv' % (job, artifact)
    rows = list(csv.DictReader((r / source).open()))
    units = rows[0]
    keys = ('dram__bytes_read.sum', 'dram__bytes_write.sum', 'gpu__time_duration.sum',
            'lts__t_sectors_op_read.sum', 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
            'lts__throughput.avg.pct_of_peak_sustained_elapsed', 'launch__registers_per_thread')
    kernels = [dict(kernel=x['Kernel Name'], metrics={k: float(x[k]) for k in keys}) for x in rows[1:]]
    return dict(source=source, units={k: units[k] for k in keys}, kernels=kernels,
                totals={k: sum(x['metrics'][k] for x in kernels) for k in keys[:4]})

promotion = read('qg-promotion-%d.json' % a.promotion)
assert promotion['state'] == 'complete', promotion
report = dict(promotion=promotion, baseline_promotion=17774, sol90_achieved=False,
              experiments=[], profiles={k: profile(k) for k in ('baseline', 'qg_scoped')},
              installed_full_module=[], qualification=dict(staged=17994, extra=17995, sanitize=17996),
              invalid_builds={17960: 'host weight name collision', 17961: 'cancelled generator offset bug',
                              17963: 'host weight name collision'},
              cancelled_dependents=[17964, 17967])
report['full_qkv_profiles']={k: profile(k,18004) for k in ('baseline','qkv_five')}
for artifact, build, bench in [('qg_serial3',17968,17972), ('qg_preload3',17969,17973),
                               ('qg_scoped',17980,17983), ('qkv_five',17991,17997)]:
    entry = dict(artifact=artifact, build_job=build, bench_job=bench,
                 build=read(artifact+'/build-ready.json'), projection_attention=[])
    for L in (384,768,1024):
        source='qg-core-%d-L%d.json'%(bench,L)
        d=read(source)
        assert d['complete']
        assert all(max(x['projection_rel']) == 0 and x['combined_o_rel'] == 0 and
                   x['combined_lse_max'] == 0 for x in d['records'])
        entry['projection_attention'].append(dict(paired(d['records'][0]), length=L, source=source,
                                                  six_cases_bitwise=True))
    report['experiments'].append(entry)
for L in (384,768,1024):
    source='qg-installed-bench-%d-L%d.json'%(a.promotion,L)
    d=read(source)
    assert d['complete'] and d['installed_dispatch']
    report['installed_full_module'].extend(dict(paired(x), length=L, source=source) for x in d['records'])
(r/'qg-fusion-results.json').write_text(json.dumps(report,indent=2)+'\n')

rows=[]
for L in (384,768,1024):
    for ending in (False,True):
        d={x['kind']:x for x in report['installed_full_module'] if x['length']==L and x['ending']==ending}
        f,b=d['forward'],d['forward_backward']
        rows.append('| %d | %s | %.4f -> %.4f | %.2f%% | %.4f -> %.4f | %.2f%% |' %
                    (L,'ending' if ending else 'starting',f['baseline_us']/1000,f['candidate_us']/1000,
                     f['reduction_pct'],b['baseline_us']/1000,b['candidate_us']/1000,b['reduction_pct']))
table='| L | Direction | Full FWD ms | Reduction | F+B ms | Reduction |\n|---|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n'
controls=[]
for e in report['experiments']:
    controls.append('| %s | %s |' % (e['artifact'], ' | '.join('%.2f%%'%x['reduction_pct'] for x in e['projection_attention'])))
text='''# Q and gate projection fused with attention

2026-09-26. Installed native CUDA/TMA `qg_scoped`, promotion%d, against frozen
Q-only checkpoint17774. H100 B1/H4/D32/C128 BF16, L384/768/1024, both directions.
Experiments ran on node02 / normal_h100. **SOL90 remains unmet.**

## Installed full workload

64 alternating AB/BA rounds x40 CUDA Graph replays per arm, including all input
and parameter gradients. Times are arm medians; reductions use paired median
ratios. All outputs and gradients are bitwise equal. Every FWD bootstrap95%%
lower bound exceeds1 and every F+B median improves. Intervals describe within-run
uncertainty, not cross-node guarantees. Raw rounds are in
[qg-fusion-results.json](qg-fusion-results.json).

%s
## Implementation and HBM traffic

Each64-query CTA loads normalized Z once, loads Q/gate weights together, projects
and saves gate, then projects and saves Q. Q stays in shared memory for attention.
After all projection readers retire, the32KiB Z/weight scratch becomes K/V/bias
pipeline storage. The gate save finishes reading the shared output tile before
Q overwrites it. Removing a redundant post-Q barrier and shortening projection
live ranges improved the scoped control. K/V still use two GEMMs.

128 threads,37,888B shared memory,96 registers,five CTA resource capacity per SM.
Compiler reports8B stack,4B spill stores and8B spill loads; these are not zero.
This is cooperative phase scheduling, with two-stage TMA attention loading;
there is no dedicated projection producer running concurrently with consumers.

NCU17990, complete **Q/K/V/gate projections + attention** atL768:
HBM reads911.06 ->760.15MB, writes703.89 ->723.44MB, summed kernel duration
1.33136 ->1.26912ms. Do not compare these four-projection totals directly with
the earlier three-projection report. The fused kernel alone grows1.02218 ->1.06320ms
because it now computes gate; the separate gate GEMM disappears. Its SM throughput
is68.34%% and L2 throughput86.76%%. Neither is whole-module SOL90 evidence.

## Controls and full-QKV follow-up

All four projections plus attention, compared with17774. Positive means reduced
time. Core controls use12 paired rounds x20 replays; full-module selection above
uses64x40. Every control passes six mask/logit fixtures at all three lengths.

| Artifact | L384 reduction | L768 reduction | L1024 reduction |
|---|---:|---:|---:|
%s

The full-QKV control keeps K/V resident and uses five projection/attention
warpgroups instead of six. It removes the six-warpgroup register spill, but
still needs up to218,112B shared memory and one CTA per SM. It computes gate
with a separate GEMM to preserve the same four-projection comparison boundary.
AtL768,12 query tiles require three rounds across five warpgroups; only two
warpgroups have work in the last round. Six warpgroups had two full rounds.
It is an isolated performance control, not a promoted implementation; it has
not received full-module gradient/sanitizer qualification.

Fresh NCU18004, four-projection boundary against17774: HBM reads910.97 ->406.49MB,
writes703.30 ->728.62MB, duration1.33053 ->1.58957ms. Total traffic falls29.68%%,
yet execution grows19.47%%. Its fused kernel SM throughput is52.31%% and L2
throughput30.66%%. Eliminating spills alone did not solve this schedule.

Full QKV fusion does remove HBM reads. Earlier resident-QKV measurements reduced
three-projection+attention reads911 ->245MB but increased time1.27 ->1.91ms.
Training still requires saved Q/K/V for backward. Lowering shared-memory occupancy
cost and projection waits remains necessary before selecting full residency.

## Validation and installed dispatch

- Staged17994:18 full-module fixtures plus8 independent FP64 gradient fixtures.
- Staged17995:cold fullgraph, schema/fake/AOT, AMP, partial/frozen weights and guards.
- Sanitizer17996:memcheck/racecheck/synccheck atL128/L768, mixed/all-masked;zero errors/hazards.
- Promotion%d:fresh installed dispatch/opt-out checks and repeated full FWD/BWD/F+B.
- Existing CUDA sources and binaries are byte-identical to17774. Only the front
  dispatcher and two existing manifests change; the new implementation has its
  own `qg_fwd_manifest.json`. The checkpoint captures all installed files.

`MINIWORLD_TRIATTN_QG_FWD=0` restores Q-only17774. `MINIWORLD_TRIATTN_Q_FWD=0`
restores separate projections. The broader training-forward opt-out still works.
All backward saves, numerical operations and native backward binaries are preserved.

[Promotion](qg-promotion-%d.json), [snapshot](checkpoint%d/snapshot.json),
[CUDA source](qg_scoped/fused.cu), [previous Q fusion](IMPROVEMENT_17774.md),
[earlier full-QKV experiments](QKV_FUSION_RESULTS.md).
''' % (a.promotion,table,'\n'.join(controls),a.promotion,a.promotion,a.promotion)
for old,new in [('README.md','IMPROVEMENT_17774.md'),('installed-results.json','installed-results-17774.json')]:
    assert not (r/new).exists(),new
    shutil.copy2(r/old,r/new)
(r/'QG_FUSION_RESULTS.md').write_text(text)
(r/'README.md').write_text(text)
installed=dict(report, measurements=[read('qg-installed-bench-%d-L%d.json'%(a.promotion,L)) for L in (384,768,1024)],
               installed_checks=read('qg-installed-verify-%d.json'%a.promotion),node='node02',qos='normal_h100')
(r/'installed-results.json').write_text(json.dumps(installed,indent=2)+'\n')
h=r.parent/'HANDOFF.md'
note=('> **Q + gate projection + attention installed, 2026-09-26:** promotion%d, `qg_scoped`. '
      '[Current results](fwd_training/QG_FUSION_RESULTS.md) compare actual installed full FWD/BWD/F+B '
      'with frozen17774 atL384/768/1024, both directions. All output/gradients bitwise; FP64, graph, '
      'AMP and three sanitizers pass. Full-QKV five-warpgroup control also measured. '
      'Existing backward binaries unchanged. SOL90 remains unmet.\n\n')%a.promotion
s=h.read_text();header='# TriangleAttention kernel optimisation — handoff\n\n'
assert header in s
h.write_text(s.replace(header,header+note,1))
print(table)
