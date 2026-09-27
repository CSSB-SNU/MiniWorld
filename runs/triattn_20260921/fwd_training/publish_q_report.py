"""Publish current results only after the installed comparison completes."""
from pathlib import Path
import json
import shutil

r=Path(__file__).resolve().parent
def read(n):return json.loads((r/n).read_text())
d=read('qkv-fusion-results.json');p=d['promotion']
assert p['state']=='complete' and p['job']=='17774'
for old,new in (('README.md','IMPROVEMENT_17628.md'),('installed-results.json','installed-results-17628.json')):
    assert not (r/new).exists(),new
    shutil.copy2(r/old,r/new)
rows=[]
for L in (384,768,1024):
    for end in (False,True):
        cell={x['kind']:x for x in d['installed_full_module'] if x['length']==L and x['ending']==end}
        f,b=cell['forward'],cell['forward_backward']
        rows.append('| %d | %s | %.4f -> %.4f | %.2f%% | %.4f -> %.4f | %.2f%% |'%(
            L,'ending' if end else 'starting',f['baseline_us']/1000,f['candidate_us']/1000,f['reduction_pct'],
            b['baseline_us']/1000,b['candidate_us']/1000,b['reduction_pct']))
table='''| L | Direction | Full FWD ms | Reduction | F+B ms | Reduction |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'\n'
verify=read('q-installed-verify-17774.json')
installed=dict(promotion=p,measurements=[read('q-installed-bench-17774-L%d.json'%L) for L in (384,768,1024)],
    confirmation=[read('qkv-module-17756-L%d.json'%L) for L in (384,768,1024)],
    installed_checks=verify,amp=[x for x in verify['records'] if x['kind']=='amp'],
    profile=d['profiles']['17767'],sol90_achieved=False,node='node02',qos='normal_h100',
    paired_statistics=d['installed_full_module'])
(r/'installed-results.json').write_text(json.dumps(installed,indent=2)+'\n')
text='''# Installed training attention forward

2026-09-26. **Native CUDA/TMA Q-projection plus attention fusion is installed and
verified**, promotion17774, artifact `q_only_head4`. Scope: H100,
B1/H4/D32/C128 BF16, L384/768/1024, both directions. Experiments used
node02 / normal_h100. **SOL90 remains unmet.**

## Actual installed full-workload comparison

Compare the preserved17628 separate-projection path with the installed Q-fused
path in the same process, using64 alternating AB/BA rounds x40 CUDA-graph
replays per arm. Times below are arm medians; reductions use paired median
ratios. Every output and input/parameter gradient is bitwise equal. Every FWD
bootstrap95% lower bound is positive, and every F+B median improves. Full raw
rounds and intervals are in [installed-results.json](installed-results.json).
These intervals describe within-run uncertainty, not cross-node guarantees.

'''+table+'''
## Selected implementation

The attention CTA projects its own64 query rows with TMA/WGMMA, saves BF16 Q
for backward, then consumes the same Q shared-memory tile. K/V retain their
two GEMMs and use the existing two-stage attention pipeline. Projection scratch
aliases K/V/bias storage after projection completion. Four consecutive head
CTAs reuse normalized input Z. Resource use:128 threads,90 registers,zero spills,
37,888B shared storage,five CTAs/SM. The attempt to force six CTAs spilled and
was slower. This is a cooperative phase schedule, not dedicated producer/
consumer overlap.

A combined front+attention autograd boundary prevents duplicate Q projection
and preserves all backward saves. Existing backward CUDA binaries and numerical
operations remain unchanged; only the front dispatch/autograd wiring changes.
The original separate-projection path remains as `ln_backward.unfused_forward`.

Set `MINIWORLD_TRIATTN_Q_FWD=0` to use that baseline. Unsupported inputs and
disabled required fusions also retain the baseline. The broader
`MINIWORLD_TRIATTN_TRAINING_FWD=0` opt-out still disables the new Q path.
The existing `cooperative_head2` attention core is preserved for these paths.

## Validation and measured limits

- FP64 projections/attention and8 independent FP64 gradient fixtures;
  production shapes on6 mask/logit fixtures; changed-input CUDA-graph replay.
- Staged17762:18 full-module cases including dropout, optimizer updates,
  fullgraph and frozen parameters, plus the8 FP64 gradient fixtures.
- Sanitizer17766:memcheck/racecheck/synccheck atL128/L768;zero errors/hazards.
- Fresh installation17774:cold fullgraph, schema/fake/AOT, dtype/layout/shape/
  opt-out guards, actual native dispatch, BF16 AMP and partial-parameter gradients.
- Actual installed paired FWD/BWD/F+B repeats the complete production workload,
  including all input and parameter gradients. Promotion rolls back on failure.
- NCU17767, complete projections+attention:L768 HBM reads910.94 ->759.91MB,
  writes555.06 ->570.54MB. Total traffic falls about9.25%, while L2 read sectors
  rise259.00 ->274.58million. Fused kernel SM70.51%,L288.05%;SOL90 is not achieved.
- Seven CUDA fusion candidates are documented in
  [QKV_FUSION_RESULTS.md](QKV_FUSION_RESULTS.md). Full-QKV residency reduced reads
  but lost on large shapes because of projection serialization and occupancy.

## Evidence

[Promotion17774](q-promotion-17774.json),
[frozen package](checkpoint17774/snapshot.json),
[selected CUDA source](q_only_head4/fused.cu),
[all fusion experiments](qkv-fusion-results.json),
[fusion design](QKV_ATTENTION_FUSION.md).

The only existing package files changed are `ln_backward.py` and its manifest;
the new Q forward implementation has its own `q_fwd_manifest.json`. All existing
CUDA binaries/sources, the attention dispatcher and the original forward guard
match checkpoint17628. The new snapshot contains40 files.

Prior results remain in [IMPROVEMENT_17628.md](IMPROVEMENT_17628.md),
[IMPROVEMENT_17345.md](IMPROVEMENT_17345.md),
[INITIAL_17310.md](INITIAL_17310.md), and
[PIPELINE_FOLLOWUP.md](PIPELINE_FOLLOWUP.md).
'''
assert p['snapshot_files']==40,p['snapshot_files']
(r/'README.md').write_text(text)
q=r/'QKV_FUSION_RESULTS.md';text=q.read_text()
text=text.replace('''Installation17774 is being verified; until its state becomes `complete`,
checkpoint17628 remains the selected baseline. This report will be updated with
the actual installed measurements, or with the reason for rollback.''',
'''**Installed and verified: promotion17774**, `q_only_head4`. The full17628
baseline remains available through the Q-fusion opt-out. All pre-install and
actual installed selection gates passed.

## Actual installed full-module result

64 alternating AB/BA rounds x40 graph replays per arm, same inputs/process.
Times are arm medians; reductions use paired ratios. All outputs/gradients
are bitwise equal. Full intervals and raw timings are in `qkv-fusion-results.json`.

'''+table)
q.write_text(text)
h=r.parent/'HANDOFF.md';text=h.read_text()
note='''> **Q projection + attention installed, 2026-09-26:** [Current results](fwd_training/README.md) records promotion17774 (`q_only_head4`) against frozen17628. Seven CUDA fusion candidates were tested. Full QKV residency lost on L768/1024; Q-only fusion improves full FWD and F+B at all three lengths and both directions, with every output/gradient bitwise equal. Native CUDA/TMA, four adjacent-head CTAs, shared scratch reuse; all backward binaries remain unchanged. FP64, graph, three sanitizers, cold installed dispatch and AMP gates pass. [All fusion results](fwd_training/QKV_FUSION_RESULTS.md). SOL90 remains unmet.

'''
text=text.replace('# TriangleAttention kernel optimisation — handoff\n\n','# TriangleAttention kernel optimisation — handoff\n\n'+note,1)
text=text.replace('[QKV-attention fusion design](fwd_training/QKV_ATTENTION_FUSION.md) is the new proposed direction; no fused QKV kernel is implemented.',
                  '[QKV-attention fusion design](fwd_training/QKV_ATTENTION_FUSION.md) motivated the later17774 experiments above.')
h.write_text(text)
print(table)
