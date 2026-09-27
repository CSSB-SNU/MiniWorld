"""Publish verified installed results; keep historical experiments separate."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
pkg=ROOT.parents[1]/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda'
promotion=json.loads((ROOT/'promotion-17310.json').read_text())
amp=json.loads((ROOT/'amp-17316.json').read_text())
assert promotion['state']=='complete' and amp['complete'] and len(amp['records'])==4
installed=json.loads((ROOT/'installed-17310.json').read_text())
assert installed['complete']
manifest=json.loads((pkg/'fwd_manifest.json').read_text())
assert manifest==promotion['manifest']
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for f,h in manifest['files'].items():assert digest(pkg/f)==h
assert digest(pkg/manifest['dispatcher']['path'])==manifest['dispatcher']['sha256']

keys=['gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed',
      'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
      'dram__cycles_active.avg.pct_of_peak_sustained_elapsed','lts__throughput.avg.pct_of_peak_sustained_elapsed',
      'sm__warps_active.avg.pct_of_peak_sustained_active','dram__bytes_read.sum','dram__bytes_write.sum',
      'launch__registers_per_thread','launch__shared_mem_per_block']
units,row=list(csv.DictReader((ROOT/'candidate-profile-17294.csv').open()))
metrics={k:dict(value=float(row[k]),unit=units[k]) for k in keys}
table=[];core_table=[];measurements=[]
for L in (384,768,1024):
    d=json.loads((ROOT/('installed-bench-17310-L%d.json'%L)).read_text())
    assert d['complete'] and d['installed_dispatch']
    measurements.append(d)
    for ending in (False,True):
        f=next(r for r in d['records'] if r['ending']==ending and r['kind']=='forward')
        fb=next(r for r in d['records'] if r['ending']==ending and r['kind']=='forward_backward')
        table.append('| %d | %s | %.4f -> %.4f | %.2f%% | %.4f -> %.4f | %.2f%% |'%(
            L,'ending' if ending else 'starting',f['baseline_us']/1000,f['candidate_us']/1000,
            100*(1-1/f['speedup']),fb['baseline_us']/1000,fb['candidate_us']/1000,100*(1-1/fb['speedup'])))
    d=json.loads((ROOT/('core-17289-L%d.json'%L)).read_text());assert d['complete']
    r=d['records'][0]
    core_table.append('| %d | %.4f | %.4f | %.2f%% |'%(L,r['baseline_us']/1000,r['candidate_us']/1000,100*(1-1/r['speedup'])))
result=dict(promotion=promotion,amp=amp,installed_checks=installed,measurements=measurements,
            profile=metrics,profile_job=17294,sol90_achieved=False,node='node02',qos='normal_h100')
(ROOT/'installed-results.json').write_text(json.dumps(result,indent=2)+'\n')

# Freeze a complete set of all seven native manifests and their dependencies,
# including the forward dispatcher in its correct relative directory.
snapshot=ROOT/'checkpoint17310'
files=set()
for p in pkg.glob('*manifest.json'):
    files.add(p)
    info=json.loads(p.read_text())
    files.update(pkg/f for f in info.get('files',info.get('sha256',{})))
files.add((pkg/manifest['dispatcher']['path']).resolve())
hashes={}
for p in sorted(files):
    rel=p.resolve().relative_to(pkg.parent.resolve())
    dst=snapshot/rel;dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dst);hashes[str(rel)]=digest(dst)
(snapshot/'snapshot.json').write_text(json.dumps(hashes,indent=2)+'\n')

text='''# Installed training attention forward

2026-09-25. **Native CUDA/TMA `cooperative_q2` is installed and verified**, job
17310 (`PROMOTION_COMPLETE_FWD 17310`). B1/H4/D32 BF16 projection layout on H100,
L384/768/1024, both directions. All work ran on node02 / normal_h100.
The six backward artifacts from checkpoint17211 remain unchanged.
**SOL90 is not achieved.**

## Complete workload, actual installed dispatch

Same-process opt-out versus enabled installed forward, 16 alternating AB/BA
rounds, 20 graph replays per arm. Nonzero weights and all parameter/input
gradients. Times are arm medians; reductions use median paired ratios.
The baseline is the training Triton forward immediately before installation,
not the separate historical standalone CUDA forward. New kernels allocate no
score/probability HBM intermediate and require no preparation kernel.

| L | Direction | Full FWD ms, before -> installed | Reduction | F+B ms, before -> installed | Reduction |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

Full backward alone remains effectively unchanged; this update does not claim
a backward-only improvement. F+B includes the new forward and all installed
backward fusions. Files: `installed-bench-17310-L{384,768,1024}.json`.

## Core-only experiment

Job17289, same qualified binary, randomized projection-layout inputs, mixed
mask, 12 AB/BA rounds. These isolated timings are distinct from the complete
installed-module measurements above.

| L | Baseline ms | CUDA ms | Paired reduction |
|---|---:|---:|---:|
'''+ '\n'.join(core_table)+'''

## Selected design and remaining bottleneck

One 128-thread CTA owns 64 complete query rows of one outer pair row. TMA
prefetches three K/V/bias stages; an elected thread issues copies, and the
cooperating warpgroup consumes them. Two score fragments support QK/PV
scheduling. Resident Q is reused as the final TMA output staging allocation.
Whole-consumer barriers precede stage reuse. Softmax uses stable online max,
FP32 unrounded probability sums and FP32 base-2 LSE; only PV operands/output
are rounded to BF16. Four CTAs can reside per SM: 116 registers/thread,
54272 bytes kernel shared storage, zero spills.

NCU17294, L768: **1.073536 ms**, SM 63.03%, Tensor pipe 22.36%, HBM 29.70%,
L2 71.33%, active-warp occupancy 24.53%. The prior training baseline NCU17262
was 1.614912 ms, Tensor pipe 14.68%, HBM 11.34%. Utilization is not speedup.
PTXAS still reports C7514 serialization; the intended overlap is not fully
realized, and this remains a limit rather than a solved pipeline claim.

## Validation

- Core O and LSE against the prior training implementation and independent
  sampled FP64 attention: L64/384/768/1024; dense, mixed, one key, no keys,
  late live keys, large logits; changing-input/mask CUDA graph replay.
- Qualification17290: eight independent FP64 dQ/dK/dV/dBias cases at L64/128,
  plus 18 full-module/dropout/optimizer/fullgraph/frozen-parameter checks across
  both directions at L384. Complete-module gradient checks also cover L768/1024
  in both experimental and installed paired benchmarks.
- Memcheck, racecheck and synccheck: six runs at L64/256 (all six core cases),
  and three full-shape runs at L768 (mixed and fully masked), job17300. No
  errors, hazards or warnings in sanitizer summaries.
- Fresh-process installed verification17310: cold fullgraph F+B, opcheck
  schema/fake/AOT, bitwise identity to the qualified binary, all six actual
  module dispatches, all gradients, dtype/layout guards and opt-out.
- BF16 autocast with BF16 parameters and activations, both directions
  at L384/768, job17316: native dispatch and all gradients pass. FP16 is declined.
  Two exploratory AMP fixtures hit existing baseline restrictions before the
  candidate ran: FP32 activations select the reference branch (17312); FP32
  parameters with BF16 activations fail the existing gate backward's mixed
  BF16/FP32 dot (17315). Those combinations are not newly qualified here.

## Rejected alternatives

| Candidate | Finding |
|---|---|
| `stream_r2` | Two consumers share bias; correctness/sanitizers pass but L384 core 346.8us vs 244.6us; C7508/C7518 compiler warnings. |
| `stream_r2_pipe` | Static first/steady/final schedule fixes register allocation; L768 core 1.981ms vs 1.581ms; one CTA and low active occupancy; full workload slower. |
| `stream_r2_bounds` | Norm/bias range certification plus stable fallback passes initial O/LSE checks but L768 core including preparation 2.626ms vs 1.583ms; rejected before promotion qualification. |

Range certification is not part of the installed implementation. No low-rank
bias approximation, exp polynomial, attention sparsification or changed
gradient threshold was selected.

## Artifacts and reproduction

- [Native source](cooperative_q2/fused.cu), [generator](make_cooperative.py),
  [build identity](cooperative_q2/build-ready.json).
- [Installed machine-readable results](installed-results.json),
  [promotion record](promotion-17310.json), [frozen complete native package](checkpoint17310/snapshot.json).
- `MINIWORLD_TRIATTN_TRAINING_FWD=0` selects the prior training forward.
  Unsupported shape/dtype/layout/device also retains the established fallback.
- Baseline dispatcher is frozen in `before-17310/main.py`; no unrelated source
  edits were included. Existing standalone CUDA/inference forward is separate.
- Builders and benchmark/qualification scripts are here; use Slurm on node02,
  never compile CUDA or run GPU tests on the login node. `promote.py` is a
  one-time guarded transaction, not an idempotent reinstall command.
'''
(ROOT/'README.md').write_text(text)
print('PUBLISHED',len(hashes),'frozen files',flush=True)
