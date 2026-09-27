"""Summarize completed measurements and source/binary provenance."""
from pathlib import Path
import json,hashlib

R=Path(__file__).resolve().parent
ROOT=R.parent.parent
ENGINE=ROOT/'.engine-release-2.0.0'
rows=[json.loads((R/f'final-D{d}-L{n}.json').read_text()) for d in (256,384,512) for n in (384,768)]
assert all(r['complete'] for r in rows)
summary=[]
for r in rows:
    t=r['dropout25'];old=t['previous']['median_us'];new=t['new']['median_us'];tr=t['triton']['median_us']
    summary.append(dict(D=r['D'],L=r['L'],previous_us=old,new_us=new,reduction_pct=100*(1-new/old),triton_us=tr,triton_speedup=tr/new))
sanitizers={}
for d in (256,384,512):
    for n in (384,768):
        for tool in ('memcheck','racecheck'):
            p=R/f'{tool}-D{d}-L{n}.log'
            text=p.read_text() if p.exists() else ''
            ok='ERROR SUMMARY: 0 errors' in text if tool=='memcheck' else 'RACECHECK SUMMARY: 0 hazards' in text
            sanitizers[p.name]=dict(complete=ok,tail=text[-400:])
sources=[ENGINE/'src/miniworld_engine/kernels/trimul_inproj/cuda/h100_wide_forward.py',*sorted((ENGINE/'src/miniworld_engine/kernels/trimul_inproj/cuda/h100_sources/wide_forward').glob('*'))]
manifest=dict(sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()},cubins={f"{r['D']}-{r['L']}":r['cubins'] for r in rows},sanitizers=sanitizers)
context_path=R/'final-context-graph.json'
manifest['final_context_graph']=json.loads(context_path.read_text()) if context_path.exists() else None
(R/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
table='\n'.join(f"| {r['D']} | {r['L']} | {r['previous_us']/1000:.3f} | {r['new_us']/1000:.3f} | {r['reduction_pct']:.1f}% | {r['triton_us']/1000:.3f} |" for r in summary)
sanity='6 shapes × memcheck/racecheck: all passed, zero errors/hazards.' if all(v['complete'] for v in sanitizers.values()) else 'Sanitizer checks are still running; see manifest.json.'
readme=f'''# MiniWorld wide forward · 2026-09-23

Our K1/K3 forward kernels now cover D256/384/512, L384/768, B1, BF16,
bidirectional hidden width 2D, residual, and row-broadcast dropout.
The packaged implementation is `h100_wide_forward.Forward` in the engine release
worktree. This is an explicit forward plan with saved activations, not an autograd
registration. Existing automatic dispatch and backward implementations are unchanged.

## Measured full forward

H100, same-process alternating CUDA graph replays, 150 samples per path, median.
Includes live weight packing, K1, both contractions, K3, x_n/left/right/tri saves,
fixed 25% dropout scale and residual. Excludes compilation, plan construction,
RNG generation, CPU dispatch and backward. No clock/power lock was imposed.

| D | L | Previous CUDA ms | New ms | Time reduction | Triton ms |
|---:|---:|---:|---:|---:|---:|
{table}

The previous CUDA path is the engine's current wide `h100_width.Training` forward,
not the D128-specialized path. Triton was measured in the same process; missing
cache entries used the engine's heuristic-24 search and are not an exhaustive
best-Triton comparison. Dropout-zero measurements are in each final JSON.

## Changes

- Reuse our optimized K1 math, TMA/WGMMA primitives and BF16 mask support; save
  x_n and keep channel-major left/right and tri. The two contractions use cuBLAS.
- K3 normalizes the entire 2D channel row in shared memory, then reuses it across
  output-channel groups. K64/K128 weight chunks replace full-width weight slots.
  The normalized triangle never becomes a global intermediate.
- Warpgroup-local ldmatrix/stmatrix transposes replace scalar transposes. D256
  and D384 keep LN values in registers. D512 uses bounded packed shared-memory
  passes to avoid the larger register live set.
- D512 K1 uses two consumer warpgroups, a shared normalized operand and early
  release of each consumed weight chunk. Its separate input LN remains selected.
- The training epilogue retains BF16 projection/gate rounding followed by
  dropout and residual. No backward-only projection/gate buffers are saved.

The selected settings are in the packaged `wide_forward/selection.json`.
K3 uses respectively 4/3/4 consumer warpgroups at D256/384/512 and K64 chunks.
The selected K3 builds have zero ptxas spill loads/stores at all three widths;
the selected D512 K1 also has zero spills. D384 K1 retains a small 4-byte
load/store spill in the existing register-operand schedule; the spill-free
alternatives measured slower and were not selected.
Rejected candidates (extra producer warpgroup in K3, whole-block/smaller-ring
K1 variants, wider K chunks) remain in the experiment directory for comparison.
No SoL or maximal-optimization claim is made.

## Validation

- All six shapes passed both dropout25 and dropout0 against independent compiled
  PyTorch and the engine Triton forward, relative L2 < 0.005. Maximum PyTorch error
  was {max(r['checks'][m]['pytorch'] for r in rows for m in ('dropout25','dropout0')):.6f}; maximum Triton error was
  {max(r['checks'][m]['triton'] for r in rows for m in ('dropout25','dropout0')):.6f}.
- Saved x_n is identical to the previous implementation in all six selected cases;
  left/right and tri relative-L2 checks passed. The tanh sigmoid changes some BF16
  results, so forward output is not claimed bit-identical to the old path.
- Changed inputs, weights, affine values and dropout scales after graph capture:
  graph/eager bit-exact, framework-reference comparison passed.
- Separate plans retain independent saves; zero mask and zero input gamma passed.
- {sanity}
- Backward accuracy/performance with this new forward has not been tested or wired.

## Use

Use the engine release worktree on PYTHONPATH, not the older package pinned in
MiniWorld's current environment. Under `torch.no_grad()`:

```python
from miniworld_engine.kernels.trimul_inproj.cuda.h100_wide_forward import Forward

# leaves = (x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo)
plan = Forward(leaves, mask, dropscale)
y = plan()
xn, left_right, tri = plan.saved
```

A plan reuses its output and saved buffers on replay. Allocate separate plans for
outstanding forwards whose saves must coexist. Leaf tensors and dropout scales
remain live; the plan owns its BF16 mask copy (`plan.mask`).

## Reproduce

```bash
sbatch runs/trimul_forward_wide_20260923/final.sbatch
sbatch runs/trimul_forward_wide_20260923/sanitize.sbatch
python3 runs/trimul_forward_wide_20260923/report.py
```

Final job 16622; sanitizer job 16628. Raw per-shape checks, samples and selected
cubin SHA-256 are in `final-D*-L*.json`; packaged source hashes and sanitizer
status are in `manifest.json`. Sources preserve the upstream Apache-2.0 notices
and attribution while exposing MiniWorld kernel names.
'''
(R/'README.md').write_text(readme)
print(table)
print(sanity)
