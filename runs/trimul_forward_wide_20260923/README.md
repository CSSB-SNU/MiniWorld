# MiniWorld wide forward · 2026-09-23

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
| 256 | 384 | 1.150 | 0.936 | 18.6% | 1.286 |
| 256 | 768 | 4.654 | 3.810 | 18.2% | 5.294 |
| 384 | 384 | 2.016 | 1.816 | 9.9% | 2.274 |
| 384 | 768 | 8.328 | 7.359 | 11.6% | 9.464 |
| 512 | 384 | 3.261 | 2.940 | 9.9% | 3.602 |
| 512 | 768 | 13.575 | 12.020 | 11.4% | 14.761 |

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
  was 0.003381; maximum Triton error was
  0.001023.
- Saved x_n is identical to the previous implementation in all six selected cases;
  left/right and tri relative-L2 checks passed. The tanh sigmoid changes some BF16
  results, so forward output is not claimed bit-identical to the old path.
- Changed inputs, weights, affine values and dropout scales after graph capture:
  graph/eager bit-exact, framework-reference comparison passed.
- Separate plans retain independent saves; zero mask and zero input gamma passed.
- 6 shapes × memcheck/racecheck: all passed, zero errors/hazards.
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
