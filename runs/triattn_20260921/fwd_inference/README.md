# TriangleAttention inference forward

The user's 2026-09-26 direction supersedes the training-forward constraint for this work:
develop inference-only forward; do not retain activations for backward. Training installation
18246 is a separate workload and remains unchanged. These artifacts are explicit experiments,
not installed dispatch.

**Current selection,19134/19157:** L1024 uses native CUDA `qfull4v2`, retaining
the whole Q tile in registers across the key loop. It uses128 registers without
spills, single-head CTAs, resident KV and two TMA bias slots. Actual-default
complete FWD takes **2.22–2.23% less time** than18962; the isolated native boundary
improves3.67%. L384 `h4kv_local1` and L768 `hot6t` stay unchanged; L768's small
pilot gain is inconclusive. All lengths keep `front8` and `outproj_c1`.
Against original Anthropic, current complete FWD is **32.00–32.25% /6.73–8.47%
/5.87–7.43% faster** at384/768/1024. Native bitwise/FP64/retry/changed graphs,
all three sanitizers at128/768/1024, and actual-default full-module/fullgraph
checks pass. [Results and sixteen controls](CORE_FOLLOWUP_RESULTS.md),
[JSON](core-followup-results.json), [design](CORE_FOLLOWUP_DESIGN.md).
This is an explicit inference experiment; production and training are unchanged.
Reproduce the prior1024 path with `load('hot4t', out_artifact='outproj_c1')`.
SOL90 remains unmet.

**Previous output selection,18946/18962:** the existing per-length cores use native CUDA
`outproj_c1` for output projection + BF16-rounded residual + ending-orientation store.
Actual-default complete FWD improves **7.09–8.85% at384,5.82–5.83% at768 and4.32–4.83%
at1024** over the previous selection. Relative to original Anthropic it is **32.12–32.45%,
6.47–8.63% and2.98–5.03% faster**, respectively. Output HBM traffic falls about32–39% in
NCU. Selected output native FP64/graph/sanitizers and complete-module/fullgraph checks pass.
Those core pipeline controls fail to improve performance; the later query-reuse
follow-up above replaces only the1024 core.
[Results](PIPELINE_OUTPUT_RESULTS.md), [JSON](output-pipeline-results.json),
[design](PIPELINE_OUTPUT_DESIGN.md). This remains an explicit inference experiment;
production and training are unchanged. `load(out_artifact=None)` retains the previous tail.

**Prior investigation,18909:** the user closed four-head attention CTAs. Fresh selected
kernel profiling finds HBM throughput only10.5%/7.0% at768/1024, with tensor-pipe activity
about28% and special-function pipe activity43–45%. The next core hypothesis is true
QK/softmax/PV overlap within a feasible register budget. Output-GEMM plus residual
epilogue fusion is a separate concrete HBM-saving opportunity. [Bottleneck analysis
and proposed algorithms](BOTTLENECK_ANALYSIS.md), [measurements](bottleneck-analysis.json).
That profiling investigation itself introduced no kernel or speedup.

**Head-last bias / four-head CTA experiment18806–18831, rejected:** native CUDA directly
produces `[1,L,L,4]` and four warpgroups share query input within one CTA. Six controls
cover one/two stages, direct vector/scalar bias reads, shared-memory rearrangement,
planar bias and independent per-head TMA rings. Best head-last full FWD is37–48% slower
than the current selection. The existing front already writes planar bias directly;
there was no separate transpose to remove. DRAM traffic remains essentially unchanged,
while shared-memory read and synchronization costs increase. Native FP64/graph pilots
pass for all six; limited-shape sanitizer/full-module checks pass for the primary and
rearranged candidates. [Results and qualification scope](CTA_HEADS_RESULTS.md),
[design](CTA_HEADS_DESIGN.md), [JSON](cta-heads-results.json). Selection remains unchanged.

**Previous core selection, qualified18721/18722 and selected-entry18745:** the cores were
L384 `h4kv_local1`, L768 `hot6t`, and L1024 `hot4t`, with `front8`. Relative to the previous
hot6t default, full inference FWD is **8.06–8.10% faster at L384** and **1.35–1.63% faster at
L1024**; L768 is unchanged. Against original Anthropic, L384 is **25.97–26.88% faster**, L768
**1.23–2.51% faster**, while L1024 medians remain **0.56–2.35% slower**. Fourteen controls
cover KV materialization, CTA order, head projection width, bias staging and front fusion.
[Qualified results](HEAD4_KV_RESULTS.md), [design](HEAD4_KV_DESIGN.md), [JSON](head4-results.json).
Native FP64/retry/graph, selected-shape sanitizers and actual-default complete-module/fullgraph
checks pass. Production dispatch and all 48 training18246 files remain unchanged.

**Previous selection, qualified18586/18598:** `candidate.load("hot6t")` retains this checkpoint.
The QKV/attention/gate kernel accumulates both P*V and P*1 on Tensor Cores, with a stable
on-chip retry for unsafe rows. Against our previous `resident6` fusion, full FWD time falls
**6.37–6.57% at L768** and **8.46–8.50% at L1024**, with L384 essentially unchanged.
Against original Anthropic, L384 is **18.78–20.08% faster**, L768 **1.74–3.30% faster**, but
L1024 remains **1.47–3.45% slower**. [Qualified results and twelve controls](QKV_ALGORITHM_RESULTS.md),
[algorithm details](QKV_ALGORITHM_DESIGN.md), [machine-readable evidence](algorithm-results.json).
Native/full-module FP64, changed-input/weight/mask graphs, fullgraph and three sanitizers pass.
Default engine dispatch is unchanged; all48 training checkpoint18246 files still match.

**Previous resident6 comparison, job18528:** against the original Anthropic fused inference block,
`resident6 + front8` takes **18.51-19.84% less time at L384**, but **3.71-5.40% more at L768**
and **10.84-13.55% more at L1024**. [Anthropic comparison](ANTHROPIC_COMPARISON.md) records
both directions, actual default dispatch, two residual forms and 64x40 paired graph timings.
All-masked semantics differ, so this is not an unrestricted drop-in replacement.

Checkpoint18488 qualified this candidate and found **3.71-6.58% starting** and
**17.14-23.55% ending** reductions against the installed ordinary eval module. Those are
[internal-baseline results](RESULTS.md), not Anthropic gains. Use `candidate.load("resident6")`
to retain this old control.

## Fusion boundary

1. Native CUDA `front8`: optional ending transpose + LayerNorm + bias projection + key mask.
   Returns only normalized Z and contiguous bias. No mean/rstd output. Supports FP32/BF16 LN
   affine parameters and BF16 projection weights.
2. At L384, native CUDA/TMA/WGMMA `h4kv_local1` projects all four KV heads together, then
   streams two KV/bias slots per attention warpgroup. Grid X groups queries and neighboring
   heads for one outer row; Q/gate remain local. K and V are forward intermediates, with no
   backward saves. At L768, `hot6t` keeps KV in shared memory for one head/outer-row CTA;
   Q and gate are computed per query batch. Online attention and sigmoid gating share the
   kernel. Returns only the BF16 gated attention result A. Q/K/V/gate/LSE have no global tensor
   allocations or explicit global stores. The BF16 attention-output rounding before gating is
   preserved. The normal path uses tensor-core numerator/denominator reduction and retries
   unsafe tiles with stable online softmax. Register spills remain: 80 registers and 80-byte
   stack frame at L768. L1024 uses `qfull4v2`: four query warpgroups, two bias slots,
   128 registers and zero stack/spills, keeping Q/K/V/gate on chip. Both QK K16
   slices reuse Q in eight packed registers per thread across the key loop.
3. Native CUDA/TMA/WGMMA output projection plus residual, preserving the intermediate
   BF16 projection rounding and storing directly in the optional ending orientation.
   The post-projection global temporary is eliminated. `outproj_c1` uses128 threads,
   90 registers with no spills and66560B dynamic shared memory.

`front8` assigns eight lanes per LN row and processes eight BF16 values per residual thread.
The initial 32-lane `front` control regressed starting L384. `front16` improved that result;
the final `front8` also vectorizes residual access. They are complete envelope controls, so
the difference between front16/front8 cannot be attributed solely to lane count.

Z, bias and A remain forward dependencies. The projected output is now consumed inside
the fused output kernel. Bias is shared across the outer-row
dimension; the implementation does not materialize attention scores/probabilities globally.

## Candidates

| Artifact | Organization | Observation |
|---|---|---|
| h4kv_local1 | Joint four-head KV projection, one query warpgroup, two TMA KV/bias slots, row-local query grid | Current L384 selection; 8.1% faster than hot6t |
| hot4t | Four query warpgroups; resident KV; two bias slots; no spills | Previous L1024 core; retained as a frozen control |
| qfull4v2 | Full Q register reuse; four query warpgroups; resident KV; two bias slots; no spills | Current L1024 selection; complete FWD time 2.22–2.23% lower than18962 |
| hot6t | Six cooperative warpgroups; resident KV; P*V and P*1 WGMMA; stable on-chip retry | Current L768 selection; previous default at every length |
| resident6 | Six cooperative warpgroups; resident KV, batched Q/gate; one/two bias stages | Initial actual module improvement about 2-5%; 80 registers with 68-72 spill bytes/thread at production lengths |
| resident4 | Four cooperative warpgroups; two bias stages | 107 registers, no spills; slower at L384, approximately tied at L768, improves L1024 |
| stream4 | One KV producer and four attention consumers (two consumers at L384) | No backward saves; KV recomputation across query groups loses at L768/1024 |
| stream2 | One KV producer, two consumers | More query groups/recomputed KV; slower at L768/1024 |

No setmaxnreg redistribution is used. TMA transaction/phase barriers protect loads, WGMMA
completion protects projection inputs, CTA barriers protect scratch-union reuse, and TMA store
source completion precedes overwriting output scratch.

## Comparison and validation

The primary baseline is original Anthropic `opt_core.attn.pair_fused.tri_attn_block` with
`impl=fpf`, `core=default`, `ln=fused`, under `torch.inference_mode()`. The comparison includes
LN, bias, QKV, attention, gate/output projection and residual in both directions. It uses the
faster of two input-preserving Anthropic residual forms; the fused in-place form includes an
input copy. Nonzero random weights are identical between paths, with LN affine values exactly
representable in BF16 to match upstream's internal rounding. See the comparison for provenance,
numerical tolerances and the all-masked difference.

The earlier checkpoint18488 baseline is installed `TriangleAttention.eval()`; its training
front-fusion path is disabled by its existing grad-enabled guard. It is retained as a separate
internal comparison.

`check.py` also records a diagnostic native boundary comparison against separate cuBLAS
projections + frozen attention + unfused gate arithmetic. Its large apparent speedup must not
be reported as a full-module gain. Module graphs run alternating AB/BA orders; changed inputs,
weights and masks are replayed and compared.

Validation is forward-only: native edge cases and sampled FP64 arithmetic, complete module
outputs including attention-update errors, nontrivial LN affine parameters, constant and
near-constant LN inputs, no_grad/inference_mode, CUDA Graph replay, fullgraph compilation,
memcheck, racecheck and synccheck. Gradients/backward timing are outside this inference task.

Jobs: 18451 resident builds/native checks; 18455 resident actual-module comparisons;
18458 streaming controls; 18463 resident sanitizers; 18482 first inference envelope;
18483 first envelope qualification; 18486 envelope tuning; **18488 selected qualification**;
**18489 fresh actual-dispatch traces**. All are complete. Earlier18461 failed compilation on
an implicit BF16 conversion; its blocked dependent18474 was cancelled, and explicit conversion
fixed the build before18482. No other jobs were cancelled.

Native core tests cover six masks/logit fixtures at L64/128/384/768/1024. Core sanitizers pass
at L128/384/768/1024. Selected front checks cover FP32/BF16 affine parameters, both orientations,
random/constant/near-constant inputs and changed-input/weight graph replay at L128/384/768/1024;
all three front sanitizers pass at1024. Complete module checks at L384/768/1024 cover mixed,
dense, one-key, all-masked and absent masks, sampled FP64 full-module arithmetic, changed-input
graphs, no_grad/inference_mode and fullgraph compilation. Gradient tests are intentionally out
of scope. Source/binary hashes remain intact, and all48 files in training checkpoint18246 are
unchanged (`training-preserved.json`).

Some initial one-call profiler traces missed the first CUDA events; job18489 collects three
synchronized forwards per trace and confirms all selected native stages in both directions.
It does not replace or modify the paired timing evidence from18488.

## Explicit experiment entry

Add this directory to `sys.path`, then:

```python
from candidate import load
inference_forward = load()  # qualified per-length selection; call before capture/compile
model.eval()
with torch.inference_mode():
    y = inference_forward(model, pair, mask)
```

The entry supports B1, C128, H4/D32, BF16 and L384/768/1024. It rejects training/grad-enabled use
and unsupported shapes instead of silently changing
semantics. It never compiles at runtime. `native.py` verifies the source and binary hashes before
loading an artifact. No SOL90 claim is made for this new inference path.
