# TriangleAttention kernel inventory

> **Latest backward update, 2026-09-23 — below80:** [bwd/below80/README.md](bwd/below80/README.md) records installed CUDA `vector_bias` dK/dV and dQ plus `warp_packed` projection/LN/residual. Actual installed job16580 passes all manifests, cold eager/compiled backward, AMP and10 opt-out checks. Additional whole-backward reduction versus the immediately preceding installation: L384 8.85–10.10%, L768 8.20–8.32%, L1024 7.17–7.51%; F+B also improves. Projection/LN/residual reaches **HBM SOL84.94% starting /82.84% ending**, so tuning stops there under the user's80% cutoff. dK/dV and dQ are faster but remain below80 (SM53.02% /58.09%). Gate and the final bias reducer are unchanged. No whole-backward SOL80/90 claim; forward optimization remains closed.

2026-09-22. Scope: this TriangleAttention forward optimization, H100,
B1 square BF16 C128/H4/D32, default dispatch with a broadcast key mask.
This does not cover other MiniWorld modules or backward kernels.
The subsequent backward optimization is recorded separately in
[bwd/README.md](bwd/README.md): native CUDA/TMA projection-gradient fusion is
installed in the engine training module. The 2026-09-23 update additionally fuses
backward dK/dV with grouped bias reduction at L768/1024; dQ remains unchanged.
[bwd/bias_fusion/README.md](bwd/bias_fusion/README.md) records qualification and
installed0.23–0.32% /2.74–2.99% additional whole-module backward improvement, half the bias scratch,
and32.7% less L768 whole-core DRAM traffic. L384 keeps the previous backward path.

## Final disposition requested by the user

The core SM SOL90 search is closed as unsuccessful. SOL90 was not reached;
recent candidates supplied no additional installed speedup. Preserve the
qualified installed core improvements and stop further SOL90 experiments.

| Workstream | Disposition |
|---|---|
| CUDA/TMA prologue | Complete for qualified L384/768/1024 forward cases |
| L384 producer/consumer prologue | Complete and selected by default at384 |
| CUDA/TMA epilogue | Complete for qualified cases; slower persistent variants rejected |
| Broadcast-mask preparation | Complete and selected at768/1024 |
| Bias preparation and uniform-row handling | Existing implementation retained; no further optimization commitment |
| SAFE recomputation | Barrier corrections and full-shape checks complete |
| Core improvements already installed | Retained with existing correctness/performance qualification |
| Additional core SM SOL90 objective | Unsuccessful; further search stopped by user decision |

Completion here covers the documented H100 forward stack, shapes, directions
and layouts. It does not claim backward, all shapes, or whole-training-step
qualification. The surrounds'90% result uses a measured streaming reference,
not90% theoretical hardware SOL.

## Installed execution kernels

Times are CUPTI component measurements from installed validation, shown as the
range of starting/ending directions. L384 uses job13916; L768/1024 use job14862.
They are not fresh simultaneous benchmarks or NCU/whole-block graph times.
The numerical source records and original kernel names are collected in
[kernel-inventory.json](kernel-inventory.json).

| Component | Actual implementation | L384 us | L768 us | L1024 us |
|---|---|---:|---:|---:|
| LayerNorm and Q/K/V/gate/bias projections | CUDA/TMA prologue; pipeline variant at384 | 78.4–79.8 | 273.8–277.9 | 476.6–484.7 |
| Mask preparation | CUDA `broadcast_mask_stage` | No separate kernel | 3.6 | 4.0–4.1 |
| Bias preparation | Triton `_stage_bias` at384; CUDA `stage_bias_m1_kernel` otherwise | 5.7–5.8 | 13.9–14.4 | 20.4–20.5 |
| Attention hot loop | Triton `_fwd` at384; CUDA `ta_core_broadcast::triattn_m1_kernel` otherwise | 161.7–162.7 | 782.5–783.5 | 1729.8–1732.8 |
| Stable SAFE recomputation | CUDA same kernel family, flag1024 | No separate kernel | 3.0–3.1 | 3.1–3.2 |
| Fully masked rows, mean of V | CUDA `uniform_rows_kernel` | No separate kernel | 3.5 | 4.1–4.2 |
| Gate, output projection, residual | CUDA/TMA `epilogue_tma<L>` | 65.3–66.2 | 206.9–213.9 | 355.9–369.2 |

Four kernels run in the captured L384 default path; seven at L768/1024.
SAFE/uniform values above are the light follow-up launches in the measured
normal input, not the cost of extensive SAFE recomputation or an empty mask.

| Main-kernel branch | Selection | Implementation status |
|---|---|---|
| L384 hot | Existing small-length route | Triton; main loop unchanged in this campaign |
| L768 hot | Qualified standard-scale square shape | CUDA hot1073741824, installed |
| L1024 hot | Qualified standard-scale square shape | CUDA hot536870912, installed |
| Broadcast generic hot | Other shapes/scales accepted by the broadcast wrapper | CUDA hot0, retained |
| Original native M1 | Nonbroadcast mask or broadcast-core opt-out | Existing CUDA M1, retained |
| SAFE | Exceptional rows/tiles requiring stable recomputation | CUDA flag1024; barrier correctness fixed |
| Surround fallback | Unsupported shape/layout/stack or disabled CUDA surrounds | Existing routes retained |

## CTA and TMA arrangement

| Kernel | CTA threads | Producer/consumer arrangement |
|---|---:|---|
| L384 prologue | 288 | One32-thread TMA producer and two128-thread WGMMA consumers; two resident CTAs |
| L768/1024 prologue | 256 | Cooperative CUDA/TMA schedule; dedicated-producer variant was not faster consistently |
| L768/1024 attention | 512 | One128-thread producer WG, three128-thread consumers; independent K/V producer warps and free barriers; one resident CTA |
| Epilogue | 256 | Cooperative CUDA/TMA schedule; persistent producer/consumer alternatives rejected |

## Installed improvements

Each row has its own comparison baseline. Percentages must not be added to
claim a new end-to-end gain.

| Kernel/change | Measured improvement | Disposition |
|---|---|---|
| L768 CUDA/TMA prologue versus already-tuned Triton | NCU340.864 →273.888us, 19.65% reduction | Installed |
| L768 CUDA/TMA epilogue versus already-tuned Triton | NCU216.768 →207.712us, 4.18% reduction | Installed |
| L384 prologue producer/consumer pipeline versus prior CUDA | About1–2% prologue reduction | Installed only at384 |
| Broadcast mask preparation at768 | About18.6 →4.0us; about14.6us saved | Installed |
| Broadcast mask preparation at1024 | About29.0 →4.1us; about24.8us saved | Installed |
| Initial broadcast attention specialization | Paired kernel reduction3.8% at768,2.9% at1024 | Installed |
| Subsequent exact core improvements | Shape constants, independent producers, N40 PV/den, constant descriptors, Q cache | Installed; individual paired results in CORE_SOL90_REPORT |
| Last L1024 specialization versus preceding package | Paired core reduction4.163% starting /3.971% ending | Installed |
| Recent search after that installation | No additional installed speedup | All measured candidates rejected |
| Bias stage | Existing implementation retained | No new speedup claimed |
| SAFE | Barrier fixes and full-shape validation | Correctness improvement; no throughput gain claimed |
| Uniform-row kernel | Existing behavior retained | No new speedup claimed |

Prologue LN reduction differs slightly from the prior Triton implementation;
its numerical qualification is documented in CUDA_REPORT. The qualified core
changes and epilogue preserve bitwise outputs against their stated baselines.

## Utilization definitions

| Kernel/profile | Time us | Metric against hardware peak | Against2.95TB/s measured stream |
|---|---:|---|---:|
| L768 prologue,13870 | 273.888 | DRAM80.883% | 91.905% |
| L768 epilogue,13870 | 207.712 | DRAM85.170% | 96.769% |
| Installed L768 attention,14862 | 795.904 | SM61.349%; tensor34.966% | Not the relevant roof |
| Installed L1024 attention,14862 | 1751.712 | SM65.249%; tensor37.068% | Not the relevant roof |
| L384 main attention | Not newly profiled for SOL here | No SOL90 qualification | No claim |

The surrounds passed90% of the measured streaming reference. None of these
figures establish90% SM SOL for the attention core or whole block.

## Rejected experimental families

This table groups related source variants; it is not a list of every generated
binary. Individual experiments and invalid runs remain in core_sol90/STATUS.md.

| Family | Result |
|---|---|
| Larger-shape producer/consumer prologue | No repeatable block gain;384-only default retained |
| Persistent/split epilogue | Lower traffic did not improve time; corrected split version also rejected |
| More consumers / smaller tiles / two resident CTAs | Loss of reuse, extra waits or register pressure; no replacement for installed core |
| Resident K/V, cluster multicast, cooperative/shared P | Compiler failures or slower measured core |
| Approximate exponentials, polynomial, LUT, half2 | Numerical or performance failures; no installed numerical approximation |
| FP16/scaled operands / repacked V / raw bias | Conversion, SAFE, accuracy or latency costs prevent a qualified win |
| WGMMA order, compiler flags, fence/dependency controls | Serialization, no-op code generation, or slower paired results |
| Lean TMA producer and mixed register allocation | Valid corrected variants bitwise but about0.7–1.9% slower |
| Native N16 M64/R4, four scores | Initial bitwise;914.397 versus789.325us |
| Native N16 M128/R3, four scores | Initial bitwise;853.927 versus795.098us |
| Native N16 M128/R4, two scores, fixed descriptors | Initial bitwise;859.529 versus787.573us; NCU SM56.017% |
| Native N16 four-score Q3/Q2, fixed descriptors | Both completed CPU builds; C7512 serialization, Q3 also spills; no GPU launch |

Sources: [CUDA_REPORT](CUDA_REPORT.md), [CTA_REPORT](CTA_REPORT.md),
[CORE_REPORT](CORE_REPORT.md), [CORE_KERNEL_REPORT](CORE_KERNEL_REPORT.md),
[CORE_SOL90_REPORT](CORE_SOL90_REPORT.md), [Q1024_REPORT](Q1024_REPORT.md),
[PHASE_N16_REPORT](PHASE_N16_REPORT.md).
