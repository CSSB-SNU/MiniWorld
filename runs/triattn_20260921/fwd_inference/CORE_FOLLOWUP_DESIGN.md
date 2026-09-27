# Inference core follow-up after output fusion

The baseline is the qualified18962 inference entry: front8, per-length
h4kv_local1/hot6t/hot4t core, and outproj_c1. Both candidates and controls keep
the fused output boundary. Production/training and four-head attention CTA
work are outside this experiment.

## WGMMA loop-backedge completion

The failed pipeclean32_c4 PTX emits scalar accumulator copies at the loop
backedge, after WGMMA commit but before the next iteration's initial wait.
Those compiler-generated copies force serialization. pipeend32_c4 and
pipeend64_c2/c4 add completion and operand fences before the backedge. PTXAS
serialization diagnostics disappear; the emitted SASS issues next-QK HGMMA,
then current MUFU.EX2, then the completion wait. This fixes the instruction
schedule but does not imply a speedup. The separate score lifetime, smaller
key tiles and register/CTA budget still cost time.

Representative before/after instruction sequences are in
[endwait-codegen-evidence.json](endwait-codegen-evidence.json). NVIDIA's
[WGMMA guide](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/wgmma_programming.html)
and the [FA3 implementation](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp)
were consulted for completion and scheduling context; the diagnosis and
performance evidence here come from our generated binaries.

## Numerator and denominator fusion

N40 WGMMA computes32 output channels plus8 replicated P*1 channels. V retains
the same resident shared layout. For each K16 slice, the descriptor's next
N32-group offset points at one shared32x16 ones tile, reused for all slices.
This adds only1KiB of shared allocation after alignment; it creates no global
V repacking or intermediate tensor. Normal BF16 P and summation order are
preserved. Stable retry remains the original N32 path. This adapts the local
training-core N40 descriptor technique to the resident inference algorithm.

## Query register reuse and probability lifetime

Qhalf keeps the first16 query channels in four packed registers per thread.
Every key iteration issues an RS QK for that slice and the existing SS QK for
the other slice, with the same accumulation order. Qfull tests retaining both
slices in eight registers. A single head still owns each CTA; queries remain
independent warpgroup work. Full-module timing must decide whether fewer
shared reads outweigh register spills.

Stream-pack candidates consume eight score/bias values at a time, immediately
compute exp and pack pairs into BF16 P registers. Stable retry keeps its old
calculation. This tests shorter scalar lifetimes with and without N40; it can
also delay the next bias request in the one-stage path, so fewer live values
alone do not establish a gain.

## Stable retry separation

Split-retry candidates instantiate normal and stable kernels separately.
The fast kernel initializes flags, computes the existing max-free path and
marks only unsafe query tiles. The second kernel exits immediately for rows
without flagged tiles and overwrites only flagged outputs. This retains
bitwise behavior for normal tiles in a partly unsafe row. No Q/K/V/gate
intermediate is added; the extra flags take4*L*(L/64+1)*4 bytes,278528B at1024.
The normal path still pays one extra launch/flag write, and rare repairs
recompute row-local projections. Register allocation and complete FWD decide
whether this is useful.

The original row-wide prototype19099 was cancelled before qualification:
repairing unflagged tiles would change their rounding. The corrected variants
are split6q/split4q. Full-Q prototype19110 was cancelled after detecting an
inactive second-slice generator replacement; qfull4v2 contains both RS issues.
N40 build19053 failed due to the extended-MMA header configuration;19062 uses
the required feature macro. These are not performance evidence.
