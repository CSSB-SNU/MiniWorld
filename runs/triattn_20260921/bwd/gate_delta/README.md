> **Latest update:** n64_staged is qualified16324 and installed. It preserves all
> four outputs bitwise, uses33792B shared/80 registers with6CTA resource limits,
> and improves the isolated gate kernel8.60%/1.96%/1.21% at384/768/1024 over the
> previously installed context_ready. NCU16325 HBM86.41%; SOL90 is not reached.
> Actual installed cold/default validation16333 passes, including first compiled
> backward, ten opt-out/metadata cases, BF16 autocast and FP16 refusal. See SOL90_PROGRESS.

# CUDA gate/output backward + delta

This extends the autograd boundary through attention and gate/output projection.
The native epilogue computes dO, dG, gated output A for dWo, and delta while O/dO
are on chip. dO, dG and A remain BF16; delta uses the rounded dO. dWo remains cuBLAS.
Forward uses the established kernels and is bitwise unchanged in qualification.
New backward kernels are CUDA C++/TMA/WGMMA; no Triton implementation was added.

The selected GPU kernel is `tma_store`, qualified16254. It has one producer WG
and one consumer WG, 256 threads,82944 bytes shared; all source operands arrive
by TMA. WGMMA produces the128-channel dgrad. Three output tiles reuse the dead
input/weight shared buffers; TMA stores them after an async-proxy fence and the
128-consumer barrier. The issuing thread waits for bulk reads before CTA exit.
No intermediate dA reaches HBM and no separate delta-preprocess pass remains.

Pilot16253: epilogue+delta L3840.136->0.088ms, L7680.483->0.319ms,
L10240.843->0.556ms. FP64/mixed data checks pass, with dO/dG/A bitwise matching
stock and delta relative L2 about3.5e-8. Whole-module16254 gains another2-3%
backward over the already optimized installation. All parameter/input gradients,
masks, dropout.25, SGD updates, frozen parameters, fullgraph Inductor, memcheck,
racecheck and synccheck pass. NCU16255:318.048us, HBM83.849%, SM23.967% at L768.
This kernel does not yet reach SOL90.

The original source versions using dynamically indexed delta accumulators have
32-byte local stacks despite the compiler reporting zero spills. SASS shows64
LDL/66STL instructions. Replacing division with approximate reciprocal, packing
BF16 shared/global accesses and indexing the eight sums by constant fragment
indices removes the local stack, but direct global stores are still too slow.
TMA stores are the decisive improvement. N64 and cooperative variants are not
adopted; see raw pilot files and `../SOL90_PROGRESS.md`.

A cold autograd-worker context bug was found AFTER warmed module qualification:
job16262/16264 fails in cuTensorMapEncodeTiled with CUDA_ERROR_INVALID_CONTEXT.
`context_ready` adds explicit cudaSetDevice in the host entry, after CUDAGuard.
`context-ready-gpu-equivalence.json` proves its GPU SASS is identical to the
qualified tma_store kernel. Fresh eager and compiled validation is job16265.
Fresh eager and compiled checks pass16265. The corrected artifact is installed
and enabled by default; complete installed verification16266 passes all ten
opt-out/metadata checks, cold compiled execution, BF16 autocast and paired timing.

## N64 staged schedule

Adjacent output-channel CTAs share dY through L2, so measured DRAM reads do not
increase relative to the prior full-width kernel. One128-thread CTA issues TMA
for dY and a64-channel weight tile, computes WGMMA, then reuses the retired weight
storage for gate/output TMA loads. The pointwise epilogue reuses these shared
tiles for dG/A and the old dY tile for dO. Three TMA stores commit the outputs;
delta is computed from rounded dO as before. No additional HBM intermediate,
precision reduction, or global atomics are introduced. Explicit host cudaSetDevice
is retained for fresh autograd-worker contexts. The old two-WG description above
belongs to context_ready and is preserved as experiment history.

A128-row variant initially failed delta indexing16304. Constant fragment-derived
indexing fixes it (16312), with all outputs bitwise equal, but its reduced CTA
occupancy makes it22-26% slower. It is not installed.
