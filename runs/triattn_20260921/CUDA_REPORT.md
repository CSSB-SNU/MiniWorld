# CUDA/TMA TriangleAttention optimization

> Follow-up: [CTA_REPORT.md](CTA_REPORT.md) adds the producer/consumer prologue and persistent CTA experiments. The bandwidth figures below describe the preceding CUDA/TMA version; L768 defaults remain that version.

The default `r07` / `r16` cells now select native CUDA/TMA surrounds for H100,
CPython 3.10, torch 2.10.0+cu128, bf16 C128/H4/D32, L384/768/1024, both directions.
The implementation and prebuilt library are in
`oc/opt_core/kernels/triattn_surround_tma/`. Unsupported shapes, layouts, stacks,
and non-fused LayerNorm retain their previous routes. No environment override is
needed for the qualified cases.

## Verified block performance

Same-job baseline and candidate alternate for three rounds. CUDA graph forward,
B=1, nonzero randomized output weights, fused out-of-place residual. Baseline is
the already-tuned serving Triton surrounds from the first Codex pass, not the
slower release defaults. Final serving qualification: Slurm job 13867.

| Direction | L | Tuned Triton (us) | CUDA/TMA (us) | Time saved |
|---|---:|---:|---:|---:|
| starting | 384 | 320.6 | 303.2 | 5.42% |
| starting | 768 | 1512.7 | 1468.0 | 2.95% |
| starting | 1024 | 3078.6 | 3009.8 | 2.23% |
| ending | 384 | 327.3 | 305.0 | 6.82% |
| ending | 768 | 1529.8 | 1489.5 | 2.63% |
| ending | 1024 | 3135.1 | 3045.4 | 2.86% |

The core dominates total block time and individual large-shape rounds vary by
several tens of microseconds. These results do not measure backward or complete
training-step throughput. At L384 the existing core is still Triton `_fwd`;
at L768/1024 it is the existing CUDA `triattn_m1_kernel`.

## Bandwidth target: measured stream versus hardware peak

Final installed-default NCU measurement: job 13870, L768 starting direction,
`--clock-control none --cache-control none`. Bandwidth below uses measured DRAM
bytes, not nominal tensor sizes. The predeclared streaming reference is 2.95 TB/s
from the handoff. The copy check in this job achieved about 2.86 TB/s.

| Kernel | DRAM MB | NCU us | TB/s | % of 2.95 TB/s stream | NCU % hardware peak |
|---|---:|---:|---:|---:|---:|
| baseline prologue | 763.38 | 340.86 | 2.240 | 75.92% | 66.81% |
| baseline epilogue | 594.23 | 216.77 | 2.741 | 92.93% | 81.80% |
| cuda prologue | 742.56 | 273.89 | 2.711 | 91.90% | 80.88% |
| cuda epilogue | 592.95 | 207.71 | 2.855 | 96.77% | 85.17% |

The 90% target is met for both surrounds relative to the measured streaming
reference. This is **not** a claim of 90% NVIDIA theoretical hardware SOL, nor
90% utilization of the attention core or whole block. Those are distinct targets.

## Implementation

- TMA loads pair activations and projection weights; WGMMA performs projections.
- Vectorized 128-bit LayerNorm loads/stores and hoisted affine weights remove
  the scalar memory-instruction bottleneck.
- `stmatrix` packs WGMMA fragments directly into shared output tiles. TMA stores
  Q/K/V in the existing per-head layout and gate in the pair layout.
- Epilogue TMA loads attention output, gate, weights, and residual. A universal
  65536-entry BF16 sigmoid table plus native BF16x2 multiplication replaces
  scalar activation work. This table is independent of input data and weights.
- `ldmatrix` / `stmatrix` handle epilogue fragments; TMA writes the final residual
  output, including ending-direction transposed addressing.
- The installed kernels have no register spills. Host checks validate tensor
  devices, sizes, dtypes and layouts. Source and binary SHA256 values are checked
  when the surround extension is loaded.

## Correctness

All six direction/length combinations passed the installed-default route check,
fp32-reference error gate, and dense / fully masked / irregular fresh-input
checks. Both residual aliasing paths were tested. The epilogue is bitwise equal
to the previous implementation, including a sweep through every finite BF16 gate
bit pattern at L384 in both directions. The prologue uses a slightly different
fp32 LN reduction order: it is not bitwise equal; its projection relative RMS
difference is about 2e-5 and block fp32-reference RMS remains about 1.7e-3.

Each profile verifies the actual CUDA surround names and the expected core name.
No fallback was used. The core source and binary are unchanged; core binary
SHA256 remains `3c766e0296351d643329c7203f94f25ad3c584c5d400b1ca92b74d35fdb225ba`.

## Reproduction

```sh
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/cuda_tma/build_native.py
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/cuda_tma/install_native.py
sbatch runs/triattn_20260921/cuda_tma/serving.sbatch
sbatch runs/triattn_20260921/cuda_tma/roof_serving.sbatch
```

`FPF_TRIATT_BACKEND=triton` selects the previous surrounds for controlled A/B
runs. `serving-e{0,1}-L{384,768,1024}.json` records timings and correctness;
`serving-roof-*.ncu-rep` / `.csv` record the bandwidth evidence. The source patch
`codex_cuda_tma.patch` is incremental to the first Codex optimization pass; rebuild
and install the binary after applying it. Earlier `codex_optimization.patch`
and `CODEX_REPORT.md` describe that first pass.

## Rejected designs

The initial serial TMA prologue took 529 us. An oversized producer/consumer
version took 779 us, and register-capped variants still took about 690–706 us.
Replacing scalar output scatter with stmatrix reduced this to about 358 us;
vectorized LN was needed to reach the final approximately 276–280 us range.
Larger 256-row tiles and the tested warp-specialized schedules were slower.
The initial scalar CUDA epilogue took 410 us; stmatrix/unrolling reduced it to
315 us, a scalar sigmoid table to 266 us, and paired BF16 math to about 209 us.
Negative results and source snapshots remain in `cuda_tma/`.
