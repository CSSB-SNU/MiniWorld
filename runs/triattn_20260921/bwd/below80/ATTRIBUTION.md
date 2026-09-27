# Installed backward time attribution — job16607

> **Historical profile:** [../priority_pass/ATTRIBUTION.md](../priority_pass/ATTRIBUTION.md) records the current installed profile16664 after dK/dV and joint weight-gradient improvements. This file is the pre-pass attribution16607.

H100 80GB, B1 BF16 C128/H4/D32, all input and parameter gradients, dropout0,
every seventh key masked. All five fusion flags are enabled; installed
vector_bias dK/dV and dQ, warp_packed projection/LN, n64_staged gate.

All component times come from complete backward CUDA graph replays under
CUPTI, with ten profiler-attached warmup replays discarded and fifteen
measured replays. Percentages use the measured kernel-time sum for that
same profile. No scaling to the independent CUDA-event timing is applied.
CUDA-event totals outside the profiler use the median of nine rounds of
fifteen graph replays. Small differences reflect profiling, clocks and
inter-kernel gaps; this is backward without the forward or optimizer step.

| L | Direction | Event total ms | Profile kernel sum ms | Kernels |
|---:|---|---:|---:|---:|
| 384 | starting | 1.159 | 1.159 | 22 |
| 384 | ending | 1.213 | 1.213 | 23 |
| 768 | starting | 6.320 | 6.307 | 22 |
| 768 | ending | 6.526 | 6.676 | 23 |
| 1024 | starting | 13.611 | 13.588 | 22 |
| 1024 | ending | 13.945 | 14.088 | 23 |

## L384

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 0.4978 | 42.97% | 0.4975 | 41.01% |
| dQ | 0.2031 | 17.53% | 0.2028 | 16.72% |
| Bias final reduction | 0.0821 | 7.08% | 0.0822 | 6.77% |
| Weight gradients (six) | 0.1814 | 15.66% | 0.1818 | 14.99% |
| Projection/LN/residual | 0.1048 | 9.05% | 0.1084 | 8.94% |
| Gate/output/delta | 0.0773 | 6.67% | 0.0783 | 6.46% |
| Ending dY layout copy | 0.0000 | 0.00% | 0.0500 | 4.12% |
| Bias mask/layout | 0.0077 | 0.67% | 0.0076 | 0.63% |
| LN parameter reductions | 0.0044 | 0.38% | 0.0045 | 0.37% |

Weight-gradient detail, including each GEMM split-K reduction:

| Weight | Starting ms | Ending ms |
|---|---:|---:|
| W_out | 0.0319 | 0.0319 |
| W_q | 0.0336 | 0.0338 |
| W_k | 0.0305 | 0.0307 |
| W_v | 0.0304 | 0.0304 |
| W_gate | 0.0305 | 0.0304 |
| W_bias | 0.0245 | 0.0246 |

## L768

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 3.0980 | 49.12% | 3.2056 | 48.01% |
| dQ | 1.2785 | 20.27% | 1.3202 | 19.77% |
| Bias final reduction | 0.5849 | 9.27% | 0.5877 | 8.80% |
| Weight gradients (six) | 0.6480 | 10.27% | 0.6488 | 9.72% |
| Projection/LN/residual | 0.3652 | 5.79% | 0.3797 | 5.69% |
| Gate/output/delta | 0.3082 | 4.89% | 0.3096 | 4.64% |
| Ending dY layout copy | 0.0000 | 0.00% | 0.1995 | 2.99% |
| Bias mask/layout | 0.0162 | 0.26% | 0.0166 | 0.25% |
| LN parameter reductions | 0.0083 | 0.13% | 0.0086 | 0.13% |

Weight-gradient detail, including each GEMM split-K reduction:

| Weight | Starting ms | Ending ms |
|---|---:|---:|
| W_out | 0.1166 | 0.1170 |
| W_q | 0.1184 | 0.1188 |
| W_k | 0.1156 | 0.1149 |
| W_v | 0.1153 | 0.1151 |
| W_gate | 0.1150 | 0.1148 |
| W_bias | 0.0671 | 0.0683 |

## L1024

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 6.9879 | 51.43% | 7.0762 | 50.23% |
| dQ | 2.9008 | 21.35% | 2.9362 | 20.84% |
| Bias final reduction | 1.3596 | 10.01% | 1.3607 | 9.66% |
| Weight gradients (six) | 1.1198 | 8.24% | 1.1209 | 7.96% |
| Projection/LN/residual | 0.6367 | 4.69% | 0.6554 | 4.65% |
| Gate/output/delta | 0.5466 | 4.02% | 0.5498 | 3.90% |
| Ending dY layout copy | 0.0000 | 0.00% | 0.3516 | 2.50% |
| Bias mask/layout | 0.0252 | 0.19% | 0.0253 | 0.18% |
| LN parameter reductions | 0.0119 | 0.09% | 0.0124 | 0.09% |

Weight-gradient detail, including each GEMM split-K reduction:

| Weight | Starting ms | Ending ms |
|---|---:|---:|
| W_out | 0.2023 | 0.2029 |
| W_q | 0.2039 | 0.2043 |
| W_k | 0.2011 | 0.2009 |
| W_v | 0.2012 | 0.2007 |
| W_gate | 0.2010 | 0.2006 |
| W_bias | 0.1102 | 0.1114 |

The six weight gradients launch twelve kernels: one GEMM and one split-K
reduction each. Attribution uses the validated launch order in
`AttentionGate.backward` (output weight) and `Front.backward`
(Q/K/V/gate/bias), and asserts their position relative to the native
projection/LN kernel. Ending adds one full dY layout conversion before
gate backward; the remaining mask/copy work belongs to bias preparation.

Raw data: `attribution-16607-L*.json`; Chrome traces:
`attribution-16607-L*-e*-trace.json`. The summary JSON preserves
every kernel-to-category mapping and verifies that nothing is omitted.

Pilot16606 attached CUPTI and immediately measured five replays; its
ending captures showed a larger difference from event totals. The
current measurement adds profiler-attached warmup and supersedes it.
