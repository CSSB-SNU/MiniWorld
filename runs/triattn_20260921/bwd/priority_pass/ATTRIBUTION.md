# Installed backward time attribution — job16664

H100 80GB, B1 BF16 C128/H4/D32, all input and parameter gradients, dropout0,
every seventh key masked. All five fusion flags are enabled; installed
ldmatrix_bias dK/dV, vector_bias dQ, warp_packed projection/LN,
n64_staged gate, shared_z_tensor Q/K/V/gate weight gradients.

All component times come from complete backward CUDA graph replays under
CUPTI, with ten profiler-attached warmup replays discarded and fifteen
measured replays. Percentages use the measured kernel-time sum for that
same profile. No scaling to the independent CUDA-event timing is applied.
CUDA-event totals outside the profiler use the median of nine rounds of
fifteen graph replays. Small differences reflect profiling, clocks and
inter-kernel gaps; this is backward without the forward or optimizer step.

| L | Direction | Event total ms | Profile kernel sum ms | Kernels |
|---:|---|---:|---:|---:|
| 384 | starting | 1.111 | 1.111 | 16 |
| 384 | ending | 1.151 | 1.167 | 17 |
| 768 | starting | 6.061 | 6.032 | 16 |
| 768 | ending | 6.237 | 6.611 | 17 |
| 1024 | starting | 13.030 | 13.370 | 16 |
| 1024 | ending | 13.396 | 13.641 | 17 |

## L384

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 0.4868 | 43.80% | 0.4880 | 41.81% |
| dQ | 0.2048 | 18.43% | 0.2049 | 17.55% |
| Bias final reduction | 0.0861 | 7.75% | 0.0867 | 7.43% |
| Weight gradients (six) | 0.1382 | 12.44% | 0.1387 | 11.88% |
| Projection/LN/residual | 0.1057 | 9.51% | 0.1084 | 9.29% |
| Gate/output/delta | 0.0779 | 7.01% | 0.0780 | 6.68% |
| Ending dY layout copy | 0.0000 | 0.00% | 0.0507 | 4.34% |
| Bias mask/layout | 0.0077 | 0.69% | 0.0076 | 0.65% |
| LN parameter reductions | 0.0043 | 0.38% | 0.0043 | 0.37% |

Weight-gradient detail, including each GEMM split-K reduction:

| Weight | Starting ms | Ending ms |
|---|---:|---:|
| W_out | 0.0313 | 0.0314 |
| W_q/k/v/gate (shared input) | 0.0778 | 0.0781 |
| W_bias | 0.0292 | 0.0291 |

## L768

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 2.9727 | 49.28% | 3.2259 | 48.80% |
| dQ | 1.2783 | 21.19% | 1.3762 | 20.82% |
| Bias final reduction | 0.6290 | 10.43% | 0.6297 | 9.53% |
| Weight gradients (six) | 0.4540 | 7.53% | 0.4543 | 6.87% |
| Projection/LN/residual | 0.3659 | 6.07% | 0.3882 | 5.87% |
| Gate/output/delta | 0.3076 | 5.10% | 0.3082 | 4.66% |
| Ending dY layout copy | 0.0000 | 0.00% | 0.2029 | 3.07% |
| Bias mask/layout | 0.0162 | 0.27% | 0.0169 | 0.26% |
| LN parameter reductions | 0.0079 | 0.13% | 0.0083 | 0.13% |

Weight-gradient detail, including each GEMM split-K reduction:

| Weight | Starting ms | Ending ms |
|---|---:|---:|
| W_out | 0.1117 | 0.1120 |
| W_q/k/v/gate (shared input) | 0.2711 | 0.2676 |
| W_bias | 0.0712 | 0.0747 |

## L1024

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 6.9498 | 51.98% | 6.8716 | 50.38% |
| dQ | 2.9985 | 22.43% | 2.9769 | 21.82% |
| Bias final reduction | 1.4453 | 10.81% | 1.4460 | 10.60% |
| Weight gradients (six) | 0.7506 | 5.61% | 0.7509 | 5.51% |
| Projection/LN/residual | 0.6432 | 4.81% | 0.6581 | 4.82% |
| Gate/output/delta | 0.5452 | 4.08% | 0.5458 | 4.00% |
| Ending dY layout copy | 0.0000 | 0.00% | 0.3531 | 2.59% |
| Bias mask/layout | 0.0256 | 0.19% | 0.0255 | 0.19% |
| LN parameter reductions | 0.0120 | 0.09% | 0.0125 | 0.09% |

Weight-gradient detail, including each GEMM split-K reduction:

| Weight | Starting ms | Ending ms |
|---|---:|---:|
| W_out | 0.1939 | 0.1934 |
| W_q/k/v/gate (shared input) | 0.4389 | 0.4399 |
| W_bias | 0.1178 | 0.1176 |

The six weight gradients now launch six kernels: shared-input Q/K/V/gate
GEMM plus FP32 split reduction, output GEMM plus reduction, bias GEMM
plus reduction. The complete backward has 16/17 kernels (starting/ending),
down from 22/23 before this pass. The complete sequence and all six
parameter gradients remain accounted for. Ending retains its dY copy.

Raw data: `attribution-16664-L*.json`; Chrome traces:
`attribution-16664-L*-e*-trace.json`. The summary JSON asserts that
the category totals equal every kernel in the profile.
