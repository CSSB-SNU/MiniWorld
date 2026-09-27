# Installed backward time attribution — job17211

H100 80GB, B1 BF16 C128/H4/D32, all input and parameter gradients, dropout0,
every seventh key masked. All five fusion flags are enabled; installed
rs8_async_q4 dK/dV, rs_softmax_overlap dQ, warp_packed projection/LN,
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
| 384 | starting | 0.846 | 0.842 | 16 |
| 384 | ending | 0.899 | 0.898 | 17 |
| 768 | starting | 4.665 | 4.648 | 16 |
| 768 | ending | 4.885 | 4.844 | 17 |
| 1024 | starting | 10.183 | 10.004 | 16 |
| 1024 | ending | 10.509 | 10.492 | 17 |

## L384

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 0.3071 | 36.46% | 0.3082 | 34.30% |
| dQ | 0.1552 | 18.42% | 0.1554 | 17.30% |
| Bias final reduction | 0.0460 | 5.46% | 0.0459 | 5.11% |
| Weight gradients (six) | 0.1384 | 16.43% | 0.1390 | 15.48% |
| Projection/LN/residual | 0.1054 | 12.52% | 0.1086 | 12.09% |
| Gate/output/delta | 0.0782 | 9.28% | 0.0784 | 8.73% |
| Ending dY layout copy | 0.0000 | 0.00% | 0.0508 | 5.65% |
| Bias mask/layout | 0.0078 | 0.93% | 0.0077 | 0.86% |
| LN parameter reductions | 0.0042 | 0.50% | 0.0044 | 0.49% |

Weight-gradient detail, including each GEMM split-K reduction:

| Weight | Starting ms | Ending ms |
|---|---:|---:|
| W_out | 0.0320 | 0.0320 |
| W_q/k/v/gate (shared input) | 0.0773 | 0.0780 |
| W_bias | 0.0290 | 0.0290 |

## L768

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 2.1421 | 46.09% | 2.1363 | 44.10% |
| dQ | 1.0409 | 22.40% | 1.0432 | 21.53% |
| Bias final reduction | 0.3002 | 6.46% | 0.3022 | 6.24% |
| Weight gradients (six) | 0.4591 | 9.88% | 0.4460 | 9.21% |
| Projection/LN/residual | 0.3719 | 8.00% | 0.3817 | 7.88% |
| Gate/output/delta | 0.3086 | 6.64% | 0.3096 | 6.39% |
| Ending dY layout copy | 0.0000 | 0.00% | 0.2002 | 4.13% |
| Bias mask/layout | 0.0168 | 0.36% | 0.0167 | 0.34% |
| LN parameter reductions | 0.0084 | 0.18% | 0.0085 | 0.18% |

Weight-gradient detail, including each GEMM split-K reduction:

| Weight | Starting ms | Ending ms |
|---|---:|---:|
| W_out | 0.1173 | 0.1173 |
| W_q/k/v/gate (shared input) | 0.2679 | 0.2548 |
| W_bias | 0.0739 | 0.0739 |

## L1024

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 4.9418 | 49.40% | 5.0137 | 47.78% |
| dQ | 2.3845 | 23.84% | 2.4164 | 23.03% |
| Bias final reduction | 0.6911 | 6.91% | 0.6900 | 6.58% |
| Weight gradients (six) | 0.7577 | 7.57% | 0.7606 | 7.25% |
| Projection/LN/residual | 0.6433 | 6.43% | 0.6683 | 6.37% |
| Gate/output/delta | 0.5476 | 5.47% | 0.5479 | 5.22% |
| Ending dY layout copy | 0.0000 | 0.00% | 0.3572 | 3.40% |
| Bias mask/layout | 0.0257 | 0.26% | 0.0259 | 0.25% |
| LN parameter reductions | 0.0119 | 0.12% | 0.0124 | 0.12% |

Weight-gradient detail, including each GEMM split-K reduction:

| Weight | Starting ms | Ending ms |
|---|---:|---:|
| W_out | 0.2029 | 0.2035 |
| W_q/k/v/gate (shared input) | 0.4366 | 0.4369 |
| W_bias | 0.1182 | 0.1201 |

The six weight gradients now launch six kernels: shared-input Q/K/V/gate
GEMM plus FP32 split reduction, output GEMM plus reduction, bias GEMM
plus reduction. The complete backward has 16/17 kernels (starting/ending),
unchanged from the baseline of this core pass. The complete sequence and all six
parameter gradients remain accounted for. Ending retains its dY copy.

Raw data: `attribution-17211-L*.json`; Chrome traces:
`attribution-17211-L*-e*-trace.json`. The summary JSON asserts that
the category totals equal every kernel in the profile.
