
## Completed sanitizer and NCU gates

Job 19252 completed with exit code 0. Norm and fused LNLinear each pass memcheck,
racecheck, and synccheck: **0 errors / 0 hazards / 0 warnings** in sanitizer
summaries. The earlier API-only invalid-handle report disappears with extension
preloading and eager CUDA module loading; no error-report suppression was used.

Final NCU jobs 19263_0..2 all completed with exit code 0. These are profiler
measurements, not CUDA-graph benchmark timings. DRAM throughput percentages are
not an application-level SOL claim.

| Workload | Kernel | Profile time ms | DRAM % | L2 % | Registers/thread |
|---|---|---:|---:|---:|---:|
| LN M147456/D128 | forward_vec | 0.02797 | 62.51 | 71.88 | 48 |
| LN M147456/D128 | backward_fused | 0.04672 | 66.70 | 71.86 | 96 |
| LN M147456/D128 | affine_finish | 0.00531 | 3.34 | 23.13 | 18 |
| RMS M147456/D384 | forward_vec | 0.07818 | 79.67 | 79.73 | 39 |
| RMS M147456/D384 | backward_fused | 0.12502 | 78.99 | 77.21 | 80 |
| RMS M147456/D384 | affine_finish | 0.00733 | 7.30 | 40.96 | 18 |
| LN M2048/D4096 | wide_forward | 0.02451 | 22.81 | 40.22 | 62 |
| LN M2048/D4096 | wide_backward | 0.03565 | 40.85 | 55.65 | 97 |
| LN M2048/D4096 | finish | 0.01203 | 41.85 | 45.77 | 25 |

The D128 affine finishing kernel fell from 0.00947 ms in the intermediate
warp-partial design (job 19160) to 0.00531 ms after CTA-local reduction.
Its main backward reached 66.70% of profiled sustained DRAM throughput; this
work does not establish a SOL90 result.
