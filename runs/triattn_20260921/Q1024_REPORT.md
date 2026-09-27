# L1024 CUDA/TMA core specialization

2026-09-22, H100 80GB, C128/H4/D32 BF16. **SOL90 remains unachieved.**

Job 14862 installs a separate hot flag 536870912 for square N=S1024, H4 and the usual
scale. It uses the qualified Q cache with twelve registers, independent K/V TMA
producers, constant descriptors, N40 PV/denominator fusion and exact grouped
FFMA/EX2. Row clamps handle the final partial group of three pair rows.
All floating-point operations and probability rounding retain their order.
Other scales retain hot flag 0. L768 retains hot flag 1073741824; SAFE remains flag 1024.

Job 14851 used 16 balanced paired graph rounds, cloning each captured output
before another graph could overwrite it. Full core/block outputs are bitwise
equal. Percentages use paired ratios, rather than ratios of timing medians.

| Direction | Previous core us | Current core us | Paired core reduction | Paired block reduction |
|---|---:|---:|---:|---:|
| starting | 1966.960 | 1879.503 | 4.163% | 2.444% |
| ending | 1989.822 | 1905.414 | 3.971% | 2.494% |

The baseline is the preceding 4be package, saved in
`core_sol90/before_q1024_install/`. L768's actual machine words are unchanged;
the preceding Q cache change's 0.45% gain remains in that path.

Final package 14856 passes 20 full N1024/B2,20 full N768/B2,40 generic and 2
custom-scale bitwise cases. Its machine words for all four flags exactly match
the measured prototype; hot flag 0, SAFE flag 1024 and L768 hot flag 1073741824 exactly match4be.
Both hot kernels compile with 128 initial registers, zero spills and no WGMMA
serialization warning; C7519 inserted fences remain.

FullN1024/B1 strided racecheck, synccheck and memcheck each pass five patterns
(dense, prefix, empty batch, late seed, forced SAFE) with zero hazards/errors.
Installed job 14862 confirms both directions, both lengths, five masks,
bitwise default/opt-out outputs, expected native flags and no flash fallback.
All 17 shipped tests pass. Expected vectors and the separate generic M1 binary
were not regenerated.

Installed NCU job 14862 uses separate kernel profiling runs, not paired graph
timing. Do not infer incremental speedups from different NCU jobs.

| Length | Kernel us | SM SOL | Tensor | Occupancy |
|---|---:|---:|---:|---:|
| 768 | 795.904 | 61.349% | 34.966% | 22.296% |
| 1024 | 1751.712 | 65.249% | 37.068% | 22.523% |

Installed SHA256: `9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225`.
Measurements and validation: [q1024-results.json](core_sol90/q1024-results.json),
`core_sol90/serving-q1024-*.json` and `installed-q1024-L*-profile.*`.
Rebuild commands and cumulative history: [CORE_SOL90_REPORT.md](CORE_SOL90_REPORT.md).
