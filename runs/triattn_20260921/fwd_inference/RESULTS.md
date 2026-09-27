# Qualified inference-only forward: internal baseline

These timings compare the installed ordinary eval module, not Anthropic's original fused block.
Use [the Anthropic comparison](ANTHROPIC_COMPARISON.md) for the primary external baseline.

Explicit experiment: `resident6` + `front8`; production dispatch is unchanged.

Complete eval/inference_mode module, native CUDA/TMA plus cuBLAS output projection. Nonzero output weights, alternating AB/BA CUDA Graph timings (64 rounds x 40 replays), node02 / normal_h100. Time reduction uses the median paired ratio.

| L | Direction | Baseline ms | Inference ms | Time reduction | Speedup 95% bootstrap CI |
|---:|---|---:|---:|---:|---|
| 384 | starting | 0.3952 | 0.3794 | 3.97% | 1.0411–1.0416 |
| 384 | ending | 0.4911 | 0.3754 | 23.55% | 1.3076–1.3083 |
| 768 | starting | 1.9433 | 1.8119 | 6.58% | 1.0683–1.0829 |
| 768 | ending | 2.3132 | 1.8120 | 21.61% | 1.2689–1.2809 |
| 1024 | starting | 4.0637 | 3.8975 | 3.71% | 1.0310–1.0596 |
| 1024 | ending | 4.7131 | 3.8980 | 17.14% | 1.1974–1.2145 |

Evidence: `qualified-front8-resident6-{384,768,1024}-18488.json`, `results.json`, native/core checks18451 and sanitizers18463. The qualification job also checks the selected front against FP64 and runs memcheck/racecheck/synccheck.

The native core allocates only gated output. The full path has no Q/K/V/gate/LSE or LN-statistic saves. Resident6 still has register spills; source-level tensor removal is not a measurement of actual HBM bytes. No inference SOL90 claim.
