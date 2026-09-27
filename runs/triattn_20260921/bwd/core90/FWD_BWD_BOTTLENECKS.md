# Current training forward and backward bottlenecks

> Historical attribution17241, before the training-forward change. [Installed CUDA FWD17310](../../fwd_training/README.md) supersedes the forward dispatch and forward timings below. The backward binaries and their attribution remain checkpoint17211.

B1 BF16 C128/H4/D32, dropout0, every seventh key masked, node02 / normal_h100.
Forward: fresh installed training-path attribution17241. Backward: installed attribution17211.
Percentages use each profile's complete kernel-time sum; CUDA-event totals are recorded separately.
No kernel or dispatch was changed for this diagnosis.

## Dispatch distinction

The backward benchmark's training module uses Front.forward -> layernorm plus five F.linear calls,
AttentionGate.forward -> Triton _tri_attn_fwd/_attn_fwd, gate/output projection, layout copies and residual add.
The previously optimized separate CUDA forward block is not selected by this training path.
The historical inference/block forward timing must not be substituted for these measurements.

## L384 training forward

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| Attention core (Triton) | 0.2483 | 49.03% | 0.2460 | 40.96% |
| Q/K/V/gate/bias projections | 0.1401 | 27.67% | 0.1382 | 23.00% |
| LayerNorm | 0.0270 | 5.33% | 0.0260 | 4.33% |
| Gate/output projection | 0.0428 | 8.46% | 0.0427 | 7.11% |
| Layout copies | 0.0029 | 0.57% | 0.1028 | 17.11% |
| Residual add | 0.0376 | 7.43% | 0.0373 | 6.22% |
| Mask and other | 0.0077 | 1.51% | 0.0076 | 1.27% |

Event total: 0.5212 / 0.6123 ms. Profile kernel sums: 0.5064 / 0.6006 ms.

## L768 training forward

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| Attention core (Triton) | 1.5854 | 62.80% | 1.5859 | 54.48% |
| Q/K/V/gate/bias projections | 0.5199 | 20.59% | 0.5196 | 17.85% |
| LayerNorm | 0.1024 | 4.06% | 0.1016 | 3.49% |
| Gate/output projection | 0.1511 | 5.99% | 0.1508 | 5.18% |
| Layout copies | 0.0062 | 0.25% | 0.3949 | 13.57% |
| Residual add | 0.1479 | 5.86% | 0.1463 | 5.03% |
| Mask and other | 0.0117 | 0.46% | 0.0118 | 0.41% |

Event total: 2.5409 / 2.9100 ms. Profile kernel sums: 2.5245 / 2.9109 ms.

## L1024 training forward

| Stage | Starting ms | Share | Ending ms | Share |
|---|---:|---:|---:|---:|
| Attention core (Triton) | 3.7462 | 69.27% | 4.1784 | 63.36% |
| Q/K/V/gate/bias projections | 0.9312 | 17.22% | 0.9506 | 14.41% |
| LayerNorm | 0.1809 | 3.35% | 0.1803 | 2.73% |
| Gate/output projection | 0.2644 | 4.89% | 0.2674 | 4.06% |
| Layout copies | 0.0098 | 0.18% | 0.7438 | 11.28% |
| Residual add | 0.2606 | 4.82% | 0.2590 | 3.93% |
| Mask and other | 0.0147 | 0.27% | 0.0157 | 0.24% |

Event total: 5.3449 / 5.9953 ms. Profile kernel sums: 5.4078 / 6.5951 ms.

## Backward and utilization

Current [backward attribution](ATTRIBUTION.md) accounts for every16/17 kernel.
L768 starting: dK/dV+bias partial2.1421ms (46.09%); dQ1.0409ms (22.40%);
six weight gradients0.4591ms (9.88%); projection/LN0.3719ms (8.00%);
bias final reduction0.3002ms (6.46%). The two target cores together take68.49%.

Current native profiles17205/17191 distinguish HBM from overall memory SOL:
| Core | HBM throughput | L2 throughput | Active warp occupancy |
|---|---:|---:|---:|
| dK/dV | 35.52% | 45.53% | 18.36% |
| dQ | 40.51% | 78.58% | 24.46% |

The dK/dV profile has one resident CTA, low occupancy and exposed barrier/latency costs;
its depth4 prefetch improvement is consistent with better latency hiding. dQ puts substantially
more pressure on L2/request throughput. Neither core currently saturates HBM bandwidth.
Forward attribution locates the dominant stage; this run does not establish a hardware-unit
roof for its different Triton attention kernel. Historical CUDA-core SOL does not apply.
