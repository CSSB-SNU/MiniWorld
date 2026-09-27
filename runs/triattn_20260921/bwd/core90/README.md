# Installed backward core improvements

> **Later bounded experiment,19266/19268/19277:** [Q+dO register reuse](../reuse_20260926/README.md) passes full qualification and improves dQ about1.8% at768/1024. Complete backward improves0.10–0.66%; L1024 F+B improves0.22–0.36%. This remains an explicit process-local experiment; the installed pair documented below is unchanged.

> This node02 continuation is complete. [Direct comparison with the starting installation](NODE02_CONTINUATION.md) shows7.05–7.90% additional full-backward reduction; all13 owned jobs finished. The campaign table below uses the separate original16663 baseline.

**The two core kernels have not reached SOL90.** The installed changes reduce
latency and memory traffic; they do not constitute a SOL90 result.

Baseline: installed job16663, including shared-Z Q/K/V/gate weight gradients.
The baseline manifests and every hashed file are frozen in `baseline/`.
Both directions use all input/parameter gradients and all five native fusion flags.

Installed dK/dV: `rs8_async_q4`; dQ: `rs_softmax_overlap`.
All new device code is native CUDA C++, using TMA and WGMMA.

The current continuation runs on `node02`, with low `normal_h100` QoS.
All `core90/*.sbatch` defaults reflect that user constraint.

## Full module timing

Twelve balanced AB/BA rounds, fifteen CUDA graph replays per measurement.
Time reductions use the median paired speed ratio; times are arm medians.

| L | Direction | Backward before ms | After ms | Reduction | F+B reduction |
|---:|---|---:|---:|---:|---:|
| 384 | starting | 1.1014 | 0.8456 | 23.23% | 16.00% |
| 384 | ending | 1.1654 | 0.8989 | 22.85% | 14.74% |
| 768 | starting | 6.0846 | 4.6847 | 22.83% | 15.29% |
| 768 | ending | 6.2671 | 4.9198 | 22.05% | 15.02% |
| 1024 | starting | 13.1315 | 10.2615 | 22.00% | 14.72% |
| 1024 | ending | 13.6450 | 10.6154 | 21.92% | 14.63% |

## Measured L768 SOL

NCU `SpeedOfLight`: maximum measured compute/memory throughput. These are
hardware utilization metrics, not ratios against previous execution times.

| Kernel | ms | Compute SOL | Memory SOL | Maximum |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 2.0706 | 54.00% | 55.70% | 55.70% |
| Bias final reduction | 0.2912 | 5.02% | 93.47% | 93.47% |
| dQ | 1.0169 | 62.64% | 78.62% | 78.62% |

## Changes and correctness

- Use register-source WGMMA operands for BF16 probability/dS. Reuse retired
  resident shared tiles for TMA gradient stores; no extra HBM tensor.
- R8 dK/dV retains deterministic FP32 bias partials. The partial tensor and
  its final reduction reads are each half the previous R4 size. At L768,
  the eliminated write plus read is 1.812 GB. R8 changes FP32 association,
  so bias gradients are not required to be bitwise identical.
- dQ retains the resident Q/dO/stats and double-buffered K/V/bias pipeline.
- dQ overlaps probability calculations with its pending dP WGMMA group.
  All reader barriers remain in place.
- Q/dO/stats use four shared prefetch stages, eliminating local spills
  and reducing TMA waits without introducing another HBM tensor.
- Two otherwise idle producer warps reduce the eight shared dS tiles
  while the consumers compute the next query tile. Separate ready/empty
  barriers protect both dS stages, including every reducer reader.
  FP32 association and HBM partial volume are unchanged from R8.
  All full-module gradients are bitwise equal to checkpoint17175.
  The producer/consumer register budgets are56/224 per thread.
- Loader metadata records actual row-group8; legacy group4 native calls
  remain accepted for old callers and the A/B qualification harness.
- Final artifacts are rebuilt with NVCC `--objdir-as-tempdir` to isolate
  temporary files across the separate process namespaces used by tools.
- FP64, none/mixed/one-key/all-masked cases, exact bias cancellation,
  memcheck/racecheck/synccheck, all module gradients, SGD updates, frozen
  parameters, fullgraph/cold compilation, opt-outs and AMP are checked.

Bias qualification: job17203. 
dQ qualification: job17055; independent FP64 job17056.
Installed package check: `final-17211.log`. Full timings:
`combined-17211-L*.json`. Current attribution is in `ATTRIBUTION.md`.

## Rejected directions and remaining work

See `EXPERIMENTS.md` and `results.json` for individual jobs. Larger key
tiles, cross-CTA multicast, cooperative grouped-row dQ, extra shared
probability staging, FP32 partial TMA stores and higher forced occupancy
did not improve the selected end-to-end result. Register spills, extra
synchronization and reduced tensor/warp overlap offset their savings.
Two R4 CTAs combining dS through DSM also took53-60% longer in isolation
at the target lengths despite keeping the same R8 global partial size.
TMA eviction hints retaining K/V or bias gave no consistent dQ gain.
The core SOL90 target remains open. No forward changes were made.
