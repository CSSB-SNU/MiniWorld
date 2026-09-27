# Installed backward core improvements

**The two core kernels have not reached SOL90.** The installed changes reduce
latency and memory traffic; they do not constitute a SOL90 result.

Baseline: installed job16663, including shared-Z Q/K/V/gate weight gradients.
The baseline manifests and every hashed file are frozen in `baseline/`.
Both directions use all input/parameter gradients and all five native fusion flags.

Installed dK/dV: `rs8_producer32`; dQ: `rs_softmax_overlap`.
All new device code is native CUDA C++, using TMA and WGMMA.

The current continuation runs only on `node01`, with low `normal_h100` QoS.
All `core90/*.sbatch` defaults reflect that user constraint.

## Full module timing

Twelve balanced AB/BA rounds, fifteen CUDA graph replays per measurement.
Time reductions use the median paired speed ratio; times are arm medians.

| L | Direction | Backward before ms | After ms | Reduction | F+B reduction |
|---:|---|---:|---:|---:|---:|
| 384 | starting | 1.0983 | 0.9145 | 16.81% | 11.66% |
| 384 | ending | 1.1656 | 0.9698 | 16.74% | 10.70% |
| 768 | starting | 6.0773 | 5.0766 | 16.96% | 10.91% |
| 768 | ending | 6.2560 | 5.2528 | 16.11% | 10.48% |
| 1024 | starting | 13.2732 | 11.0794 | 16.29% | 11.36% |
| 1024 | ending | 13.5322 | 11.4117 | 15.85% | 10.73% |

## Measured L768 SOL

NCU `SpeedOfLight`: maximum measured compute/memory throughput. These are
hardware utilization metrics, not ratios against previous execution times.

| Kernel | ms | Compute SOL | Memory SOL | Maximum |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 2.5473 | 44.11% | 46.29% | 46.29% |
| Bias final reduction | 0.3141 | 4.66% | 86.62% | 86.62% |
| dQ | 1.0223 | 62.62% | 78.09% | 78.09% |

## Changes and correctness

- Use register-source WGMMA operands for BF16 probability/dS. Reuse retired
  resident shared tiles for TMA gradient stores; no extra HBM tensor.
- R8 dK/dV retains deterministic FP32 bias partials. The partial tensor and
  its final reduction reads are each half the previous R4 size. At L768,
  the eliminated write plus read is 1.812 GB. R8 changes FP32 association,
  so bias gradients are not required to be bitwise identical.
- dQ retains the resident Q/dO/stats and double-buffered K/V/bias pipeline.
- The latest follow-up assigns32 registers to the dK/dV producer and
  overlaps dQ probability calculations with its pending dP WGMMA group.
  All reuse barriers remain. The scheduling changes are bitwise equal
  to checkpoint17025 for all full-module input/parameter gradients.
- Loader metadata records actual row-group8; legacy group4 native calls
  remain accepted for old callers and the A/B qualification harness.
- Final artifacts are rebuilt with NVCC `--objdir-as-tempdir` to isolate
  temporary files across the separate process namespaces used by tools.
- FP64, none/mixed/one-key/all-masked cases, exact bias cancellation,
  memcheck/racecheck/synccheck, all module gradients, SGD updates, frozen
  parameters, fullgraph/cold compilation, opt-outs and AMP are checked.

Bias qualification: job17049. 
dQ qualification: job17055; independent FP64 job17056.
Installed package check: `final-17069.log`. Full timings:
`combined-17069-L*.json`. Current attribution is in `ATTRIBUTION.md`.

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
