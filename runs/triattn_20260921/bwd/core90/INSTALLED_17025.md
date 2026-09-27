# Installed backward core improvements

**The two core kernels have not reached SOL90.** The installed changes reduce
latency and memory traffic; they do not constitute a SOL90 result.

Baseline: installed job16663, including shared-Z Q/K/V/gate weight gradients.
The baseline manifests and every hashed file are frozen in `baseline/`.
Both directions use all input/parameter gradients and all five native fusion flags.

Installed dK/dV: `rs8_tma_barrier`; dQ: `rs_tma_ldmatrix`.
All new device code is native CUDA C++, using TMA and WGMMA.

The current continuation runs only on `node01`, with low `normal_h100` QoS.
All `core90/*.sbatch` defaults reflect that user constraint.

## Full module timing

Twelve balanced AB/BA rounds, fifteen CUDA graph replays per measurement.
Time reductions use the median paired speed ratio; times are arm medians.

| L | Direction | Backward before ms | After ms | Reduction | F+B reduction |
|---:|---|---:|---:|---:|---:|
| 384 | starting | 1.0946 | 0.9151 | 16.47% | 11.38% |
| 384 | ending | 1.1578 | 0.9679 | 16.39% | 10.49% |
| 768 | starting | 6.0650 | 5.0542 | 16.58% | 11.25% |
| 768 | ending | 6.2409 | 5.2643 | 15.83% | 10.84% |
| 1024 | starting | 13.1062 | 11.0053 | 15.91% | 10.59% |
| 1024 | ending | 13.4644 | 11.3936 | 15.44% | 11.04% |

## Measured L768 SOL

NCU `SpeedOfLight`: maximum measured compute/memory throughput. These are
hardware utilization metrics, not ratios against previous execution times.

| Kernel | ms | Compute SOL | Memory SOL | Maximum |
|---|---:|---:|---:|---:|
| dK/dV + bias partial | 2.5824 | 43.77% | 46.23% | 46.23% |
| Bias final reduction | 0.3145 | 4.62% | 86.51% | 86.51% |
| dQ | 1.0346 | 62.08% | 76.83% | 76.83% |

## Changes and correctness

- Use register-source WGMMA operands for BF16 probability/dS. Reuse retired
  resident shared tiles for TMA gradient stores; no extra HBM tensor.
- R8 dK/dV retains deterministic FP32 bias partials. The partial tensor and
  its final reduction reads are each half the previous R4 size. At L768,
  the eliminated write plus read is 1.812 GB. R8 changes FP32 association,
  so bias gradients are not required to be bitwise identical.
- dQ retains the resident Q/dO/stats and double-buffered K/V/bias pipeline.
- Loader metadata records actual row-group8; legacy group4 native calls
  remain accepted for old callers and the A/B qualification harness.
- Final artifacts are rebuilt with NVCC `--objdir-as-tempdir` to isolate
  temporary files across the separate process namespaces used by tools.
- FP64, none/mixed/one-key/all-masked cases, exact bias cancellation,
  memcheck/racecheck/synccheck, all module gradients, SGD updates, frozen
  parameters, fullgraph/cold compilation, opt-outs and AMP are checked.

Bias qualification: job16876. 
dQ qualification: job16864; independent FP64 job16862.
Installed package check: `final-17025.log`. Full timings:
`combined-17025-L*.json`. Current attribution is in `ATTRIBUTION.md`.

## Rejected directions and remaining work

See `EXPERIMENTS.md` and `results.json` for individual jobs. Larger key
tiles, cross-CTA multicast, cooperative grouped-row dQ, extra shared
probability staging, FP32 partial TMA stores and higher forced occupancy
did not improve the selected end-to-end result. Register spills, extra
synchronization and reduced tensor/warp overlap offset their savings.
The core SOL90 target remains open. No forward changes were made.
