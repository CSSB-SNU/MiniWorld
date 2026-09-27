# CTA scheduling and producer/consumer follow-up

2026-09-21, H100, CUDA/CUTLASS SM90a. This follows [CUDA_REPORT.md](CUDA_REPORT.md).
The attention core is unchanged; L384 still uses the existing Triton attention core.
The changes here concern the CUDA/TMA surrounds.

## Installed prologue

`oc/opt_core/kernels/triattn_surround_tma/prologue_pipeline.cu` implements a
288-thread CTA: two 128-thread WGMMA consumer groups and one TMA producer warp.
The tile is M128/N64/K128, with two weight slots and two output slots. Shared
storage permits two resident CTAs per H100 SM. Each consumer group handles
separate rows and progresses independently after the initial LayerNorm barrier.

The producer loads the input and bias weights, primes two projection weight
slots, refills a weight slot as soon as both consumer groups have finished
reading it, and stores completed output tiles. TMA stores can remain in flight
while consumers work on the next projection. This is real warp specialization;
the producer does not run the LayerNorm or WGMMA computation.

Ownership invariants:

- `ready_w[stage]`: TMA transaction completion releases weights to consumers.
- `empty_w[stage]`: two arrivals, one from each consumer WG after its WGMMA wait,
  release the weight buffer to the producer.
- `ready_out[stage]`: two arrivals after stmatrix, async-proxy fence and each
  group's named barrier release output to the producer.
- `empty_out[stage]`: the producer releases a slot only after its TMA store
  completes (`wait_group 1` releases the previous slot, final `wait_group 0`
  drains the last slot).
- Reused slots alternate barrier phase using `(chunk / 2) & 1`; consumers wait
  for the previous output generation before writing a reused output slot.

The default chooses this prologue at L384 in either direction. Larger shapes
retain the previous faster CUDA prologue. `FPF_TRIATT_PRO_PIPELINE=on` forces
this pipeline for all qualified lengths; `off` selects the previous CUDA
prologue; `auto` is the default. The normal CUDA stack/shape guards still apply.
The native artifact and all compiled sources are hashed in the manifest.

## Persistent epilogue experiments

`cta_pipeline/epilogue_pipeline.cu`, `epilogue_pipeline128.cu` and
`epilogue_split.cu` implement persistent CTAs with two input/output slots,
one TMA input producer warp and a separate TMA output producer warp. Weights
stay in shared memory across tiles. A CTA advances by gridDim.x tiles.
The input producer replaces consumed attention output with the residual while
WGMMA executes. The output producer releases a slot only after its store ends.

The M64 variant uses one consumer WG and two resident CTAs. The M128 variant
uses two consumer WGs but occupies enough shared memory to allow only one CTA.
The N-split M64 variant uses two consumer WGs, 320 threads total, retains two
resident CTAs and uses paired BF16 rounding/addition in its output pack.

A correctness hazard was found during additional random-input validation of
N-split: both WGs read the same input A, so one WG must not recycle A as output
until *both* asynchronous WGMMA groups have finished. A 256-thread barrier after
WGMMA completion fixes the ownership boundary. Initial split/grid timings before
this fix are invalid candidates and must not be used to select a serving kernel.

The persistent epilogue is experimental and is not installed as the default.
The M128 NCU result reduced L2 sectors from 31,416,182 to 28,447,348 (~9.45%),
but time increased from 207.552 to 235.552 us and active warps fell from 22.76%
to 15.74%. Less weight traffic did not compensate for lower occupancy. Sources,
build scripts, benchmarks and sanitizer entry points are retained under
`cta_pipeline/` so the scheduling work is reproducible.

## Measurements and verification

All comparison kernels run against the same native attention core. Benchmarks
check the actual CUPTI kernel names and reject flash_triattn fallback. Timings
are medians of three alternating CUDA graph rounds; whole-block clock variation
is larger than the small prologue gains.

Initial six-case prologue matrix (job 13908), previous CUDA -> independent WGs:

| Direction | L | Previous prologue us | Pipeline us |
|---|---:|---:|---:|
| starting | 384 | 81.08 | 79.05 |
| ending | 384 | 81.47 | 80.14 |
| starting | 768 | 281.09 | 281.83 |
| ending | 768 | 285.28 | 288.46 |
| starting | 1024 | 493.55 | 503.32 |
| ending | 1024 | 504.87 | 514.04 |

Every output was bitwise equal to the previous CUDA implementation, including
Q/K/V/gate/bias and the whole block. This does not mean bitwise equivalence to
the original Triton LayerNorm; its earlier rounding difference is documented
in CUDA_REPORT.md.

Independent prologue WGs passed racecheck and synccheck (job 13907).
Persistent M128 epilogue passed both checks, both directions, separate and aliased
residual output (job 13898). Final installed-artifact and corrected N-split checks
are recorded below.

## Reproduction

From the repository root:

```bash
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/cuda_tma/build_native.py
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/cuda_tma/install_native.py
sbatch runs/triattn_20260921/cta_pipeline/serving.sbatch
sbatch runs/triattn_20260921/cta_pipeline/sanitize_pro_serving.sbatch
sbatch runs/triattn_20260921/cta_pipeline/sanitize_split.sbatch
```

The experimental epilogue extension is built with `cta_pipeline/build_epi.py`.
Older epilogue harness JSON uses `pro_us`/`pro_rounds` for *epilogue* time;
these field names are inherited from the original prologue harness.

## Final installed-artifact checks

Job 13916 passed all six length/direction combinations, with exact Q/K/V/gate/bias
and whole-block equality against the previous CUDA path. Fresh dense, empty and
irregular masks also passed in each case. CUPTI confirmed `auto` selects the
pipeline only at L384 and the original core route remains active.

| Direction | L | Previous prologue us | Forced pipeline us | Previous block us | Auto block us |
|---|---:|---:|---:|---:|---:|
| starting | 384 | 80.63 | 79.64 | 303.21 | 300.74 |
| starting | 768 | 282.38 | 274.57 | 1445.33 | 1456.42 |
| starting | 1024 | 495.78 | 493.91 | 2996.01 | 2984.65 |
| ending | 384 | 81.31 | 79.66 | 304.32 | 304.15 |
| ending | 768 | 286.55 | 281.02 | 1436.56 | 1436.91 |
| ending | 1024 | 499.58 | 501.93 | 2988.28 | 3005.95 |

For L384 the installed pipeline saves roughly 1–2% of prologue time. Whole-block
gains are small and noisy; no large block speedup is claimed. At L768 the isolated
prologue can improve, but the whole-block result is inconsistent, so it is not
selected automatically. At L1024 there is no repeatable gain.

Installed prologue racecheck and synccheck passed with zero errors (job 13917).
Corrected N-split epilogue racecheck and synccheck passed with zero errors for both
directions and both residual aliasing modes (job 13918). The corrected epilogue
also passed exact whole-block and component equality at all six shapes (job 13919,
8 CTAs per SM in the launch grid, at most 2 resident CTAs per SM). It remains
experimental: starting-direction epilogue times were 64.21 -> 65.36 us (L384),
211.09 -> 212.50 us (L768), and 357.31 -> 367.81 us (L1024). Ending-direction
results were approximately tied at the larger shapes, with no reliable gain.

`codex_cta_pipeline.patch` is an incremental source patch over the previous
CUDA/TMA surround implementation. Rebuild and run install_native.py afterward;
the binary and manifest are generated artifacts. The exact previous installed
package is retained at `cta_pipeline/serving_before/`.

NCU job 13925 verified the installed L384 pipeline occupancy: 58 registers per
thread (64 allocated), 102,528 bytes dynamic shared memory plus 1,024 bytes
driver overhead; the shared-memory limit is 2 CTAs per SM, register limit 3,
and barrier limit 4. Theoretical occupancy is 18 warps / 28.125%; measured
active-warps occupancy was 25.77%. The extra producer warp therefore does not
reduce residency below the intended two CTAs.
