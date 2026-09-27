# Backward optimization below SOL80 — 2026-09-23

> **Superseded installation:** [../priority_pass/README.md](../priority_pass/README.md) records the current installed dK/dV and joint weight-gradient update16663. This file preserves the preceding installation16580 and its own baseline.

> Whole-module kernel time attribution: [ATTRIBUTION.md](ATTRIBUTION.md),
> measured after this installation in job16607. At L768, dK/dV plus dQ consume
> about68–69% of kernel time; the six weight gradients about10%. All L384/768/1024
> starting/ending stages and individual weight-gradient times are included.

User accepts kernels above 80% for now. Gate/output/delta and final bias
reduction are retained. Forward optimization remains closed. Three CUDA C++
changes are installed: `bias_fusion/vector_bias`, `dq/vector_bias`, and
`ln_residual/warp_packed`. Full backward SOL80/SOL90 is not claimed.

The baseline snapshot in `baseline/` is the installation immediately before
this pass: q32_combo_small dK/dV, three_cta dQ, split_norm projection/LN, and
n64_staged gate. This differs from the older q32_four baseline in
`../SOL90_PROGRESS.md`; do not add the percentages from those two reports.

## What changed

- dK/dV: read adjacent BF16 dS values with one 32-bit shared-memory load, then
  independently sum the two FP32 bias partials in the original row order.
  No precision reduction or partial-layout change. All three outputs match
  the previous installation bitwise. Producer/consumer grouping, FP32 partial
  capacity, TMA inputs, and the already-fast final bias reducer are retained.
- dQ: read adjacent BF16 bias values with one 32-bit shared load. Preserve
  score arithmetic, masks, double-buffered TMA, and three resident CTAs.
  Output matches the previous installation bitwise.
- Projection dgrad + LN backward + residual: use one 32-thread producer and
  one 128-thread consumer warpgroup. After WGMMA readers retire, reuse the
  two B stages for X and residual, and the retired A stages for rounded dZ.
  After LN reads X, reuse that tile for final dX and issue a TMA store with
  the correct starting/ending strides. Shared memory falls from 89,088 to
  56,320 bytes; 90 registers, no spills, four resident CTAs instead of two.
  Pair BF16 bias epilogue and LN shared reads/stores. Retain BF16 dZ and
  branch-gradient rounding. Feature-sum reassociation changes input gradients
  by about 8e-6 relative L2 in isolated tests; both LN parameter gradients
  remain bitwise equal. No numerical tolerance was relaxed.

No new full-size HBM intermediate is introduced. Norm-parameter scratch and
its existing two-stage reducer are retained. Q/K/V/gate/bias/output weight
gradients still use their existing cuBLAS operations.

## Individual paired timings

H100 80GB, BF16, 12 balanced AB/BA rounds of 20 warmed CUDA graph replays.
Reductions use median paired ratios, not the ratio of independently reported
medians. The dK/dV stage includes the unchanged final bias reduction; dQ is
isolated; LN includes its two parameter reducers. These are not whole-module
times and should not be summed across independent runs.

| Stage | L384 time reduction | L768 time reduction | L1024 time reduction |
|---|---:|---:|---:|
| dK/dV + bias reduction | 2.36% | 3.48% | 3.55% |
| dQ | 6.25% | 8.20% | 6.85% |
| Projection/LN/residual, starting | 41.59% | 48.21% | 49.89% |
| Projection/LN/residual, ending | 45.06% | 51.34% | 52.59% |

Evidence: `pilot-16545-L*.json`, `pilot-16550-L*.json`,
`pilot-16572-L*.json`.

## NCU hardware SOL at L768

Individual warmed graph replays, cache-control none, clock-control none.
Columns are hardware peak utilization, not a measured-copy ceiling. Lower
instruction count can reduce latency without raising SM SOL proportionately.

| Kernel | Time ms | SM SOL | HBM SOL | Disposition |
|---|---:|---:|---:|---|
| Grouped dK/dV + bias partial | 3.110912 | 53.02% | 32.33% | Faster; below80 |
| dQ | 1.294976 | 58.09% | 31.82% | Faster; below80 |
| Projection/LN/residual, starting | 0.375328 | 48.57% | **84.94%** | Above80; stop tuning |
| Projection/LN/residual, ending | 0.384576 | 48.32% | **82.84%** | Above80; stop tuning |
| Norm parameter partial reduction | 0.007520 | 4.38% | 37.68% | Unchanged small stage |
| Norm parameter final reduction | 0.003456–0.003712 | 0.015–0.016% | 0.166–0.176% | Unchanged small stage |
| Final bias reduction | 0.576288 | 4.38% | 94.17% | Unchanged; above80 |
| Gate/output/delta | 0.308896 | 24.51% | 86.41% | Unchanged, earlier NCU16325 |

Evidence: `profile-16553.csv`, `profile-16557.csv`, `profile-16575.csv`, and
`../gate_delta/profile-16325.csv`. The tiny norm reducers take about 11us total
in NCU; their low hardware utilization is not a percentage of remaining
whole-backward speedup. Separate cuBLAS SOL has not been measured in this pass.

## Correctness and installation

- dK/dV qualification16552: mixed/all-masked full-module gradients, dropout,
  SGD updates, frozen parameters, fullgraph Inductor, independent FP64/mask
  comparisons, exact zero-sum bias cancellation test, and memcheck/racecheck/
  synccheck all pass. Full-module outputs/gradients equal baseline bitwise.
- dQ qualification16556: same module/compile/optimizer and three sanitizers
  pass; full-module outputs/gradients equal baseline bitwise. Independent
  FP64 none/mixed/one-key/all-masked cases at L64/128 pass16577.
- LN qualification16574: same module/compile/optimizer and three sanitizers
  pass; maximum module relative L2 is 3.36e-6. Independent FP64 checks at
  L8/16/64 in both directions, including scalar residual fallback, pass16577.
- Combined preinstallation test16576 keeps all five fusion flags enabled and
  compares all input/parameter gradients at L384/768/1024 in both directions.
- `install.py` copies only the three qualified CUDA sources and versioned
  binaries, preserving current Python dispatch and rebuild APIs. Existing
  manifests must match the saved baseline before replacement. Hash manifests
  are replaced last; mapped old binaries are not overwritten.

`results.json` contains individual experiment records and installed timings;
`installation.json` contains the three installed manifests.

## Actual installed verification16580

Job16580 completed with exit0. All five package manifests match their hashes;
all five fusion flags remain enabled. Fresh eager and compiled first backward,
ten opt-out/metadata checks, BF16 autocast and FP16 dispatch refusal pass.
Paired12-round x15-replay timings below use the actual installed package and
the saved immediate pre-pass baseline; all input/parameter gradients are checked.
B1 BF16 C128/H4/D32, dropout0, every seventh key masked. Times are ms.

| L | Direction | Bwd before | Bwd after | Reduction | F+B before | F+B after | Reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 384 | starting | 1.269 | 1.158 | 8.85% | 1.783 | 1.664 | 6.67% |
| 384 | ending | 1.348 | 1.210 | 10.10% | 1.949 | 1.820 | 6.63% |
| 768 | starting | 6.875 | 6.316 | 8.20% | 9.404 | 8.905 | 5.84% |
| 768 | ending | 7.132 | 6.531 | 8.32% | 10.052 | 9.471 | 5.49% |
| 1024 | starting | 14.751 | 13.656 | 7.17% | 20.105 | 19.303 | 4.02% |
| 1024 | ending | 15.229 | 14.080 | 7.51% | 21.256 | 20.096 | 5.20% |

Evidence: `final-16580.log`, `combined-16580-L*.json`,
`../sol90_installed/guards-16580.json`, `../sol90_installed/amp-correctness-16580.json`.
No owned build or GPU experiment remains running.


## Rejected refinements

| Candidate | Observation | Decision |
|---|---|---|
| dK/dV parallel_tma16544 | Four producer warps, 60-byte spills; -0.33% at384, +0.18% at768, +0.17% at1024 | No material gain |
| dK/dV parallel_static16551 | Constant row specialization increases spills; roughly -0.65%/+0.15%/+0.22% | Rejected |
| dK/dV parallel_vector16558 | About1.85–2.89%, less than vector_bias; spills remain | Rejected |
| dK/dV vector_stream16561 | Cache-streaming partial stores yield only0.97–2.13% | Rejected |
| dK/dV vector_stats16578 | LSE/delta paired loads roughly tie vector_bias | No adoption |
| dQ warp4 | 160 threads, four-CTA target spills156/96 bytes | Not launched |
| dQ warp3 16570 | Spill-free, large shapes improve less than vector_bias | Rejected |
| LN staged_epilogue / staged_store | Initial256-thread three-CTA target cannot compile WGMMA within80 initial registers | Replaced by160-thread producer-warp design |
| LN warp_store4 16564/16568/16569 | Bitwise-correct and40–45% faster at large shapes, HBM69–72% | Superseded by warp_packed |
| LN warp_store3 | Same90-register resource result as four-CTA target | No separate GPU run |

Job16548 is a harness loader-name error, with no dQ measurement; corrected
job16550 is the evidence. NCU CSV exports log a Python utf-8-sig initialization
warning in this environment but produce valid kernel rows; corresponding
Slurm jobs exit0.
