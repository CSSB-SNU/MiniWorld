# Installed backward optimization — job16663

2026-09-23. Native CUDA C++/TMA; H100 80GB, B1 BF16 C128/H4/D32.
Optimize dK/dV, dQ, then weight gradients after full-backward attribution16607.
Gate, projection/LN and final bias reduction remain unchanged under the user's 80% cutoff.
Forward optimization remains closed. Whole-backward SOL80/90 is not reached.

## Installed full-module result

This pass compares against the exact pre-pass installation16580, saved in `baseline/`.
All input and parameter gradients are returned. Both directions, dropout0, every seventh
key masked; 12 balanced AB/BA rounds of 15 CUDA graph replays. Times are median ms;
reductions use median paired ratios, so they need not equal the ratio of marginal medians.
Final installed job16663 follows independent combined-candidate job16661.

| L | Direction | Bwd before | Bwd after | Reduction | F+B before | F+B after | Reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 384 | starting | 1.156 | 1.103 | 4.59% | 1.672 | 1.613 | 3.58% |
| 384 | ending | 1.220 | 1.155 | 5.30% | 1.823 | 1.767 | 3.06% |
| 768 | starting | 6.345 | 6.085 | 4.64% | 8.954 | 8.755 | 3.41% |
| 768 | ending | 6.604 | 6.275 | 4.51% | 9.583 | 9.244 | 3.60% |
| 1024 | starting | 13.773 | 13.267 | 4.09% | 19.364 | 18.822 | 2.69% |
| 1024 | ending | 14.179 | 13.509 | 4.36% | 20.152 | 19.892 | 1.38% |

## What changed

- **dK/dV:** `bias_fusion/ldmatrix_bias` reads bias with two warp-level `ldmatrix.x4.trans` operations per score fragment. This replaces sixteen scalar shared loads without a new shared/HBM buffer or barrier. Q32/K64/R4, TMA producer plus four consumer warpgroups, FP32 bias partials and the final bias reducer are preserved. All isolated outputs and full-module gradients match the previous kernel bitwise. Pilot16652 reduces dK/dV-plus-reducer time by 1.83%, 3.09%, 3.18% at L384/768/1024.
- **Four weight gradients:** `wgrad/shared_z_tensor` fuses Q/K/V/gate dW. A CTA shares one Z tile across four consumer warpgroups, each accumulating its own weight gradient. One producer warpgroup streams Z and four gradient tiles with double-buffered TMA. Tokens are split by 2304/8960/15936 at L384/768/1024, producing 128/132/132 CTAs. FP32 partials are deterministically reduced; BF16 rounding occurs only at final output. No atomics or BF16 partial reduction.
- **Dispatch:** native grouped dW is used only for qualified BF16 contiguous shapes with all four wide weight gradients requested. Partial/frozen parameter sets and unsupported metadata retain the prior GEMMs. Output dW and the narrow bias dW retain cuBLAS. The custom op returns one fresh [4,128,128] allocation; unbind happens outside the opaque boundary.
- **dQ:** no new candidate was adopted. Its installed `vector_bias` CUDA kernel remains unchanged.

Package files: `bias_fusion.cu`, `wgrad.cu`, `wgrad_backward.py`, `build_wgrad.py`,
`ln_backward.py`, versioned `.so` binaries and three updated/new manifests.
`install.py` verifies source/binary hashes, full qualification and this pass's baseline before replacing files.

## Memory and utilization

NCU16659 includes all four former cuBLAS GEMMs and their reducers versus the fused pair. Measured HBM read+write traffic falls **1.398 GB -> 0.775 GB (44.53% less)** at L768, including FP32 partials. The fused main kernel reaches **HBM SOL85.27%**. Isolated final-API pilot16657 gives about35%/42%/45% shorter four-weight-gradient time at L384/768/1024.

NCU16658 dK/dV: memory SOL61.37%, SM51.06%, HBM33.67%; bias final reducer HBM93.95%. Lower instruction count makes dK/dV faster without increasing SM utilization. dQ remains the previous below80 kernel (NCU16557 memory/L2 SOL68.54%, SM58.09%). There is no complete-backward SOL80/90 claim.

[ATTRIBUTION.md](ATTRIBUTION.md) records the installed full-backward kernel accounting, job16664. Kernel counts fall **22 -> 16 starting**, **23 -> 17 ending**. The six parameter gradients now require six kernels rather than twelve. At L768 the remaining dK/dV+dQ work still accounts for about70% of profiled backward time; weight gradients fall to about7%. CUPTI totals and separate event times are reported independently.

## Validation

- dK/dV qualification16653: full-module/masks/dropout/SGD/Inductor/frozen parameters, independent FP64, exact-zero bias cancellation, memcheck/racecheck/synccheck; no errors.
- dW qualification16662: independent FP64 (relative L2 about0.00167 including BF16 output rounding), exact cancellation, all gradients in both directions, masks/dropout/SGD, fullgraph Inductor, partially/all frozen parameters, and three sanitizers; all pass.
- Final installed16663: all six manifest hashes/default flags, custom-op schema/autograd-registration/fake/AOT checks, ten fusion opt-outs, cold first compiled backward, full backward/F+B, BF16 autocast and FP16 native refusal; all pass.
- Combined full-gradient relative L2 stays below0.00038; forward is unchanged. Timing jobs16654/16656 isolate each improvement,16661 combines prototypes,16663 checks actual installed dispatch.

## Rejected candidates

| Candidate | Pilot | L768 speedup vs installed baseline | Disposition |
|---|---:|---:|---|
| bias / shared_scaled_bias | 16643 | 0.8867x | Slower; not installed |
| dq / shared_dp | 16644 | 0.5391x | Slower; not installed |
| bias / inter_iteration | 16648 | 0.9713x | Slower; not installed |
| dq / inter_iteration | 16649 | 0.9657x | Slower; not installed |

The shared-dP dQ experiment retains spills (52B stores/60B loads) and adds on-chip traffic. Shared scaled bias adds an extra CTA barrier and layout conversion. Inter-iteration pipelining is bitwise correct but slower; dK/dV also reports compiler WGMMA serialization. Initial pipeline builds were rebuilt after source changes before any GPU test.
Test-only failures16655/16660 were Python import-registration/name-collision issues, fixed before passing qualification16662. They are not correctness or speed evidence.

## Reproduction

Use `runs/anthropic_adoption_20260919/env.sh` for the Torch/CUDA environment.
GPU scripts are Slurm H100 jobs. `final.sbatch` uses the saved pre-pass baseline
and currently installed candidates; `attribution.sbatch` measures the installed path.
`summarize.py` and `summarize_attribution.py --job 16664` rebuild these reports.
To rebuild package artifacts, run package `build_bias.py` and `build_wgrad.py`
with CUTLASS4.2 available. Rebuilt binaries require fresh performance qualification.
