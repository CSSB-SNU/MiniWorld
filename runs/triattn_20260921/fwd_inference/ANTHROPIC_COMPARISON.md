# Inference FWD versus the original Anthropic implementation

Job 18528 completed on node02 / normal_h100. Our resident6 + front8 wins at L384, but loses at L768 and L1024. Earlier checkpoint18488 gains were against the installed ordinary eval module, not the original Anthropic fused block.

B1, C128, H4, D32, BF16, eval/inference_mode. Same inputs and nonzero random weights; key mask disables every seventh key. Complete LN + bias + QKV + attention + gate + output projection + residual, including ending orientation. Weight packing and compilation are outside timing. Alternating AB/BA CUDA Graph measurements use 64 rounds x 40 replays.

The primary table takes the faster measured Anthropic residual form per cell. Both forms preserve the caller input. Negative time change means our kernel takes less time. Percentages use median paired ratios; times are separate medians. Bootstrap intervals resample the 64 paired ratios 5,000 times and describe this run, not variability across jobs.

| L | Direction | Anthropic ms | Ours ms | Our time change | Speedup 95% CI | Anthropic form |
|---:|---|---:|---:|---:|---|---|
| 384 | starting | 0.4547 | 0.3706 | -18.51% | 1.2267–1.2275 | copy_plus_fused_residual |
| 384 | ending | 0.4634 | 0.3704 | -19.84% | 1.2462–1.2479 | copy_plus_fused_residual |
| 768 | starting | 1.7480 | 1.8165 | +3.71% | 0.9599–0.9680 | copy_plus_fused_residual |
| 768 | ending | 1.7275 | 1.8177 | +5.40% | 0.9423–0.9530 | copy_plus_fused_residual |
| 1024 | starting | 3.5258 | 3.9095 | +10.84% | 0.8992–0.9039 | update_plus_add |
| 1024 | ending | 3.4550 | 3.9199 | +13.55% | 0.8790–0.8821 | copy_plus_fused_residual |

## Residual contract and all measurements

`update_plus_add` runs the original block with residual=False and adds the input. `copy_plus_fused_residual` copies the input into preallocated work storage and calls the original block with residual=True. The copy is timed: upstream fused residual overwrites its input, while our entry returns a new output. The two upstream forms return identical outputs on all tested fixtures. This is not a timing of the upstream destructive API alone. L1024 starting has effectively tied upstream forms; the observed lower median selects update_plus_add.

| L | Direction | Anthropic form | Anthropic ms | Ours ms | Paired speedup |
|---:|---|---|---:|---:|---:|
| 384 | starting | update_plus_add | 0.4629 | 0.3708 | 1.2486 |
| 384 | starting | copy_plus_fused_residual | 0.4547 | 0.3706 | 1.2272 |
| 384 | ending | update_plus_add | 0.4954 | 0.3707 | 1.3354 |
| 384 | ending | copy_plus_fused_residual | 0.4634 | 0.3704 | 1.2475 |
| 768 | starting | update_plus_add | 1.7586 | 1.8162 | 0.9677 |
| 768 | starting | copy_plus_fused_residual | 1.7480 | 1.8165 | 0.9642 |
| 768 | ending | update_plus_add | 1.8579 | 1.8100 | 1.0189 |
| 768 | ending | copy_plus_fused_residual | 1.7275 | 1.8177 | 0.9487 |
| 1024 | starting | update_plus_add | 3.5258 | 3.9095 | 0.9022 |
| 1024 | starting | copy_plus_fused_residual | 3.5271 | 3.9086 | 0.9015 |
| 1024 | ending | update_plus_add | 3.6709 | 3.9183 | 0.9364 |
| 1024 | ending | copy_plus_fused_residual | 3.4550 | 3.9199 | 0.8806 |

## Actual dispatch and provenance

Original `oc-release/opt_core`, `impl=fpf`, `core=default`, `ln=fused`; no tuned cell overrides. Default L384 selects flash_triattn (Triton). Default L768/L1024 selects tier:fast -> triattn_native (CUDA M1). Served-core counters and CUDA profiles confirm these paths; the L384 Triton selection is the intended default, not a refusal fallback. Surrounds use the original fused prologue and epilogue. Our profile confirms front8, resident6, cuBLAS output projection and residual. Some first-call profiler events are missing; every stage is present in subsequent calls. Profiles establish dispatch, not timing totals.

`anthropic-provenance.json` verifies imported opt_core sources, original surrounds, cell table, native router and M1 source hashes against the upstream checkout. The native M1 binary matches its source/build manifest: original source rebuilt for Torch2.10.0+cu128 with NVCC12.9, not a claim of the originally distributed binary. The benchmark JSON field `native_release_install` belongs to unused cuda_sm90a and is not evidence for the selected M1 core.

## Numerical scope

Upstream internally rounds LN affine values to BF16. Both paths use the same nontrivial BF16-representable values held in FP32 LN parameters, avoiding different effective weights. Initial job18527 used arbitrary FP32 affine values and stopped at a one-key tolerance failure; it supplies no reported performance result. No kernels were changed to address it.

Mixed, dense and one-key fixtures pass relative-L2 output <0.003 and update <0.025 in both directions at all three lengths. These are tolerance comparisons, not bitwise equivalence or a new independent FP64 qualification. The earlier independent FP64, graph and sanitizer qualification remains linked in [README](README.md).

**All-masked semantics differ:** our entry returns the input (zero update); Anthropic uses uniform-mean-V attention and produces a nonzero update on these fixtures. This difference is recorded explicitly, so the implementations are not interchangeable for all masks. Timed inputs always contain valid keys.

Evidence: [raw summary](anthropic-results.json), [provenance](anthropic-provenance.json), `anthropic-{384,768,1024}-18528.json`, [benchmark](anthropic_compare.py), [Slurm script](anthropic_compare.sbatch). Reproduce this report with `python3 summarize_anthropic.py --job 18528`. No production dispatch change or inference SOL90 claim.
