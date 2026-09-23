# H100 cache inventory before the seven-GPU build — 2026-09-15

## Scope

Live scan of MiniWorld's `.pixi/envs/cu128` engine after all six patches.
GPU key: `NVIDIA H100 80GB HBM3 (sm90)`.
Native identity: `998c938247b33f7711689559fc005de9c395f82037f39234634ba1333b6c62fc`.
This inventories tuning JSON, not Triton/Quack/Inductor compiled binaries.
A separate engine checkout or interpreter can have different source and caches.

## Findings

- 73 existing files: 72 Triton and 1 native; all pass the live source/revision/
  key-scheme/environment checks. No stale file was found.
- Triton: 1,568 existing keys. 1,257 have search records covering the current
  full declared grid for every matching stored implementation profile. 311 keys
  across 16 kernels lack that full search evidence. There are no keys whose
  search records are entirely unknown. All 72 declared `developed=yes` Triton
  kernels have an H100 file.
- This is **existing-key search coverage**, not full model workload coverage.
  Pruning, prediction or interrupted/incremental builds can leave candidates
  without measured search records. A missing record does not establish that a
  candidate is valid, faster, or even executable for that workload. Stored
  finite winners remain usable; do not discard them to fill the gaps.
- 56 Triton files have complete current-grid evidence for every existing key;
  this still does not prove all possible runtime keys exist.
- Native TriMul input projection: 9 exact keys fully measured, each against all
  18 declared candidates (162 pairs), with top 5 retained. Scope is BF16,
  D128, batch1, L128/384/768, inference/single/bidirectional training. Other
  lengths, widths and layouts are not certified by those nine entries.
- 12 other routine-build native kernels have **no H100 tuning JSON**. One more
  native experiment is deliberately excluded from routine builds.
- New packed contraction cuBLAS/Quack selection is a fixed measured policy;
  it does not have a new engine tuning JSON or a declared exhaustive Quack tile
  search. Quack's compilation artifacts are a separate cache.

## Native inventory

| Kernel | H100 tuning state | Routine driver eligibility |
| --- | --- | --- |
| `layernorm_bwd_split_cuda` | No file | Included; dispatch may be conditional/experimental |
| `layernorm_fwd_cuda` | No file | Included; dispatch may be conditional/experimental |
| `layernorm_linear_fwd_sm90_cute` | No file | Included; dispatch may be conditional/experimental |
| `trimul_outproj_gemm_gate_sm90_cute` | No file | Included; dispatch may be conditional/experimental |
| `layernorm_linear_fwd_foldstats_sm90_cute` | No file | Included; dispatch may be conditional/experimental |
| `transition_swiglu_fwd_sm90_cute` | No file | Included; dispatch may be conditional/experimental |
| `transition_gate_bwd_sm90_cute` | No file | Included; dispatch may be conditional/experimental |
| `transition_bwd_dx_sm90_cute` | No file | Included; dispatch may be conditional/experimental |
| `layernorm_linear_bwd_dx_sm90_cute` | No file | Included; dispatch may be conditional/experimental |
| `transition_fwd_b2b_sm90_cuda` | No file | Included; dispatch may be conditional/experimental |
| `transition_expand_gate_sm90_cuda` | No file | Excluded (`developed=no`) |
| `transition_bwd_gate_sm90_cuda` | No file | Included; dispatch may be conditional/experimental |
| `trimul_inproj_masked_sm90_cute` | 9 measured keys | Included; dispatch may be conditional/experimental |
| `transition_squeeze_residual_sm90_cute` | No file | Included; dispatch may be conditional/experimental |

## Triton keys without full current-grid search evidence

The denominator below is existing keys, not newly derived model requirements.

| Kernel | Fully searched existing keys | Keys requiring further search/eligibility assessment |
| --- | ---: | ---: |
| `adaln_bwd_pre_dx_triton` | 10 | 22 |
| `adaln_fwd_gate_triton` | 0 | 12 |
| `adaln_gemm_gate_triton` | 0 | 27 |
| `augmented_attention_bwd_split_triton` | 24 | 6 |
| `cond_transition_bwd_gemm_swiglu_triton` | 3 | 11 |
| `cond_transition_fwd_b2b_saveact_triton` | 4 | 4 |
| `cond_transition_fwd_b2b_triton` | 4 | 4 |
| `gated_projection_bwd_dx_triton` | 0 | 26 |
| `layernorm_linear_fwd_triton` | 0 | 12 |
| `rmsnorm_adamod_bwd_triton` | 18 | 36 |
| `rmsnorm_adamod_fwd_triton` | 17 | 37 |
| `transition_bwd_swiglu_recompute_triton` | 0 | 54 |
| `transition_expand_swiglu_triton` | 0 | 30 |
| `transition_fwd_b2b_triton` | 0 | 6 |
| `trimul_outproj_bwd_gate_recompute_triton` | 0 | 12 |
| `trimul_outproj_gemm_gate_triton` | 0 | 12 |

## Triton files with complete search evidence for their existing keys

- `adaln_bwd_dx_dlnw_triton`: 8 keys
- `adaln_epilogue_saveact_triton`: 20 keys
- `adaln_epilogue_triton`: 24 keys
- `adaln_fwd_triton`: 8 keys
- `augmented_attention_bwd_pre_triton`: 32 keys
- `augmented_attention_bwd_reduce_triton`: 9 keys
- `augmented_attention_fwd_triton`: 32 keys
- `cond_transition_bwd_swiglu_flat_triton`: 6 keys
- `cond_transition_expand_swiglu_saveact_triton`: 27 keys
- `cond_transition_expand_swiglu_triton`: 12 keys
- `cond_transition_squeeze_gate_saveact_triton`: 27 keys
- `cond_transition_squeeze_gate_triton`: 12 keys
- `cond_transition_swiglu_triton`: 12 keys
- `gated_projection_bwd_gate_dropres_triton`: 18 keys
- `gated_projection_bwd_gate_flat_triton`: 14 keys
- `gated_projection_bwd_gate_triton`: 12 keys
- `gated_projection_gate_dropres_triton`: 18 keys
- `gated_projection_gate_flat_triton`: 11 keys
- `gated_projection_gate_gemm_triton`: 26 keys
- `gated_projection_gate_inplace_flat_triton`: 6 keys
- `gated_projection_gate_packed_flat_triton`: 6 keys
- `gated_projection_gate_res_triton`: 18 keys
- `gated_projection_gate_triton`: 12 keys
- `layernorm_bwd_atomic_strided_triton`: 22 keys
- `layernorm_bwd_atomic_triton`: 42 keys
- `layernorm_bwd_foldstats_triton`: 36 keys
- `layernorm_bwd_split_mmajor_triton`: 1 keys
- `layernorm_bwd_split_triton`: 45 keys
- `layernorm_fwd_mmajor_triton`: 18 keys
- `layernorm_fwd_recompute_foldstats_triton`: 6 keys
- `layernorm_fwd_rowscale_triton`: 24 keys
- `layernorm_fwd_saveact_strided_triton`: 30 keys
- `layernorm_fwd_saveact_triton`: 42 keys
- `layernorm_fwd_strided_triton`: 6 keys
- `layernorm_linear_bwd_fp32_triton`: 12 keys
- `layernorm_linear_fwd_fp32_triton`: 12 keys
- `layernorm_stats_triton`: 47 keys
- `qk_norm_rope_bwd_triton`: 3 keys
- `qk_norm_rope_fwd_triton`: 5 keys
- `rmsnorm_bwd_triton`: 68 keys
- `rmsnorm_fwd_triton`: 68 keys
- `rope_fwd_triton`: 20 keys
- `swa_gate_out_fwd_triton`: 3 keys
- `transition_bwd_transpose_packed_triton`: 39 keys
- `transition_fold_triton`: 6 keys
- `transition_fwd_b2b_ktiled_triton`: 12 keys
- `transition_layernorm_expand_swiglu_triton`: 6 keys
- `triangle_attention_bwd_dkdv_triton`: 18 keys
- `triangle_attention_bwd_dq_triton`: 18 keys
- `triangle_attention_bwd_pre_triton`: 18 keys
- `triangle_attention_fwd_triton`: 24 keys
- `trimul_bwd_gate_packed_triton`: 24 keys
- `trimul_bwd_gate_recompute_triton`: 12 keys
- `trimul_gemm_gate_mmajor_triton`: 72 keys
- `trimul_gemm_gate_triton`: 12 keys
- `trimul_outproj_layernorm_gemm_gate_triton`: 36 keys

## Build-plan status and practical implications

The packaged `registry_kernel.csv` is not verified against the current sources,
and neither generated nor packaged source-specific SM90 plan exists. The current
plan identity is
`6393ec0e62766c5dd0ddb0be3c81b36ed53d2b837f3a2bd98319f692f807fa37`.
Therefore a precise complete-model count of missing runtime keys is not yet
certified. The regular `build all` command invokes `plan.ensure` before any
GPU tuning and automatically regenerates/verifies this plan. A failed derivation
prevents tuning. This audit did not run a new derivation or submit GPU build jobs.

Use the six-patch interpreter for the intended inventory. Preserve existing
valid measurements and use the builder's default incremental behavior; reserve
`--rebuild` for intentional complete remeasurement. Native policy/build-driver
workload coverage and successful final coverage verification still matter; the
mere presence of a JSON file does not establish completion.

If the portable packed contraction optimization is also to be applied to Triton,
finish that source change before the large native build: native `source_identity`
hashes all kernel source files, including Triton. Editing them afterwards will
invalidate native cache identities even when native GEMM code itself is unchanged.

## L128 and portable optimization evidence

The current Triton bidirectional training source still has the same three large
concatenations: one after the two forward contractions, two after the four
backward contractions. Direct destination-buffer writes can remove them there
as well. The previous 1.108x new-H100/old-Triton comparison includes this portable
improvement and cannot isolate a CuTe-only advantage.

L128 same-operand isolated contraction measurements: cuBLAS 5.18–5.31 us;
Quack/CuTe 6.66–6.82 us. In the packed whole-training search, cuBLAS was
0.297988 ms vs all-Quack 0.298435 ms, only about 0.15% apart. This supports a
cuBLAS preference for those contractions, not a claim that every L128 CuTe
kernel is slower. The separately measured native input projection was
0.02569 ms vs the Triton projection's 0.02653 ms.

Raw inventory: `runs/h100_cache_inventory/static_status.json`, `files.json`,
`searched_configs.json`. Search/benchmark evidence:
`runs/trimul_dispatch_search/summary.json` and
`runs/trimul_training_components/summary.json`.
