# TriMul release — 2026-09-17

## Code and measurements

Engine source is published on `main` through [b06857c0](https://github.com/SanggeunParrk/miniworld-engine/commit/b06857c0).

- Single-direction BF16 training now reuses the bidirectional F567, input dual-gradient GEMM, and input LayerNorm/residual backward fusions.
- Shared F567 chooses the gate-dot orientation by tile aspect ratio, fixing errors exposed by full-grid tuning. Targeted H100 caches cover L128/384/768, d128, for the three reused training kernels.
- Bidirectional Triton inference writes both contraction results directly into final-buffer slices through `packed_forward`. This removes the activation `torch.cat`; two cuBLAS calls remain. CuTe inference is unchanged.
- [Training measurements and validation](https://github.com/SanggeunParrk/miniworld-engine/blob/b06857c0/docs/records/unidirectional-trimul-20260917/REPORT.md): 69 regression cases, 1,728 candidate-layout numerical checks at L384, and a sanitizer check. L768 module training improves about 1.11x; smaller shapes improve less and L128 incoming without graphs measured 1.6% slower.
- [Inference contraction measurements and validation](https://github.com/SanggeunParrk/miniworld-engine/blob/b06857c0/docs/records/trimul-inference-nocat-20260917/REPORT.md): 7 tests including masks, static compile, CUDA graph replay, and profiler checks. The contraction alone improves 1.48–1.64x at L128/384/768; this is not a full-module speedup.

## Reinstall support

The consumer engine dependency remains pinned to `1bc0803e3b2fef3b963fdc383e090c0adcccdb43`. After installing that dependency, run:

```sh
python scripts/apply_engine_audit_patches.py
```

The existing three audit patches are followed by one cumulative release patch. This consolidates the intervening source fixes required by TriMul; it also contains the three targeted TriMul cache files. Historical bulk cache-only changes are omitted. Already-installed caches outside this scope are left untouched, and a fresh install uses normal tuning fallback when its cache does not match the source.

The helper validates the entire patch sequence in a temporary tree before modifying the installed package. The release was checked against the clean dependency pin and the actual cu128 installation; replay is idempotent. The patch helper regression suite also passed all four cases (fresh, partially patched, fully patched, and mismatch refusal). File hashes and the original patch lineage are recorded in [the manifest](../patches/trimul-release-20260917.json). Source-history patches named there describe provenance and are consolidated into the release patch, not separate required files.

The shared F567 source change invalidates prior F567 cache identities, including bidirectional KP256 shapes. The three targeted caches do not mean the global cache build is complete. A build from an older source snapshot does not retune this revision.
