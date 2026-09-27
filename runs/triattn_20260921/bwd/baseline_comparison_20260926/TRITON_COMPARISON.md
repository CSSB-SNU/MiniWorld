# Corrected comparison against the original Triton engine

**Correction:** The earlier report called our already optimized CUDA checkpoint18246 the "existing engine". The user meant the original Triton engine. That baseline label and the implied comparison against original Triton were wrong. The earlier data only compared two versions of our CUDA implementation.

Jobs19328 at384/768 and19337 at1024. Implementations share the same process/device per length, node02 H100 80GB, BF16 input/projection weights, FP32 LN affine, B=1, C=128, H=4, D=32, dropout=0, every seventh key masked. Same nonzero weights/input/dy and all input plus eight parameter gradients; no optimizer or gradient accumulation. CUDA Graph timings use24 rounds x10 replays and20 initial warmups. The384/768 run includes the installed CUDA checkpoint as a fourth control and rotates all24 orders;1024 uses three arms and all six orders to keep simultaneously resident CUDA Graphs within80GB. Compilation, autotuning and post-timing profiling are excluded.

- **PyTorch:** original dense einsum/softmax module path; no torch.compile or SDPA.
- **Original Triton engine:** module and attention source from engine git commit `59bdb335064caee8ec75bafe3bbafa655955a7b8`. The module is byte-identical; only two custom-op registration names in the core are prefixed to avoid reusing the installed CUDA dispatcher. A process-local proxy routes the frozen module to the frozen core while retaining lazy imports for the unchanged original surroundings.
- **Our installed CUDA:** cumulative optimized training checkpoint18246; the API option `implementation="triton"` is still used but actually dispatches our native CUDA kernels.
- **Our latest CUDA candidate:** the same installed training forward and backward except dQ uses the separately qualified `reuse_qdo` extension.
- **Anthropic optimized release:** no backward for the previously compared optimized forward rows, so BWD and F+B remain unsupported; stock-library fallback is not substituted.

## Complete BWD (ms)

| L | Direction | PyTorch | Original Triton engine | Our latest CUDA | Triton / latest CUDA |
|---:|---|---:|---:|---:|---:|
| 384 | starting | 3.1735 | 1.5545 | 0.8471 | 1.83x |
| 384 | ending | 3.2380 | 1.6268 | 0.9021 | 1.80x |
| 768 | starting | 19.3702 | 8.5743 | 4.5703 | 1.86x |
| 768 | ending | 19.6351 | 8.8400 | 4.7920 | 1.83x |
| 1024 | starting | 42.3590 | 18.0198 | 9.8341 | 1.83x |
| 1024 | ending | 42.8148 | 18.4980 | 10.2670 | 1.80x |

## Complete F+B (ms)

| L | Direction | PyTorch | Original Triton engine | Our latest CUDA | Triton / latest CUDA |
|---:|---|---:|---:|---:|---:|
| 384 | starting | 5.7471 | 2.0611 | 1.2115 | 1.70x |
| 384 | ending | 5.9082 | 2.2317 | 1.3656 | 1.63x |
| 768 | starting | 35.7111 | 11.0815 | 6.4885 | 1.71x |
| 768 | ending | 36.4525 | 11.7753 | 7.0181 | 1.67x |
| 1024 | starting | 76.7933 | 23.4020 | 13.6768 | 1.71x |
| 1024 | ending | 78.0301 | 24.5876 | 14.7146 | 1.67x |

## Verification

All12 length/direction/regime cells require finite output/gradients and a predeclared cross-implementation relative L2 limit0.03; observed maximum is0.010327. The two CUDA controls at384/768 are bitwise equal for output and every gradient. At1024, original Triton versus latest CUDA also passes the direct relative L2 limit0.015. Triton gradients are checked numerically rather than assumed bitwise equal to CUDA.

Profiles confirm actual original `_attn_fwd`, `_attn_bwd_preprocess`, `_attn_bwd_dkdv`, and `_attn_bwd_dq` launches. The original arm must have no native dQ, dK/dV, projection/LN-residual fusion or shared-input weight-gradient launch from our CUDA campaign. Its LN/gate profiles are also retained; ordinary cuBLAS projection GEMMs remain part of the original module.

The shared primitives, bias-only dispatch and gate helpers are unchanged against the git baseline. The changed standalone CUDA LN files are inactive for BF16 activations with FP32 affine parameters; original LN uses the Triton path, verified by actual kernel names. All48 installed attention source/binary files are verified before/after each length. No serving source/dispatch is changed.

The first restoration attempt19327 failed before timing because a copied namespace omitted lazy kernel exports. The loader now forwards lazy attribute lookup. No failed-cell timing is included. Job19328 completed384/768 and1024 starting, then ran out of memory holding four complete graph arms at1024 ending. The complete1024 three-arm rerun replaces both directions in this table; partial19328 data is retained separately. This is a block performance comparison, not SOL.

Reproduce: `sbatch compare_triton.sbatch` for384/768; use `--export=ALL,LENGTHS=1024,BENCH_SCRIPT=compare_triton_three.py` for1024. Then `python3 summarize_triton.py --job 19328 --large-job 19337`. Original commit/source hashes, adapted source hashes and actual profiles are preserved in `original_triton_source.json` and `triton-compare-L-JOB.json`.
