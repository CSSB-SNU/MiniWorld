# TriMul v6 versus Triton: historical comparison identity

> Follow-up: [packed training results](trimul-packed-training.md) replace the
> contraction concatenations and qualify the new installed H100 dispatch.

Audited 2026-09-15 using local Git objects, original committed benchmark tables,
raw logs, implementation diffs, and the installed engine/local patch stack.
No production changes or new GPU measurements were made for this history audit.

## Finding

The original H100 v6 speedup was real in the recorded benchmark, but its Triton
baseline was `baseline_dtv1_bidir`, not today's `implementation="triton"`.
On July 12 the latter was rewritten to mirror v6's CuTe algorithm and reuse its
backward/normalization/gate helpers. Therefore the original large improvement
and the current small CuTe-versus-Triton difference compare different baselines.

There is also a concrete limitation in the current compiled H100 dispatcher:
the local cold-compile fix bypasses calibrated Quack/CuTe choices and selects
cuBLAS. That deserves separate optimization, but current component results do
not establish that restoring those choices would recover a 1.5x whole-step gain.

## Commit evidence

| Date | Commit | What changed / what was measured |
| --- | --- | --- |
| 2026-06-27 | `5dcd6663bd145ddfce8f09cea2c50eedffdcc16a` | Introduced H100 v6/bidirectional stack and merged backward. Reports training 1.36–1.55x versus dt-v1 at L>=384. |
| 2026-06-30 | `7e28b1f6a4bddd8ff722f218596332ae6ab284d1` | Merged single-direction autograd backward; fused input-gradient addition. Historical mask placement differs from corrected current semantics. |
| 2026-07-03 | `ed2f5299dac0ee2b3754c09cccd4490f8676fae7` | Replaced out-of-place addmm with in-place accumulation; eliminated a 268 MB copy at L1024. |
| 2026-07-08 | `66ac75dec3171a1e75c43b134429d64546b7a658` | Corrected **B200/SM100** bidirectional v6 wiring, about 2.5x versus cuEquiv. This is not H100 versus current Triton. |
| 2026-07-12 | `a5b583144660f25a1205c2f179c7575983c1fceb` | Rewrote Triton bidirectional as a 1:1 CuTe-algorithm mirror with the same fusion boundaries and shared backward helpers. |
| 2026-07-12 | `8767210982e44d045ed4757d3fa68cb1ada35c72` | Ported the same BDLL pipeline to single-direction Triton. B200 recorded training about 28 -> 4.33 ms at L1024 (6.5x); not an H100 timing. |
| 2026-08-24 | `401d4dae1c825d2d78f3e7742584e4691b277af4` | Made individual launches opaque custom ops and removed wrapper graph breaks; compiler now sees surrounding reshapes/GEMMs/reductions. |
| 2026-09-08 | `a1551fed6685b02fd684d5d577ebc9b9ddfd6d92` | Fused bidirectional **inference** back half and rebuilt affected A6000 tuning units. Its platform/results cannot be relabeled H100 v6 training. |

The July 12 mirror is an ancestor of installed engine pin
`1bc0803e3b2fef3b963fdc383e090c0adcccdb43` (verified with `git merge-base
--is-ancestor`). The external source checkout's HEAD and dirty working file are
not the installed baseline; this audit uses pinned Git blobs for that comparison.

## Original H100 training table, verified against its raw log

From `5dcd6663`'s `trimul_inproj/benchmark/bidir_full.md` and `.out`:
BF16, batch 1, d_pair=128, bidirectional training.

| L | Original dt-v1 bidirectional, ms | v6 bidirectional, ms | dt-v1 / v6 |
| --- | ---: | ---: | ---: |
| 384 | 2.8921 | 2.1259 | 1.360x |
| 512 | 4.9606 | 3.3715 | 1.471x |
| 768 | 10.9358 | 7.3880 | 1.480x |
| 1024 | 19.6019 | 13.3417 | 1.469x |

The original baseline imports `fused_bidirectional_dtv1`, built from the
vendored dt-v1 input/contraction/output-normalization kernels. It is a separate
implementation, not the July 12 mirror's `_BidirBackHalfTriton`.

Historical v6 L768 is already near 7.4 ms; the new static-graph results are v6
7.413 ms and modern Triton 7.506 ms. This is useful context, **not** a controlled
no-regression proof: the old harness used ordinary CUDA-event timing, gradient
reset to None, a different dropout broadcast ([B,L,1,1] versus current [B,1,L,D]),
no holed mask, different dependencies, and wrapper graph breaks. In the original
source, `bidir_forward` is decorated `torch.compiler.disable`; the label
"torch.compile" did not imply today's fullgraph capture.

There are multiple historical tables in that commit: `bidir_train.md` records
an earlier, slower result (L768 v6 9.451 ms versus dt-v1 10.562 ms), whereas
`bidir_full.md` and its matching raw `.out` record the merged stack above.
Do not cherry-pick a table without identifying its harness and result revision.

## Why modern Triton is much closer

`a5b58314` explicitly replaces the older torch-composed/per-direction design
with the CuTe BidirBackHalf algorithm. The committed code imports the same:

- `triton_layernorm`;
- `_te_forward` and `_te_backward`;
- `front_bwd_dW`;
- gate forward/elementwise-backward helpers.

It also uses channel-major front stores, two cuBLAS contractions, merged manual
backward, and the fused input-gradient add. The newly written backend-specific
kernel is the front forward. Hence v6's main improvements were subsequently
available in the modern Triton path as well.

The historical decisive front-backward rewrite replaced a slow handwritten
split-K weight-gradient kernel with packed elementwise gradients plus cuBLAS
GEMMs (documented 4.0 -> 1.46 ms at that historical case). Current Triton shares
this implementation. The channel-major output-normalization approach and
in-place gradient accumulation also remain present.

## Actual remaining H100 restriction

In pinned `cute/dispatch.py`, an eager-calibrated cache HIT could be traced into
the compiled graph; a cold miss selected a default. In the local
`patches/miniworld-engine-h100-upgrade.patch`, the cold-compile repair instead
returns the cuBLAS candidate before constructing a stride-based key. It avoids
the observed unknown-stride failure during speculative backward tracing, but
also prevents compiled execution from using warm Quack/CuTe winners.

This was a local implementation change, not a historical claim that cuBLAS
always wins. The [component audit](trimul-training-components.md) found at L768:

- Quack modestly faster for the six contraction GEMMs on the tested operands;
- merged input gradient effectively tied;
- cuBLAS 5.3x faster for front weight gradient and 21.7x faster for gate weight
  gradient than the tested Quack candidates.

Thus blanket CuTe dispatch would regress important work. Restoring safe
shape/layout-specific choices is a concrete target, distinct from native tile
tuning and the dynamic compiler copy regression.

## Evidence on disk

`runs/trimul_git_history/provenance.json` records exact commit IDs and source
specifications. That directory contains original commit messages, benchmark
table/raw log/harness, historical v6 and Triton mirror source, and the pinned
dispatcher. All were exported with `git show`; no source checkout was modified.

Current results and method: [paired component audit](trimul-training-components.md).
