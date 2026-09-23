# AdaLN alignment and long attention training memory

## AdaLN GEMM

`adaln_gemm_gate_triton` is the fused **inference** GEMM/normalization/gate
kernel. On H100 with Triton 3.6.0 and bundled ptxas 12.8.93, the tile
`M=128, N=256, K=32, warps=4, stages=1, group=1` failed for all three
L8192 workloads: `(NX, NC) = (128,128), (384,384), (768,384)`.

Compute Sanitizer localized the error to a 16-byte **shared-memory** write at
address `0x1498` in the epilogue. This was not a misaligned input allocation.
An FP32 identity instruction (`mov.b32`) after loading/casting `x` breaks the
faulty code generation pattern without changing input bits or the equations.
The precise compiler transformation responsible has not been isolated.

Validation on H100:

- All three previously failing shapes execute and agree with the FP32 formula
  (relative L2 around 0.0017, against the existing BF16 tolerance of 0.014).
- 21 regression cases cover BF16/FP32, full-size and offset/tail inputs, and
  neighboring tile/warp/stage/group settings.
- Compute Sanitizer: the original kernel reports misaligned shared writes;
  the fixed reproducer reports **0 errors**.
- The full 2,880-configuration search space is retained. Retuning is divided
  into six disjoint shards: three workloads, each with warps `{1,2}` / `{4,8}`.

Same-config performance comparison, alternating seven rounds of CUDA graph
measurement, on previously runnable configurations:

| M / NX / NC | Before (us) | After (us) | Change |
| --- | ---: | ---: | ---: |
| 8192 / 128 / 128 | 4.621 | 4.645 | +0.5% |
| 8192 / 384 / 384 | 15.268 | 15.395 | +0.8% |
| 8192 / 768 / 384 | 28.654 | 28.949 | +1.0% |

Outputs are bitwise equal in these comparisons. These are individual GEMM
kernel times, not end-to-end AdaLN/model speedups. Raw measurements and
configs: `runs/adaln_alignment/performance.json`.

## L8192 compute-efficient attention training

The default remains **compute-efficient at every shape**. The temporary automatic
switch to the atomic backend was removed. Explicit `compute_efficient=False`
remains available as before; production/default calls use the split kernel.

The failed workload is `A=48, B=1, L=8192, H=4, D=32`. Original backward allocated
48 GiB each for `dq_expand` and unreduced `dbias`, before the latter allocation
failed. The revised **same joint dQ/dK/dV/dBias kernel** computes bounded query
windows and immediately reduces their scratch buffers:

- Each key tile still computes dQ, dK, dV and dBias together. No additional
  attention-score pass and no atomic accumulation were introduced.
- dQ keeps independent key-split slots, reduced in the original order directly
  into the query window of the final output.
- dBias is reduced over augmentation per query window, directly into the final
  `(B,H,L,L)` output.
- dK/dV carry FP32 accumulators between windows. Ping-pong input/output buffers
  make repeated autotune launches safe; scaling/casting happens once at the end.
- The launch grid records the **actual** selected key tile. Reduction reads only
  slots that this tile wrote, so clearing scratch and reading unused slots are
  removed. The unchunked small-shape path receives this improvement too.

The workspace target is 4 GiB, rounded to preserve every configured query-tile
boundary. For the failed workload it chooses 256 query rows: the two formerly
48 GiB buffers are **1.5 GiB each**, with 0.75 GiB of FP32 dK/dV ping-pong buffers
and a 1 GiB final dBias output. Workspace size is computed inside the existing
compute-efficient implementation; it does not query free device memory.

### Verification

- All 18 combinations of BF16/FP32, head dimensions 16/32/48, and query/key tiles
  32/64, 64/128, 128/32 pass. Tests cover multiple batches/augmentations, fully
  and partially masked inputs, and a 259-token tail. Nonzero gradients match
  the PyTorch reference and the unchunked split implementation (relative L2
  difference between split variants below 2e-6).
- Static fullgraph compile and CUDA graph backward/replay are checked for
  BF16/FP32 and even/odd query-window counts. Replays are bitwise repeatable.
- Compute Sanitizer: three representative masked/tail tests, **0 errors**.
- With the atomic backend replaced by a function that raises, the actual
  L8192/A48 module completes BF16 and FP32 core forward/backward. Peak PyTorch
  allocated memory, including first-call autotuning, is **15.519 GiB** and
  **16.457 GiB**, respectively. This is module validation, not full-model training.

### Same-configuration backward performance

H100, BF16, A48/B1/H4/D32, seven alternating CUDA-event measurements. The baseline
uses the original source, not the modified unchunked helper. Each pair uses the
same original cached tile/warp/stage settings. These are attention-core backward
times, excluding forward, module projections and full-model work.

| L | Original (ms) | Revised (ms) | Speedup | Core peak memory, original / revised |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 5.534 | 4.503 | 1.23x | 6.25 / 4.20 GiB |
| 4096 | 21.665 | 17.633 | 1.23x | 24.55 / 4.83 GiB |
| 8192 | OOM | 79.056 | — | OOM / 6.20 GiB |

The L8192 measurement uses the L4096 tile as an explicit provisional setting;
it is not presented as a tuned L8192 winner. Artifacts are under
`runs/attention_compute_fix/`: `performance.json`, `result-{bfloat16,float32}.json`,
GPU regression, sanitizer and cache-build logs.

## Cache publication and installation

The overnight cache publisher joined diffs for JSON files lacking a final
newline without GNU patch's `No newline at end of file` marker. Thirteen
installed native cache files consequently contained the next patch's header.
The publisher and deferred native-tuning activation now handle unterminated
lines and spaces in filenames, and verify installed bytes against intended
JSON. Regression tests cover multiple adjacent unterminated JSON files.

All 42 published overnight cache files were checked against the immutable
build snapshot; the 13 damaged files were restored. The previously qualified
CuTe tuning update could then be activated successfully. Native measurements
are preserved across these fixes only after checking that all native kernel,
launcher, tuning, and compiler dependencies are unchanged; the global identity
migration records the exact two unrelated Triton source-file hashes.
AdaLN GEMM timings are rebuilt for its changed implementation.

The ordered installer patches include
`patches/miniworld-engine-attention-compute-memory.patch` to remove automatic
atomic routing and implement the compute-efficient changes above. Native cache
identities are migrated only after proving that the two changed source files
are Triton attention code and all native kernels/launchers/tuning dependencies
remain byte-identical. No attention timing is relabeled as a new measurement.
Root and team-gm carry matching patches and tests.

## Recovery cache status

AdaLN job `13133` completed all six L8192 tuning shards. Their three shared keys
were published by job `13147` in `miniworld-engine-adaln-cache-20260916.patch`.
The previous atomic publication job `13143` and unfinished atomic build were
cancelled; atomic recovery results were not published.

Compute-efficient L8192 module caches completed in job `13152` for both core
dtypes (189 split configurations and 105 reduction configurations per workload).
Job `13157` published four cache files with no rejected shards or files, in
`miniworld-engine-attention-compute-cache-20260916.patch`. The complete 15-patch
stack replays from the pinned source and exactly reproduces the installed H100
caches. Reports and guarded publication are under
`runs/attention_compute_fix/cache_publication/`. These recovery runs do not
constitute a full new-source `build all`: other affected shape/dtype profiles
still need rebuilding, and existing timings are not silently retagged.

## Full current-source cache build

Job `13159` runs the incremental `build all` on seven H100 GPUs from the fixed
source snapshot in `runs/h100_build_20260916_compute_adaln/`. It includes the
remaining AdaLN and compute-efficient attention shape/dtype profiles, the full
declared module matrix, and the alternative-kernel driver pass. Compatible
measurements are reused; changed implementations are measured again.

Before restarting, the fake-launch recorder needed to evaluate callable launch
grids with a declared configuration: attention now obtains its actual dQ split
count from that callback. Without this, both backward launchers raised
`KeyError: 'splits'` during plan derivation. The planner-only patch
`miniworld-engine-derive-grid-callback.patch` fixes this without changing GPU
kernels or native measurement identities. Both failing cases now pass, together
with the patch-stack and cache-publication regressions (11 tests total). Initial
job `13158` was cancelled before tuning and replaced by `13159`.

This build is **in progress**, not certified complete. The job checks module
coverage and cache staleness after building, then uses guarded publication to
install valid results and create `miniworld-engine-h100-cache-build-20260916.patch`.
Its `coverage-final.txt`, `cache-status-final.txt`, `publication.json` and
`job_exit.json` record the eventual result. Completion applies to the registered
H100 build matrix, not arbitrary shapes or other GPU architectures.

## BF16 head-48 backward repair and Triton-only continuation

Job `13159` was cancelled with 31 completed module shards preserved after the
requested scope changed to **Triton only**. Its six failures were token attention
training at `d_single=768, H=16, D=48`, BF16 core, lengths
128/256/384/512/640/768. The H100-native driver pass was not started.

Compute Sanitizer reproduced the first fault at the dQ WGMMA instruction using
`BLOCK_M1=32, BLOCK_M2=64, warps=4, stages=1`. The generated SASS uses uniform
registers UR34/UR35 to construct a shared-memory descriptor before initializing
them. Isolated launches can pass; the actual full-grid build reliably failed.
The diagnostic inputs, compiler metadata, PTX and disassembly are saved under
`runs/attention_head48_fix/`. This is a shared-memory descriptor failure, not a
global dQ scratch allocation overflow.

`miniworld-engine-attention-dq-layout.patch` forms dQ in query-major order,
`dS @ K`, and transposes the result for the existing strided store. This is the
transpose-equivalent of the previous `K.T @ dS.T`; no extra score pass, atomic
accumulation, scratch allocation, or configuration exclusion was added. The
joint dQ/dK/dV/dBias backward and bounded query workspace remain in place.

Validation of the installed fix:

- Job `13172`: **48 GPU regression tests passed**, including the six failing
  lengths, BF16/FP32, chunked/unchunked, masked nonzero reference gradients,
  and existing compile/CUDA graph replay checks.
- Job `13173`: stage-1 head-48 chunked/unchunked memory checks, **0 errors**.
- Job `13171`: the actual module builder completed the 189-candidate split
  search and 105-candidate reduction search under Compute Sanitizer, **0 errors**.
  Sanitizer timings are diagnostic only and are not merged into production caches.
- 12 CPU checks passed for patch replay, publication, planner callbacks and
  exclusion of native driver searches.

Production tuning restarts as array `13174` over the six failing lengths.
Job `13180`, dependent on that array, merges its complete shards and continues
the normal incremental module plan plus **12 Triton driver kernels** on seven
GPUs using `scripts/build_engine_triton_cache.py`. The 14 native driver kernels
are excluded. Its snapshot and eventual coverage/publication reports are under
`runs/h100_triton_build_20260916/`.

Native source code and native timing files are untouched. Their conservative
global source fingerprint changes when Triton attention changes; native cache
requalification is deferred with the native work. Final staleness checking
separately records deferred native files and gates this build on Triton only.

### Scope correction

The 11 unchanged Triton driver kernels had already completed in job `13096`.
A fresh source/environment scan confirmed all 11 valid; they are not unfinished
caches. Pending continuation `13180` was replaced by `13181` with
`--driver-ops adaln_gemm_gate_triton`. The module planner still checks current
required keys and selects only missing work; the driver pass now revisits only
the changed AdaLN GEMM. Native tuning remains deferred. Two scope tests cover
both the full Triton complement and this narrowed driver selection.
