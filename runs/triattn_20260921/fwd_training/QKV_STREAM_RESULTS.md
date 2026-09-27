# Full Q/K/V/gate fusion follow-up

> Historical full-QKV follow-up through18162. Later L1024 full fusion is installed; see [current results](QKV_ALIAS_RESULTS.md).

2026-09-26. **No new candidate selected. Installed checkpoint18006 (`qg_scoped`) remains unchanged. SOL90 is unmet.**

Native CUDA/TMA, B1/H4/D32/C128 BF16 on node02 / normal_h100. All twelve valid candidates fuse Q/K/V/gate projections and attention into one kernel, preserving all six outputs needed by the existing backward. The comparison baseline already fuses Q and gate with attention; K/V use two GEMMs.

## Four projections plus attention

Speedup is the median paired baseline/candidate time ratio; greater than1 is faster. Each measurement uses12 AB/BA rounds with20 CUDA Graph replays. These are the combined projection/attention boundary, not attention-only or full-module FWD.

| Candidate | Build / bench jobs | L384 speedup | L768 speedup | L1024 speedup |
|---|---|---:|---:|---:|
| qkv_stream_c4_static | 18063 / 18071 | 0.8000x | 0.7080x | 0.7711x |
| qkv_stream_c2_late88 | 18065 / 18072 | 0.8217x | 0.7180x | 0.7235x |
| qkv_stream_wide2b | 18089 / 18095 | 0.9287x | 0.8033x | 0.7995x |
| qkv_stream_wide4b | 18090 / 18096 | 0.8929x | 0.7694x | 0.8136x |
| qkv_stream_wide6b | 18091 / 18097 | 0.8879x | 0.2845x | 0.8142x |
| qkv_compact | 18094 / 18101 | 0.9073x | 0.8909x | 0.9186x |
| qkv_compact_async4 | 18105 / 18111 | 0.9044x | 0.9093x | 0.9357x |
| qkv_stream_slots3_c2 | 18106 / 18112 | 0.9367x | 0.8113x | 0.8036x |
| qkv_stream_slots3_c4 | 18107 / 18113 | 0.8755x | 0.7811x | 0.8190x |
| qkv_compact_qresident | 18119 / 18123 | 0.8851x | 0.8662x | 0.9231x |
| qkv_compact_serial | 18120 / 18124 | 0.7911x | 0.8234x | 0.8888x |
| qkv_compact_remat | 18140 / 18155 | 0.9140x | 0.9264x | 0.9670x |

L64/128 also pass and sometimes improve; they are not the target lengths used for selection. Separate arm medians and every paired ratio are retained in [machine-readable results](qkv-stream-results.json). The median of ratios need not equal the ratio of arm medians.

## Final candidate: complete module

`qkv_compact_remat` recomputes epilogue coordinates after attention. PTXAS reports zero spill stores/loads for all compiled configurations, down from28B/28B for the large compact_async4 configurations. It still loses at the target lengths.

Module job18159 uses16 paired rounds x20 replays, both directions, with bitwise output and all input/parameter gradient comparisons. Positive percentages below mean more time than installed18006.

| L | Direction | Full FWD time change | BWD time change | F+B time change |
|---|---|---:|---:|---:|
| 384 | starting | +5.91% | -0.03% | +2.00% |
| 384 | ending | +4.49% | -0.04% | +1.50% |
| 768 | starting | +4.24% | +0.14% | +0.01% |
| 768 | ending | +2.25% | +0.27% | +0.15% |
| 1024 | starting | +1.37% | +0.64% | +0.02% |
| 1024 | ending | +1.74% | +0.77% | -0.18% |

The earlier compact_async4 module result (job18125) is also retained in the JSON. No backward algorithm changed. Small BWD/F+B differences do not establish a consistent gain.

## L768 HBM evidence

Each row sums the same four-projection-plus-attention region. NCU timing is profiler timing, not the paired graph timing above. Baselines are freshly measured within each profile job.

| Job / candidate | HBM read MB, baseline -> candidate | HBM write MB, baseline -> candidate | Duration ms, baseline -> candidate |
|---|---:|---:|---:|
| 18076 / qkv_stream_c4_static | 760.13 -> 158.38 | 723.57 -> 747.18 | 1.26752 -> 1.76854 |
| 18128 / qkv_compact_async4 | 760.14 -> 273.36 | 723.47 -> 809.84 | 1.26589 -> 1.35933 |
| 18162 / qkv_compact_remat | 760.18 -> 257.70 | 723.62 -> 746.52 | 1.26259 -> 1.33370 |

The158MB streaming read result and the faster compact timings belong to different candidates. They must not be combined into a fictitious result. Training still needs globally saved Q/K/V/gate; fusion removes their forward rereads, not those required backward-save writes.

Final remat reduces reads66.10% and total HBM traffic32.32% in job18162, but the profiled region takes5.63% longer. Its SM throughput is59.77%, L2 throughput35.81%, and active-warps metric36.61%; none establishes SOL90. The NCU raw export emits a nonfatal embedded-Python site/encoding traceback after each completed report. Both native runs complete, the .ncu-rep files exist, and CSV exports contain the expected three baseline kernels and one candidate kernel with valid metrics.

## Implementation and bottleneck evidence

- Streaming: one producer warpgroup projects64-key K/V tiles;2/4 consumers retain Q and share a two- or three-slot K/V ring. Only query group0 writes K/V saves; other groups recompute them. QK completion retires the preceding PV before consumers release its slot. The N64 producer computes K and V together with double-buffered Z loads.
- Compact: K/V stay in shared memory. A configurable subset of six warpgroups projects Q+gate then K+V; all six perform attention. Weight/Z scratch is reused, and projection-save waits are deferred where lifetimes permit. AtL768, compact_async4 uses230400B dynamic shared memory and one CTA per SM, versus37888B and a five-CTA resource ceiling for installed QG. Residency and projection phase costs remain despite reduced HBM traffic.
- Q-resident and serial-weight controls trade additional shared memory against Q rereads and repeated Z loads; neither wins. The rematerialized epilogue eliminates the local-memory coordinate spills without changing projection/softmax order.
- PTXAS still emits C7515 WGMMA serialization warnings in the compact builds, including [18140](stream-build-18140.log). Zero spill does not imply a fully overlapped WGMMA pipeline. The current evidence does not isolate the exact cost of compiler serialization.
- Job18129 instruments CTA cycles around initialization, projection completion and attention completion. Mean projection fractions are36.26%,20.89%,20.29% forL384/768/1024. The instrumented binary has extra register/spill cost. These are CTA-cycle fractions, not exact whole-GPU time attribution or SOL evidence.

Asynchronous save scheduling distinguishes shared-source retirement from global-destination completion: `cp.async.bulk.wait_group.read` covers source reads; default wait covers destination writes as well. See [NVIDIA PTX semantics](https://docs.nvidia.com/cuda/archive/12.6.2/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group). Q reloads occur only after destination completion and the required CTA synchronization.

## Validation and rejected experiments

- All12 valid native candidates:30 fixtures each (L64/128/384/768/1024, mixed/dense/one-key/all-masked/late-live/large-logits). Q/K/V/gate, O and LSE match the frozen baseline bitwise; independent sampled FP64 references and changed-input CUDA Graph replay pass.
- Compact_async4: full-module bench18125, qualification18126 (18 module fixtures plus8 independent FP64 gradient fixtures), sanitizer18127.
- Compact_remat: full-module bench18159, qualification18160 (same26 fixtures), sanitizer18161. Both candidates pass memcheck/racecheck/synccheck atL128/L768, mixed/all-masked, with zero errors or hazards. No production installation or fresh-installed qualification is claimed.
- The first C2 register redistribution required32768 registers but had30720 allocated, and could deadlock. Jobs18060/18061 were cancelled. Later88-register consumers plus64-register producer fit the pool. Every valid streaming benchmark runs the SASS register-pool audit first.
- Initial combined-N64 weight loading used the wrong swizzle (18079/18080/18085). It failed correctness and was replaced by fourM32/K64 TMA boxes. Dependent18086/18087 were cancelled. None of those measurements is accepted as performance evidence.
- Source-generation failures18116/18117 were cancelled before the corrected builds. 18056/18057 built binaries but their initial wrapper failed to find barecuobjdump; all later wrappers use the project environment. Unrelated training jobs were preserved.
- All44 installed files still match checkpoint18006 SHA256 values. The experimental six-output wrapper uses the existing backward and remains outside production dispatch.

[Design and ownership](STREAMING_QKV.md), [final CUDA source](qkv_compact_remat/fused.cu), [all hashes, timings and evidence paths](qkv-stream-results.json), [installed QG results](QG_FUSION_RESULTS.md).
