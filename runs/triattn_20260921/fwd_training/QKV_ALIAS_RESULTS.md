# Full Q/K/V/gate fusion selected at L1024

2026-09-26. Installed promotion18246, `qkv_compact_retire6`, compared with frozen18006. **L1024 uses full Q/K/V/gate + attention fusion. L384/L768 retain Q+gate fusion. SOL90 remains unmet.**

H100, B1/H4/D32/C128 BF16, both directions, node02 / normal_h100. All six native outputs and existing backward saves are preserved. No backward algorithm or binary changed.

## Installed complete workload

64 AB/BA rounds x40 CUDA Graph replays per arm. Reduction is `100 * (1 - 1 / median(paired baseline/candidate ratios))`; arm medians need not reproduce that ratio. Both full FWD and F+B pass a bootstrap95% lower-bound speedup greater than1 before and after installation.

| Direction | FWD ms, baseline -> installed | FWD reduction | F+B ms, baseline -> installed | F+B reduction |
|---|---:|---:|---:|---:|
| starting | 3.9352 -> 3.8254 | 2.83% | 14.3506 -> 14.1890 | 1.09% |
| ending | 4.5752 -> 4.4594 | 2.52% | 15.3163 -> 15.1420 | 1.13% |

The earlier complete16x20 pilot18214 and independent staged64x40 run18240 are retained in [qkv-alias-results.json](qkv-alias-results.json), including all paired rounds and confidence intervals. The L384 pilot is approximately tied, and L768 is slower, so neither length is selected for full fusion.

## What changed

Projection Z/weights and attention Q/bias have disjoint lifetimes. They now occupy the same shared-memory union. K/V are projected first into persistent shared tiles. After the Q+gate MMA retires its Z readers, the Q/gate outputs overwrite that dead Z tile and are saved for backward. A CTA barrier separates projection storage from attention reuse.

AtL1024, this permits four projection warpgroups instead of two, followed by six attention warpgroups. The selected kernel retires PV explicitly before continuing. It uses768 threads, 230400B dynamic shared memory,80 registers and one CTA per SM. PTXAS reports16B spill stores and16B spill loads forL1024. The selected kernel is not spill-free or fully overlapped: emitted SASS still waits after each HGMMA even though C7515 is absent.

Separate4-consumer/2-producer candidates use104 consumer and32 producer registers after projection. Their61440-register requirement exactly matches the initial CTA pool. They restore 8/2/4-instruction projection/QK/PV groups. Their schedule adds barrier work and reduces active attention warpgroups; measured speed is lower. Q-carry, shared-P and extra-stage controls also lose. The absence of a compiler warning alone is not an overlap or speed claim.

The register and asynchronous-operand ordering follows [NVIDIA WGMMA guidance](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/wgmma_programming.html). The actual emitted instruction groups and full-workload timings determine selection.

## All new native controls

Four projections plus attention, against installed18006.12 paired rounds x20 replays. Speedup greater than1 is faster. These are not full-module timings.

| Artifact | Build / bench | L384 | L768 | L1024 |
|---|---|---:|---:|---:|
| qkv_compact_alias | 18193 / 18202 | 0.9956x | 0.9442x | 0.9986x |
| qkv_compact_alias4 | 18194 / 18203 | 0.9235x | 0.9392x | 0.9948x |
| qkv_compact_save_overlap | 18195 / 18204 | 0.9214x | 0.9305x | 0.9629x |
| qkv_compact_retire6 | 18196 / 18209 | 0.9946x | 0.9416x | 1.0204x |
| qkv_compact_retire4 | 18197 / 18210 | 0.8099x | 0.8811x | 0.9467x |
| qkv_compact_pshared | 18198 / 18211 | 0.8278x | 0.7255x | 0.7434x |
| qkv_compact_qcarry | 18207 / 18215 | 0.9908x | 0.9308x | 0.9648x |
| qkv_compact_qcarry_retire | 18208 / 18216 | 0.9984x | 0.9379x | 1.0118x |
| qkv_compact_biasprod | 18226 / 18237 | 0.8645x | 0.8751x | 0.9189x |
| qkv_compact_biasprod5 | 18227 / 18238 | 0.8680x | 0.8628x | 0.9204x |

## HBM and profiling limits

NCU18235, L1024 four-projection-plus-attention region: reads1361.65 -> 605.11MB, writes1332.19 -> 1395.24MB; total traffic falls25.74%. Units are normalized from NCU MB/GB columns.

That isolated NCU measurement takes2.77168 -> 2.82381ms and does not reproduce the full-module graph speedup. The retained paired full-module measurements are the performance selection evidence. The fused kernel has SM throughput59.75% and L2 throughput38.30%; neither establishes SOL90.

## Qualification and dispatch

- All10 controls pass30 native fixtures each:5 lengths x6 mask/logit cases. Q/K/V/gate, O and LSE are bitwise equal; independent sampled FP64 and changed-input graph replay pass.
- Selected staged qualification18232:18 full-module fixtures plus8 independent FP64 gradient fixtures. Output, all input/parameter gradients, dropout, optimizer updates, fullgraph and frozen parameters pass.
- Cold verifier18233:fullgraph, custom-op schema/fake/AOT, BF16 AMP, mask=None, partial weights, weights-only gradients and guards pass atL1024.
- Sanitizers18222 (L128/L768) and18234 (L1024):memcheck, racecheck and synccheck, mixed/all-masked; zero errors/hazards.
- Long job18223 was incomplete because CUDA-only profiling returned no events for the ending graph. It is not promotion evidence. The harness now attributes three actual graph replays with CPU+CUDA events before timing; job18240 and the installed run complete both directions.
- Fresh installed verification checks default full-QKV dispatch, the QKV opt-out, and unchanged L384/L768 QG dispatch. All preexisting code and binaries match18006 except the dispatcher and its associated manifests. Checkpoint18246 records48 files.

`MINIWORLD_TRIATTN_QKV_FWD=0` restores18006 Q+gate fusion. Existing QG/Q/training-forward opt-outs also disable the new path. Unsupported shapes use the existing dispatch.

[Promotion record](qkv-promotion-18246.json), [checkpoint](checkpoint18246/snapshot.json), [selected source](qkv_compact_retire6/fused.cu), [all evidence](qkv-alias-results.json), [previous full-QKV follow-up](QKV_STREAM_RESULTS.md), [previous installation](QG_FUSION_RESULTS.md).
