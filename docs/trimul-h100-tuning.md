# H100 TriMul tuning applied — 2026-09-15

> Follow-up: [packed training results](trimul-packed-training.md) replace the
> contraction concatenations and qualify the new installed H100 dispatch.

**Training follow-up:** the [paired component audit](trimul-training-components.md)
finds the native training projection faster than Triton and near the best of all
18 candidates. The earlier compiled bidirectional loss depends on dynamic shape
promotion; explicit production-style `dynamic=False` removes that loss.

## Scope and result

The active native SM90 kernel in the current TriMul path,
`trimul_inproj_masked_sm90_cute`, is now tuned and installed for batch1,
d_pair128, BF16, L128/384/768: inference and single/bidirectional training.
These are9 distinct normalized runtime keys. Every key has measurements for all18
declared candidates, and the runtime audit observes9/9 native cache hits alongside
33/33 existing Triton hits. The corrected build driver emits all9 runtime keys.

This completes the identified native tile-cache gap. It does not complete all
TriMul performance work: the separate compiled bidirectional backend/layout
issues remain. Unused experimental kernels and other module families were outside
this TriMul tuning pass.

## Code and installation

- The masked-front cache key now uses the actual contiguous FP32[1,M] mask sent
  to the kernel. Equivalent bool/BF16 inputs with different original ranks or
  strides share a tuning entry. The numerical mask computation is unchanged.
- The driver now emits the real projection contracts: [D,2D] without saved
  preactivations for per-side inference, [D,4D] with saved preactivations for
  single-direction training, and [D,8D] for bidirectional training.
- The new `patches/miniworld-engine-trimul-tuning.patch` contains both fixes and
  the measured H100 JSON. It follows the four existing engine patches in
  `scripts/apply_engine_audit_patches.py`. MiniWorld and team-gm carry matching
  copies, so engine-setup restores the source fixes and tuning file on reinstall.
- Source identity remains checked by the native cache reader. These timings are
  not relabeled for a different kernel revision or compiler environment.

## Measurement

Three GPUs worked in parallel by sequence length. The first native builder pass
measured all162 workload/config pairs using its normal cache-clearing event
benchmark. Every candidate passed output/preactivation comparisons and zeroed
masked rows correctly. That pass filled the missing entries, but whole-module
CUDA graph checks found a small regression on compiled L128 inference.

A second complete162-pair sweep used the actual graph-replay regime:12 alternating
candidate-order rounds,50 replays per sample, median time. Each candidate was
again checked numerically before capture. Final cache ranking comes ONLY from
these graph measurements, merged in a separate staging tree; earlier event-based
rankings were archived instead of mixed into the final cache. The installed JSON
records the measurement regime in provenance. Original shards retain all candidate
measurements; the installed file retains the top5 per key and searched-grid data.

Maximum relative L2 error across all graph-sweep candidates was0.0016604 for the
masked output and0.0016590 for raw preactivations against FP32 GEMM/GLU math. These
are tensor-norm errors, not a maximum elementwise bound. All masked rows were
exactly zero; preactivations remain unmasked for backward.

## Isolated native kernel results

Milliseconds per launch. These exclude mask conversion and the rest of TriMul.
A kernel winner does not guarantee a whole-module improvement under every
working set or graph allocation.

| L | Contract | Declared default | Graph winner | Speedup |
| --- | --- | ---: | ---: | ---: |
| 128 | inference | 0.007440 | 0.007377 | 1.009x |
| 128 | single_training | 0.014192 | 0.013234 | 1.072x |
| 128 | bidir_training | 0.026108 | 0.025896 | 1.008x |
| 384 | inference | 0.047873 | 0.042800 | 1.119x |
| 384 | single_training | 0.108459 | 0.106744 | 1.016x |
| 384 | bidir_training | 0.195621 | 0.194071 | 1.008x |
| 768 | inference | 0.174894 | 0.146971 | 1.190x |
| 768 | single_training | 0.414545 | 0.406210 | 1.021x |
| 768 | bidir_training | 0.764477 | 0.738548 | 1.035x |

## Whole-module qualification

The final audit compares tuned/default settings in the same process/GPU, using
identical model parameters and inputs. Only the target native selector is forced
to its declared default for the control graph. Other dispatch decisions stay
fixed. Each sample is20 graph replays;12 rounds alternate execution order.
Training includes forward, backward and gradient reset, with dropout0.25 and
holed masks. All cases use manual CUDA graphs; compiled cases additionally use
fullgraph torch.compile with compiler-owned graphs disabled.

L768 times in milliseconds:

| Family | Mode | Compile | Default | Tuned | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| outgoing | inference | True | 1.02726 | 0.97284 | 1.0559x |
| outgoing | training | True | 4.42657 | 4.43058 | 0.9991x |
| outgoing | inference | False | 1.03702 | 0.98051 | 1.0576x |
| outgoing | training | False | 4.34115 | 4.34073 | 1.0001x |
| incoming | inference | True | 1.03771 | 0.98197 | 1.0568x |
| incoming | training | True | 4.41411 | 4.41203 | 1.0005x |
| incoming | inference | False | 1.04478 | 0.98845 | 1.0570x |
| incoming | training | False | 4.32902 | 4.32850 | 1.0001x |
| bidir | inference | True | 2.63698 | 2.53179 | 1.0415x |
| bidir | training | True | 9.12438 | 9.15877 | 0.9962x |
| bidir | inference | False | 2.10674 | 1.99421 | 1.0564x |
| bidir | training | False | 7.34323 | 7.32769 | 1.0021x |

L768 inference improves about4–6%; training remains essentially unchanged
(approximately-0.4% to+0.2% across these cases). The isolated training projection
is only a fraction of the complete backward step. Backend selection and
concatenation/transposition costs were not changed by tile tuning.

**Small-shape limitation:** compiled L128 inference remains about1.4–2.1% slower
than the declared default, approximately0.9–1.2 microseconds per whole module in
these graph allocations. The isolated L128 inference winner only beats the
kernel default by about0.9%, so its small kernel advantage does not translate to
this complete module. This is an observed limitation, not a universal speedup
claim or a claim that TriMul performance optimization is finished.

## Verification

- Final native lookup coverage:9/9; existing Triton coverage:33/33.
- Real driver meta-tensor replay:9/9 normalized native keys emitted.
- Final36 full-module cases: outgoing/incoming/bidirectional × three lengths ×
  train/eval × compile/eager; all outputs and training gradients finite; training
  graph replays advance dropout RNG. Native kernel traces were recorded.
- Final regression job13059:53 tests passed, including mask variants, all
  gradients against references, dropout/residual behavior, cold compiled
  inference/training, partial tiles and canonical mask-key reuse.
- Patch-stack CPU tests:4 passed. All five patches apply to a clean pinned source
  tree and are idempotent. Full fresh-source verification confirms its native
  source hash equals the measured hash stored in the cache.

All tuning and validation jobs completed successfully. One L384 graph job
submission was temporarily rejected by automatic approval review because its
review model was at capacity; the same bounded job was rechecked, resubmitted
normally and completed as13056.

## Reproduce and evidence

Run each length on an allocated H100 using the project cu128 environment:

```bash
python scripts/tune_trimul_h100.py --cuda-graph --length 128 --output-dir runs/trimul_h100_tuning/repeat128
python scripts/tune_trimul_h100.py --cuda-graph --length 384 --output-dir runs/trimul_h100_tuning/repeat384
python scripts/tune_trimul_h100.py --cuda-graph --length 768 --output-dir runs/trimul_h100_tuning/repeat768
```

These commands emit measurement shards; they do not concurrently overwrite the
runtime cache. Merge only the desired measurement regime with
`capture.merge_shards(..., gpu="NVIDIA H100 80GB HBM3 (sm90)",
only_ops={"trimul_inproj_masked_sm90_cute"})` after validation. Do not mix the
initial cache-clearing and final graph-replay shards in one ranking.

Whole-module paired audit:

```bash
python scripts/audit_trimul_tuning.py --family outgoing --compare-default --output runs/trimul_h100_tuning/repeat_outgoing.json
```

Repeat with incoming/bidir. The default remains the installed H100 auto module.

Artifacts under runs/trimul_h100_tuning:

- L128_13048, L384_13046, L768_13047: initial event sweep and numerical checks.
- L128_graph_13054, L384_graph_13056, L768_graph_13055: final complete graph sweeps.
- cold_event_cache.json: archived first-pass ranking; graph_cache/: final ranking.
- final_outgoing_13057.json, final_incoming_13060.json, final_bidir_13058.json:
  final paired comparisons, lookup records and corresponding GPU traces.
- final_coverage/: cache-key deduplication and actual build-driver coverage.
- summary.json: selected configurations, kernel/module timings and installed SHA-256.
- reinstall_verification.json: clean-source patch and source-identity verification.

Initial module comparison jobs13051–13053 describe the superseded event-based
ranking. Final report timings use13057/13058/13060 only. Logs are under
runs/v1.0.1/phase2b/full_wiring_<job>.out.
