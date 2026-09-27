# Completed node02 backward continuation

Installed `rs8_async_q4` dK/dV + bias; dQ remains `rs_softmax_overlap`.
All builds, measurements and validation in this continuation used node02
with normal_h100 QoS. Previously queued17175/17167/17168 were moved in
place from node01, preserving their dependencies.

## Gain since the start of this continuation

Direct comparison17212 against frozen checkpoint17069, all input and
parameter gradients, both directions.24 balanced AB/BA rounds and20
CUDA graph replays. No sum of separate improvements. Times are arm
medians; reductions use median paired ratios.

| L | Direction | Before BWD ms | Final BWD ms | BWD reduction | F+B reduction |
|---:|---|---:|---:|---:|---:|
| 384 | starting | 0.9184 | 0.8458 | 7.90% | 4.91% |
| 384 | ending | 0.9714 | 0.8988 | 7.47% | 4.48% |
| 768 | starting | 5.1113 | 4.7631 | 7.26% | 4.11% |
| 768 | ending | 5.3071 | 4.9365 | 7.20% | 4.21% |
| 1024 | starting | 11.0945 | 10.3076 | 7.05% | 4.31% |
| 1024 | ending | 11.4579 | 10.6023 | 7.41% | 3.89% |

## Current L768 SOL

| Kernel | Time ms | Compute SOL | Memory SOL |
|---|---:|---:|---:|
| dK/dV + bias partial | 2.0706 | 54.00% | 55.70% |
| Bias final reduction | 0.2912 | 5.02% | 93.47% |
| dQ | 1.0169 | 62.64% | 78.62% |

**Neither target core reaches SOL90.** Bias final reduction was
already above the80% cutoff and remains unchanged.

## Implementation and qualification

- Two previously idle producer warps reduce the eight dS tiles while
  consumers calculate the next query tile. FP32 association is preserved.
- Four shared Q/dO/stat stages keep TMA farther ahead. Shared memory is
 175104 bytes; PTXAS reports168 initial registers and zero local spills.
  No additional HBM tensor is introduced; R8 scratch is unchanged.
- Explicit full-reader and ready/empty barriers remain on every reused
  shared stage. Qualification17203 passes independent FP64/masks, exact
  bias cancellation, module10 cases and all six sanitizer runs at64/256.
- Installed job17211 passes all manifests, wgrad opcheck, ten opt-outs,
  cold fullgraph, full all-gradient timing, BF16 AMP/FP16 refusal, and
  exhaustive16/17-kernel attribution. Every gradient in comparison17212
  is bitwise equal to the starting installation.

## Rejected controls

| Pilot | Candidate | L384 time change | L768 time change | L1024 time change |
|---:|---|---:|---:|---:|
| 17168 | rs8_async_q8 | -11.64% | -10.28% | -9.51% |
| 17193 | rs_half_p4 | -0.19% | +3.58% | +4.77% |
| 17194 | rs_half_p5 | -0.58% | +1.73% | +2.85% |
| 17208 | rs_half_p_overlap | -0.04% | +1.47% | +3.08% |

These controls compare with installed17175. Depth8 is faster than
that parent, but slower than selected depth4. All three dQ half-P
controls lose at768/1024 despite zero spills and exact pilot outputs;
they are not sanitizer-qualified or installed.

The original campaign comparison against16663 is in [README.md](README.md):
full backward21.92-23.23% shorter, F+B14.63-16.00% shorter.
[ATTRIBUTION.md](ATTRIBUTION.md) records every stage of the final installed path.
