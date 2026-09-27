# Bounded backward reuse follow-up

**Baseline terminology correction:** The user's original engine is Triton.
Checkpoint18246 in this document is **our cumulatively optimized CUDA version**.
The [corrected original-Triton comparison](../baseline_comparison_20260926/TRITON_COMPARISON.md)
measures1.80–1.86x complete-BWD and1.63–1.71x F+B speedups for current CUDA
over original Triton. Those campaign gains are separate from this last dQ change.

The later [PyTorch / Anthropic / installed-engine comparison](../baseline_comparison_20260926/README.md)
retains this candidate as an explicit experiment. Its three-arm timings confirm
the large gain over dense PyTorch but do not resolve a consistent small advantage
over checkpoint18246. In particular, the L768 timing changes with execution order;
do not generalize the dedicated two-arm sub-percent result below to all benchmark
conditions. The installed engine is still the default.

Three native CUDA candidates were tested on node02/normal_h100 against the
installed training checkpoint18246. `reuse_qdo` is the only worthwhile
candidate: dQ retains both Q and dO in eight packed registers each across
the key loop. It uses122 registers, zero stack/spills and41984B dynamic
shared memory. The existing128-thread CTA, TMA K/V/bias staging and required
barriers remain. This saves repeated shared-memory operand reads; no new
HBM intermediate is added or removed.

The ordinary training package is unchanged. All48 checkpoint18246 files
still match, and the separate inference forward is unchanged. A process-local
experiment is available through [candidate.py](candidate.py); it changes only
dQ and restores the installed extension after leaving its context.

Final19268 confirms0.10–0.66% less complete-backward time. Only L1024 gives
clear F+B gains in both directions,0.22–0.36%. This is a small qualified
experiment, not a new production installation or a large backward speedup.

```python
from runs.triattn_20260921.bwd.reuse_20260926.candidate import use

# Load/warm before CUDA Graph capture. Enclose eager forward and backward.
with use():
    output = model(x, mask)
    loss_fn(output).backward()
```

## Native controls

Twelve balanced AB/BA rounds x20 CUDA Graph replays. Percentages are time
reductions relative to the installed corresponding native boundary;
negative means slower. The dK/dV boundary includes final bias reduction.
All native outputs are bitwise equal at64/384/768/1024.

| Candidate | L384 | L768 | L1024 |
|---|---:|---:|---:|
| dQ: Q reuse | -0.21% | +0.10% | -0.15% |
| dQ: Q + dO reuse | +0.58% | +1.79% | +1.83% |
| dK/dV: exp with pending dP | -0.01% | -0.26% | +0.26% |

The Q-only and dK/dV controls also receive complete backward/F+B comparisons
at all three lengths and both directions. Neither gives a consistent full
workload gain, so they remain unselected. The dK/dV producer/consumer register
budget, row-group8 partials and asynchronous bias reduction stay unchanged.
See [design](DESIGN.md) and [machine-readable results](results.json).

## Complete workload comparison

Final job19268 uses64 AB/BA rounds x40 replays, after20 extra alternating
warmup replays per graph. It includes all input and parameter gradients.
Forward and all other backward kernels are identical in each pair; dropout0
and every seventh key masked. F+B includes forward and backward, without an
optimizer step. Smaller qualification runs cover dropout and SGD updates.

| L | Direction | Previous BWD ms | Candidate BWD ms | BWD time reduction | F+B time reduction |
|---:|---|---:|---:|---:|---:|
| 384 | starting | 0.8491 | 0.8486 | 0.10% | 0.11%* |
| 384 | ending | 0.9012 | 0.9000 | 0.15% | 0.12% |
| 768 | starting | 4.7988 | 4.7810 | 0.28% | 0.08%* |
| 768 | ending | 4.9927 | 4.9736 | 0.40% | 0.03%* |
| 1024 | starting | 10.3964 | 10.3268 | 0.66% | 0.36% |
| 1024 | ending | 10.7396 | 10.7037 | 0.35% | 0.22% |

*The95% confidence interval includes zero, so no reliable F+B improvement
is claimed in those cells. All six backward intervals are above zero.
L1024 backward intervals are0.43–0.76% /0.28–0.46%, and F+B intervals are
0.32–0.46% /0.16–0.31%, starting/ending. The installed training forward is
used here; the inference-only forward has no backward saves and is a separate
workload. These are comparisons against our existing package, not Anthropic.

Paired ratios determine percentages and5000-resample bootstrap95% intervals.
Individual arm medians can move differently under clock variation and should
not be used to recalculate the paired percentages. Job19257 retains an
independent32x30 comparison. Do not add or multiply independent pilot gains.

## Qualification

Job19266 covers independent FP64 adjoints at64/128 with none/mixed/one-key/
all-masked inputs; none/mixed full-module gradients with dropout, SGD updates,
fullgraph and frozen parameters at384/768/1024; changed-input/weight/dy/mask
complete F+B graph replay and actual candidate-kernel attribution in both
directions at all three target lengths. Native and eager module comparisons
require bitwise equality to the installed algorithm. Independent FP64 keeps
the established error thresholds rather than requiring FP64 equality.

Memcheck, racecheck and synccheck pass at64/384/768/1024 for both mixed and
all-masked inputs:16 zero-error mem/sync summaries and8 zero-hazard race
summaries. The explicit experiment entry is separately verified by19277,
including extension restoration, changed graphs and actual kernel names.
Other candidates have pilot correctness/full-workload evidence, not a full
sanitizer qualification.

## Remaining bottleneck

Fresh whole-backward attribution19245 measures L1024 dK/dV at49.60%/47.95%
of GPU event time, and dQ at23.73%/22.95% (starting/ending). Bias final reduction
adds6.87%/6.56%. The candidate improves the smaller dQ component; this explains
why its complete-backward benefit is much smaller than its native gain.
Profiling uses CPU+CUDA events and discards instrumented warmups; its totals
are separate from paired timing. There is no new SOL measurement or SOL90 claim.

Reproduce the final summary with `python3 summarize.py`. Raw paired rounds,
source/binary hashes, qualification counts and current-baseline profiles are
retained in the JSON files. All jobs belong to this bounded pass; unrelated
training and normalization work are untouched. All nine owned jobs are
complete:19243/19244 builds and native checks,19245 attribution,
19251/19253/19257 full pilots,19266 qualification,19268 final measurement,
and19277 explicit-entry verification.
