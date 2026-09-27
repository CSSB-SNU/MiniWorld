# Complete training comparison

**Baseline correction:** "Existing engine" in this historical report means our
already optimized **CUDA checkpoint18246**, not the user's original Triton engine.
The earlier implication that this was a comparison against original Triton is
withdrawn. See the [corrected original-Triton comparison](TRITON_COMPARISON.md).

Job 19305, node02 H100 80GB, BF16 activations/projection weights and FP32 LN affine, B=1, C=128, H=4, D=32, dropout=0, every seventh key masked. All eight parameter gradients and input gradient are included; F+B excludes optimizer and gradient accumulation.

PyTorch means the existing eager dense einsum/softmax implementation, captured in CUDA Graph; it is neither torch.compile nor SDPA. Existing engine means installed training checkpoint18246. Candidate means the explicit `reuse_qdo` context with the same training forward. This comparison does not use the separate inference-only forward.

All three arms run in the same process/GPU for each length, 24 rounds x 10 graph replays, after 20 warmup replays. Six rotated execution orders balance position and predecessor. Timings are CUDA events; profiler runs occur afterward.

The pristine Anthropic optimized rows k2b/k2/flash/cuda_sm90a/triattn_native/triattn_exact declare no backward. The engine adapter also rejects training at runtime. Backward-capable stock/cuEq/DS4Sci/SDPA rows delegate to other libraries and are not substitutes for the previously compared Anthropic optimized forward. The release sources were hash-checked against the original checkout. Unsupported cells below are not performance measurements.

## Complete backward (ms)

| L | Direction | PyTorch | Anthropic optimized | Existing engine | Current candidate | PyTorch / candidate |
|---:|---|---:|---|---:|---:|---:|
| 384 | starting | 3.1702 | unsupported | 0.8465 | 0.8452 | 3.75x |
| 384 | ending | 3.2384 | unsupported | 0.8990 | 0.8989 | 3.60x |
| 768 | starting | 19.3711 | unsupported | 4.5554 | 4.6356 | 4.18x |
| 768 | ending | 19.6134 | unsupported | 4.7869 | 4.8452 | 4.05x |
| 1024 | starting | 42.3713 | unsupported | 10.0880 | 9.9857 | 4.25x |
| 1024 | ending | 42.8046 | unsupported | 10.3480 | 10.3744 | 4.13x |

## Complete forward + backward (ms)

| L | Direction | PyTorch | Anthropic optimized | Existing engine | Current candidate | PyTorch / candidate |
|---:|---|---:|---|---:|---:|---:|
| 384 | starting | 5.7152 | unsupported | 1.1980 | 1.1994 | 4.77x |
| 384 | ending | 5.9064 | unsupported | 1.3596 | 1.3558 | 4.35x |
| 768 | starting | 35.8144 | unsupported | 6.4036 | 6.4694 | 5.54x |
| 768 | ending | 36.4916 | unsupported | 6.9852 | 7.0447 | 5.18x |
| 1024 | starting | 76.8922 | unsupported | 13.8428 | 13.7852 | 5.58x |
| 1024 | ending | 78.0703 | unsupported | 14.7714 | 14.8342 | 5.27x |

## Increment over the existing engine

Positive means less elapsed time. Intervals are 5000-resample bootstrap 95% intervals of the median paired reduction. These differences should be interpreted together with the dedicated 64x40 comparison in [the prior qualification](../reuse_20260926/README.md).

| L | Direction | BWD reduction (95% CI) | F+B reduction (95% CI) |
|---:|---|---:|---:|
| 384 | starting | +0.168% (+0.123, +0.188) | -0.053% (-0.105, +0.019) |
| 384 | ending | -0.026% (-0.092, +0.016) | +0.078% (-0.020, +0.136) |
| 768 | starting | -1.182% (-2.491, +1.455) | -0.904% (-2.224, +1.069) |
| 768 | ending | -1.383% (-2.410, +1.811) | -0.445% (-1.681, +0.361) |
| 1024 | starting | +0.991% (+0.028, +1.375) | +0.932% (+0.159, +1.709) |
| 1024 | ending | -0.114% (-0.817, +0.799) | -0.152% (-1.561, +1.571) |

## Independent repeat

The three-arm sequence changes the immediately preceding workload. At L768 in the primary run, order-group median BWD differences span roughly -3% to +3%; the 95% intervals include zero. Therefore the primary arm medians do not establish a small candidate regression or improvement. An independent repeat uses the same timing code and fixtures with more rounds/replays:

| Job | L | Direction | Regime | PyTorch ms | Engine ms | Candidate ms | Paired reduction (95% CI) |
|---:|---:|---|---|---:|---:|---:|---:|
| 19317 | 768 | starting | backward | 19.3606 | 4.6868 | 4.7110 | -0.013% (-1.498, +0.601) |
| 19317 | 768 | starting | forward_backward | 35.6835 | 6.5719 | 6.5501 | +1.075% (-0.794, +1.946) |
| 19317 | 768 | ending | backward | 19.6091 | 4.8653 | 4.8969 | -0.788% (-1.233, +0.415) |
| 19317 | 768 | ending | forward_backward | 36.4092 | 7.0853 | 7.1075 | +0.376% (-0.715, +1.789) |

PyTorch versus either native engine is a clear multi-fold difference. The much smaller candidate versus installed-engine difference is not a consistent whole-workload win in this comparison. Keep the installed engine as the default; the candidate remains an explicit experiment.

## Validation and limits

All outputs/input gradients/parameter gradients are finite. Maximum cross-implementation relative L2 error versus the BF16 PyTorch path is 0.010290, below the predeclared 0.03 comparison limit. Candidate outputs and every gradient are bitwise equal to the existing engine in all twelve regime/direction/length cells. The prior candidate qualification additionally covers independent FP64 adjoints, changed-input graph replay, dropout/SGD/frozen parameters and memcheck/racecheck/synccheck.

CPU+CUDA profiles verify actual CUDA Graph launches and the installed versus candidate dQ symbols. PyTorch has no native dQ launch. The 48 checkpoint18246 source/binary hashes match before and after each length; module/dispatch/primitive source hashes are retained and unchanged. No production dispatch or source is modified.

Graph allocation peaks in raw JSON are cumulative across simultaneously resident arms, so they must not be interpreted as a per-implementation peak-memory comparison. This is a training block timing comparison, not an SOL measurement.

Reproduce: submit `compare.sbatch`, then `python3 summarize.py --job 19305 --repeat-job 19317`. Raw rounds, error metrics, profiles, release metadata and source/binary hashes are in `compare-L-JOB.json`; the initial L384 pilot is job19301. The original benchmark source is frozen in `compare_19305.py`. The current script only changes post-timing profiler replay count from one to three after job19315 failed its ending-BWD dispatch assertion on an incomplete profiler trace. Its two completed starting cells are retained as partial evidence, not a completed repeat.
