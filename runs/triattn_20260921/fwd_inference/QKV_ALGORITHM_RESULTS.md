# QKV fusion algorithm follow-up

Historical checkpoint: `hot6t + front8`, available through `candidate.load("hot6t")`.
The current per-length default and subsequent measurements are in [HEAD4_KV_RESULTS.md](HEAD4_KV_RESULTS.md).
`candidate.load("resident6")` retains the earlier control. Production engine dispatch is unchanged.

Whole FWD versus our previous fusion: L384 essentially unchanged (0.19–0.25% reduction), L768 improves 6.37–6.57%, and L1024 improves 8.46–8.50%. Versus original Anthropic, L384 is 18.78–20.08% faster and L768 is 1.74–3.30% faster, but L1024 remains 1.47–3.45% slower.

This is inference-only native CUDA/TMA/WGMMA. K/V remain in CTA shared memory. Query and gate projections are produced per query batch. Q/K/V/gate/LSE are not saved globally. The new normal path computes both the attention numerator P*V and denominator P*1 on Tensor Cores using identical BF16 P. It avoids repeated row-max updates and output rescaling. A per-warpgroup guard retries unsafe tiles with the existing stable softmax while QKV are still resident. Retries allocate no global projection tensors.

## Qualified complete inference FWD versus Anthropic

Node02 / normal_h100, B1/C128/H4/D32 BF16, 64 AB/BA rounds x 40 graph replays. Same contract and source baseline as [the original comparison](ANTHROPIC_COMPARISON.md): LN, bias, QKV, attention, gate, output projection, residual and orientation. The faster of the two input-preserving Anthropic residual forms is selected per cell. Nontrivial LN weights are BF16-representable to match upstream rounding. Positive reduction means faster.

| L | Direction | Anthropic ms | New fusion ms | Time reduction | Speedup 95% CI |
|---:|---|---:|---:|---:|---|
| 384 | starting | 0.4560 | 0.3706 | +18.78% | 1.2308–1.2316 |
| 384 | ending | 0.4633 | 0.3723 | +20.08% | 1.2507–1.2518 |
| 768 | starting | 1.7608 | 1.7002 | +3.30% | 1.0291–1.0397 |
| 768 | ending | 1.7328 | 1.7009 | +1.74% | 1.0147–1.0228 |
| 1024 | starting | 3.5245 | 3.5773 | -1.47% | 0.9835–0.9866 |
| 1024 | ending | 3.4632 | 3.5825 | -3.45% | 0.9644–0.9680 |

## Paired comparison with our previous resident6 fusion

| L | Direction | Previous ms | New ms | Time reduction | Speedup 95% CI |
|---:|---|---:|---:|---:|---|
| 384 | starting | 0.3717 | 0.3710 | +0.19% | 1.0018–1.0020 |
| 384 | ending | 0.3693 | 0.3684 | +0.25% | 1.0024–1.0026 |
| 768 | starting | 1.8066 | 1.6949 | +6.37% | 1.0649–1.0696 |
| 768 | ending | 1.8081 | 1.6936 | +6.57% | 1.0680–1.0712 |
| 1024 | starting | 3.9085 | 3.5755 | +8.46% | 1.0909–1.0956 |
| 1024 | ending | 3.9092 | 3.5740 | +8.50% | 1.0920–1.0946 |

## Algorithm controls

Pilot times are for screening; qualified times above determine the result. Each cell below is starting / ending ms. Raw paired Anthropic ratios for every control are in the JSON. Candidates rejected for speed did not run the full sanitizer gate.

| Candidate | L384 ms | L768 ms | L1024 ms | Status |
|---|---|---|---|---|
| splitq4 | 0.462 / 0.464 | 2.211 / 2.226 | 4.658 / 4.664 | Not selected; numerical pilot passed. |
| splitq2 | 0.405 / 0.410 | 3.719 / 3.722 | 8.013 / 8.011 | Not selected; numerical pilot passed. |
| hot4 | — | — | — | Rejected: one-key FP64 regression at L128; unrounded denominator versus BF16 numerator. |
| hot4r | 0.428 / 0.430 | 2.116 / 2.119 | 4.332 / 4.339 | Not selected; numerical pilot passed. |
| joint4 | 0.442 / 0.442 | 2.153 / 2.138 | 4.609 / 4.616 | Not selected; numerical pilot passed. |
| hot6r | 0.408 / 0.407 | 2.223 / 2.227 | 4.701 / 4.708 | Not selected; numerical pilot passed. |
| hot6p | — | — | — | Rejected: late-live FP64 regression at L64; peak guard does not cover the error. |
| hot6t | 0.370 / 0.371 | 1.702 / 1.699 | 3.566 / 3.599 | Selected; qualified18586/18598. |
| async6 | 0.370 / 0.374 | 1.898 / 1.901 | 4.075 / 4.057 | Not selected; numerical pilot passed. |
| async4 | 0.405 / 0.410 | 1.921 / 1.922 | 3.894 / 3.937 | Not selected; numerical pilot passed. |
| cluster2 | — | — | — | Rejected: nonfinite output at L768; cross-CTA scratch lifetimes were not synchronized. |
| cluster2s | 0.397 / 0.403 | 1.936 / 1.936 | 4.079 / 4.097 | Not selected; numerical pilot passed. |

- `splitq4`: 4-WG static query CTA; resident KV recomputed across query groups.
- `splitq2`: 2-WG static query CTA; more repeated KV projections.
- `hot4`: 4-WG max-free normal path, FP32 denominator and stable on-chip retry.
- `hot4r`: 4-WG max-free path with scalar sum of BF16 probabilities.
- `joint4`: One N128 WGMMA computes K,V,Q,gate from the same Z load; static query groups.
- `hot6r`: 6-WG variant of scalar BF16 probability sum.
- `hot6p`: 6-WG FP32 denominator with peak-dominance retry guard.
- `hot6t`: 6-WG tensor-core P*V and P*1 from identical BF16 probabilities; stable retry.
- `async6`: 6-WG resident KV with PV outstanding until next QK wait.
- `async4`: 4-WG resident KV with PV outstanding until next QK wait.
- `cluster2`: Two resident-QKV CTAs share bias through TMA multicast.
- `cluster2s`: TMA multicast with cluster-wide projection/bias scratch ownership barriers.

## Validation and numerical limits

Thirty native fixtures at L64/128/384/768/1024 pass the unchanged FP64 gates, including one-key, all-masked, late-live and large-logit cases. Runtime retry counters verify zero retries for normal random inputs and positive retries for extreme logits and bias offsets. A single retrying tile among normal tiles checks barrier phases across query batches. Graph replays change between hot and retry paths without recapture.

At L128/384/768/1024, memcheck, racecheck and synccheck each pass mixed, all-masked and large-logit inputs with zero errors/hazards/warnings. Complete-module validation covers both directions, five masks, sampled FP64, changed-input/weight/mask graphs, fullgraph compilation, inference_mode and no_grad. Initial qualification18586 timed out in the combined L1024 racecheck at 360 seconds; its incomplete racecheck result is not counted. Job18598 reran each L1024 race fixture independently with a 600-second limit, then completed the remaining module checks and timings. Source/binary hashes and actual profile dispatch are verified. No backward qualification is claimed.

Arithmetic is tolerance-qualified, not bitwise identical to the previous online softmax. The all-masked behavior remains zero update, whereas Anthropic uses uniform-mean-V attention; that existing semantic difference is not removed. Cold/hot path guard choices are runtime data dependent. Large-logit fixtures intentionally exercise the slower stable retry.

## HBM traffic after QKV fusion

NCU profiles only the fused core at L768 with five warmups, cache-control none and clock-control none. Use paired full-FWD timings above for selection. The export produced a Python site warning, but the reports and metric rows were written and parsed successfully.

| Core | NCU ms | HBM read+write MB | L2 read+write GB | SM throughput % |
|---|---:|---:|---:|---:|
| resident6 | 1.3809 | 392.8 | 5.363 | 57.40 |
| hot4r | 1.6751 | 298.6 | 5.133 | 60.80 |
| hot6t | 1.2380 | 413.7 | 5.530 | 51.10 |

The scalar hot4r control reduces HBM traffic but runs slower. The selected hot6t runs faster while moving slightly more HBM data. Once global QKV buffers are removed, the remaining on-chip arithmetic, register pressure, issue scheduling and synchronization must also be optimized. Register spills still exist; source-level tensor removal does not imply zero local-memory traffic. No SOL90 claim.

Evidence: [machine-readable results](algorithm-results.json), [NCU data](algorithm-ncu.json), `algo-qualify-18586.log`, `algo-finish-18598.log` and the per-shape JSON files. The benchmark and qualification scripts are `algorithm.sbatch`, `algorithm_qualify.sbatch`, `anthropic_compare.py`, `hot_audit.py`, `check.py`, and `module_check.py`.
