# TriangleAttention optimization — first Codex pass

> Superseded by the CUDA/TMA surround implementation and measurements in [CUDA_REPORT.md](CUDA_REPORT.md). The text below preserves the first pass evidence.

Applied to `runs/triattn_20260921/oc/opt_core`; no environment overrides required. Measurements are forward block CUDA-graph replay, out-of-place fused residual, B=1/C=128/H=4/D=32/bf16 on an H100 80GB. This is not a training-step or backward benchmark.

## Performance

Each value is the median of three timing rounds; each round uses the established graph harness. Baseline and candidates alternate inside one process and Slurm job 13767 on node01. `Previous default` includes the pre-existing residual fusion but lacks environment-only tuning. `Handoff best` is BN64/maxnreg128 + epilogue 8x8.

| Direction | L | Previous default (µs) | Handoff best (µs) | Final (µs) | Time saved vs default |
|---|---:|---:|---:|---:|---:|
| starting | 384 | 367.6 | 329.1 | 321.1 | 12.66% |
| starting | 768 | 1641.8 | 1526.6 | 1515.4 | 7.70% |
| starting | 1024 | 3328.4 | 3124.2 | 3103.7 | 6.75% |
| ending | 384 | 366.0 | 327.6 | 327.8 | 10.42% |
| ending | 768 | 1631.1 | 1528.8 | 1527.1 | 6.38% |
| ending | 1024 | 3292.1 | 3130.2 | 3117.7 | 5.30% |

Starting-direction time saved beyond the handoff best: 2.43% / 0.73% / 0.65% at L384 / L768 / L1024. These are small gains; individual L768/1024 timing rounds vary by tens of microseconds. Ending uses identical settings to handoff best; its difference is measurement variation, not an additional optimization.

## Changes

- Fixed `pair_fused.prologue` filtering out BN/maxnreg before launch. Merely changing JSON, as suggested in the handoff, would not apply the optimization. Forward the supported optional launch settings through the serving path.
- Persisted BN=64/maxnreg=128 in r07 and BI=BJ=8 in r16. No epilogue register cap.
- Added opt-in `j_fast` CTA traversal in the prologue. Adjacent j tiles execute first for starting attention; ending retains its original traversal. Arithmetic and storage layout are unchanged.
- Added `starting_tile=[1,64]` to r16. Starting attention uses contiguous j runs; ending keeps 8x8.

## Correctness and integrity

- All six direction/length combinations passed exact output equality against previous defaults, using randomized nonzero output weights. Each also checked fresh inputs with dense, fully masked, and irregular masks: 24 input/mask cases total.
- fp32-reference relative RMS remained identical for every configuration within each case.
- Actual core kernel verified in every timing profile: Triton `_fwd` at L384 and CUDA `triattn_m1_kernel` at L768/L1024. No `refused by kernels.triattn` in any job 13763–13767 log.
- Core source and binary unchanged. Shared binary SHA256: `3c766e0296351d643329c7203f94f25ad3c584c5d400b1ca92b74d35fdb225ba`.
- Qualification stack: torch 2.10.0+cu128, Triton 3.6.0. Other stacks and backward/training integration have not been requalified.

## Rejected experiments

- Changing tile aspect ratio alone: small component wins, inconsistent block-level gains; retained only the starting epilogue tile as part of the verified final combination. See step29/step32.
- Splitting projection output chunks into separate CTAs: bitwise correct but much slower, L768 block 4.37–4.61 ms versus 1.53 ms handoff best. Reordering these CTAs to reuse the same input tile did not rescue it (4.28–4.49 ms). Not enabled. The implementation was tested; a hardware cause was not established. See step30/step31.
- The saved NCU report also contradicts the handoff statement that both surround kernels use two CTAs per SM: the epilogue reports a register limit of three. Avoid using that statement as a premise for further tuning.

## Reproduction

```bash
sbatch runs/triattn_20260921/step33.sbatch
```

Outputs: `logs/step33-e{0,1}-L{384,768,1024}.json`; exact timing rounds, kernel census, config and correctness checks are included. The job uses `qualify_final.py` and `cells_before_codex.json` for the baseline. Do not mutate the serving tree while a job is running.

`codex_optimization.patch` contains only this handoff continuation’s production changes, relative to the tree at takeover. Original kernel files and cell table are retained as `*_before_codex.py` / `cells_before_codex.json`. Experiments remain separate from the serving path.
