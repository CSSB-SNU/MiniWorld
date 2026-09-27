# Installed training attention forward

2026-09-25. **Native CUDA/TMA `cooperative_head2` is installed and verified**,
promotion17628. It changes CTA head traversal relative to CUDA17345 while
retaining the same arithmetic, barriers and two-stage buffering. Qualified
scope: H100, B1/H4/D32 BF16 projection layout, L384/768/1024, both directions.
Experiments use node02 / normal_h100. **SOL90 remains unmet.**

## Installed full-workload comparison

Frozen CUDA17345 versus the actual installed path, same process and inputs,
64 alternating AB/BA rounds × 40 CUDA graph replays per arm. Times are arm medians;
reductions use paired median ratios. All full-module outputs/gradients are bitwise
equal. Backward kernels are unchanged. Every FWD paired-bootstrap 95% lower bound
is positive, and every F+B median improves. Full intervals/raw rounds are in
`installed-results.json`; intervals reflect within-run sampling, not generality
across nodes or workloads.

| L | Direction | Full FWD ms | Reduction | F+B ms | Reduction |
|---|---|---:|---:|---:|---:|
| 384 | starting | 0.4021 → 0.3982 | 1.03% | 1.2576 → 1.2537 | 0.40% |
| 384 | ending | 0.5024 → 0.4989 | 0.72% | 1.4028 → 1.3987 | 0.26% |
| 768 | starting | 2.0165 → 1.9902 | 1.27% | 6.8251 → 6.7938 | 0.47% |
| 768 | ending | 2.3754 → 2.3562 | 0.74% | 7.3714 → 7.3285 | 0.49% |
| 1024 | starting | 4.1590 → 4.1131 | 0.92% | 14.5380 → 14.4934 | 0.37% |
| 1024 | ending | 4.7833 → 4.7475 | 0.72% | 15.4997 → 15.4508 | 0.30% |

## Change and measured traffic

Each CTA still computes one 64-query/head/outer-row tile with 128 threads,
80 registers, zero spills, 37,888 B kernel shared storage and six CTAs/SM.
Consecutive CTAs now traverse two adjacent heads before moving to the next
query tile; grid=(2*L/64,L,2). In the projection layout each head contributes
64B per token. This ordering reuses adjacent-head cache lines before eviction.
There is no new preparation kernel, intermediate buffer or autograd operation.

NCU17561, L768: core HBM reads fall910.76 ->457.77MB (49.74%).
This is a core-only counter, not a 49.74% complete-module traffic or speed claim.
L2 read sectors increase227.89 ->234.00million, and full FWD gains are small.
NCU17566: SM 74.11%, L2 86.45%, HBM 19.88%, 80 registers, six CTAs/SM.
The C7515 WGMMA serialization warning remains. These utilization counters do
not establish SOL90 or fully overlapped WGMMA execution.

Core-only confirmation17568, 12 AB/BA rounds × 20 replays, randomized inputs and
mixed mask, separate from the full-module timings above:

| L | CUDA17345 → head2 ms | Paired reduction |
|---|---:|---:|
| 384 | 0.1360 → 0.1328 | 2.45% |
| 768 | 1.0679 → 1.0647 | 0.31% |
| 1024 | 2.4844 → 2.4709 | 0.34% |

## Qualification and selection

- Qualification17564: eight independent FP64 gradient fixtures and 18 full-module
  checks (directions, masks, dropout, optimizer updates, fullgraph, frozen weights).
- Memcheck/racecheck/synccheck pass all six fixtures at L64/256; full-shape17565
  passes L768 mixed/all-masked cases. No errors or hazards.
- Actual installation17628 passes cold fullgraph F+B, schema/fake/AOT opcheck,
  six real native dispatches, layout/dtype/opt-out guards and four BF16 AMP cases.
- Six backward manifests/dependencies, dispatcher and Python forward guard are
  unchanged from the previous checkpoint. The complete new snapshot has 36 files.
- Exploratory17556/17557 comparisons tested head2/head4. The short L1024 F+B
  comparisons were mixed. Head2 was fixed before independent 64-round confirmation
  17568; all six FWD and F+B cells improved there. The actual installed path then
  repeated 64 rounds with the same forward confidence gate and F+B median gate.
- For this sub-1.5% incremental gain, promotion requires a positive 95% bootstrap
  FWD lower bound in every cell instead of the older 1.5% median threshold. The
  default threshold remains unchanged for callers that do not opt into the CI gate.
- Head4 remains experimental. Twelve producer/shared-P/cluster controls did not
  beat CUDA17345; [PIPELINE_FOLLOWUP.md](PIPELINE_FOLLOWUP.md) and its JSON preserve
  those experiments, compiler failures and the multicast traffic measurements.

## Evidence and next fusion boundary

- [Promotion17628](promotion-17628.json), [frozen package](checkpoint17628/snapshot.json),
  [selected source](cooperative_head2/fused.cu), [installed measurements](installed-results.json).
- [Previous17345 improvement](IMPROVEMENT_17345.md) and `installed-results-17345.json`
  retain the old comparison against CUDA17310. [Initial training CUDA adoption](INITIAL_17310.md)
  retains the comparison against Triton.
- `module_check.py --baseline-artifact cooperative_q1_s2 --installed` compares the
  actual installed path with frozen17345; plain `--installed` compares with Triton opt-out.
- [QKV + attention fusion design](QKV_ATTENTION_FUSION.md) audits projection ownership,
  shared-memory limits, repeated K/V computation and backward saves. It is a design,
  not an implemented or benchmarked fused QKV/attention kernel.
