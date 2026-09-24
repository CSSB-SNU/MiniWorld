# Eight-GPU Phase1 CUDA graph integration and latency analysis

All summary latencies below are **ms per microbatch**. For an actual training
loop, this is optimizer-step wall time divided by accumulation32; it is not an
isolated kernel measurement. Effective batch is 256 on eight H100s, PF16,
L384/atom4096, MSA1024 sampled from a pool of8192, random recycle1–4.

## Measured improvement

| Boundary | ms / microbatch | Evidence |
|---|---:|---|
| Previous actual production epoch | 164.22 | 525.5s / 100 optimizer steps / 32 accumulation |
| Original Fabric profiling, steady steps3–4 | 171.03 | fabric-summary.json |
| CUDA graph, real DataLoader, 8 updates | **101.85** | real-summary.json |
| Of which graph fwd+bwd GPU interval | 87.42 | real-timing-rank*.json |
| Resumed production, complete epoch472 | **107.77** | 100 updates, no fast-sample selection; completed-epoch.json |

The real-data diagnostic saves **62.37ms**, or **1.61x**, against the previous
production epoch. It includes real input loading, H2D, NCCL, Adam and EMA. It
does not write W&B or overwrite the production checkpoint. Eight updates are a
short sample; the resumed production measurements are reported separately.

The original “50ms gap” subtracted an isolated benchmark from a real training
loop and did not match all precision/recycle boundaries. We therefore judge the
fix by actual eight-GPU training-loop wall time, not by treating that estimate
as an exact additive breakdown.

## Where time goes

The previous loop repeatedly allocates/moves a batch, submits normal compiled
forward/backward work, and calls loss.item() each microbatch. Its CPU call
measurements averaged data_next9.67ms, Batch.to13.81ms, forward/loss/item63.07ms,
backward41.74ms. Optimizer host time included outstanding backward/NCCL/rank waits,
not just Adam GPU work. These overlapping CPU/GPU intervals must not be summed
as independent GPU compute.

The new loop uses pinned input prepared in a DataLoader thread and copies into
stable GPU buffers. It captures four independent graphs for arbitrary recycle
order, accumulates gradients and loss on GPU, and communicates/logs once per
optimizer update. All ranks update their FP32 EMA from identical live weights,
removing the old per-parameter EMA broadcasts.

Real-data graph GPU intervals per microbatch:

- Input copy: **1.53ms**.
- Forward+backward replay: **87.42ms**.
- Communication including rank waits: **9.62ms**.
- Optimizer/EMA: **1.13ms**.

CPU data_next averages4.83ms and overlaps execution. On update2 rank2 waited for
data; updates4/6 were delayed by rank3; update7 by rank4. Other ranks waited at
NCCL. The continuation doubles prefetch from4 to8 with4 workers/rank. A trial
with8 workers was stopped during startup because duplicated datasets consumed
more than700GiB host memory; no optimizer update or W&B initialization occurred.

Additional Fabric instrumentation disproved an earlier hypothesis: deleting the
autograd graph costs only about0.24ms/microbatch. A later diagnostic had a
10491ms forward outlier on rank0/micro111 (plus1088ms backward); the whole update4
is excluded from its steady comparison. Update3 measured146.19ms, with about
16.86ms still unassigned by the narrow host wrappers. We do **not** claim to
have uniquely attributed every old millisecond or use the outlier-inflated mean
as the speedup baseline. See fabric-release-summary.json and its raw rank logs.

## Numerical validation and preserved training state

- Production native mixed dtypes are preserved; no new global BF16 autocast.
- Graph loss is divided by accumulation before backward; NCCL averages gradients
  once per optimizer update, retaining captured gradient buffers and strides.
- Eight ranks, 32 accumulated backwards, changed real batches and empty-template
  inputs, tested before and after an Adam update. Losses agree exactly; maximum
  per-parameter gradient relative L2 in the real-data run was1.8722e-5.
- Validation requires relative L2<1e-3. Only near-zero reference gradients
  (max<=1e-6) may instead pass max absolute error<=1e-8. This is based on a
  separate LN-bias diagnostic: graph absolute difference1.76e-9 versus ordinary
  repeat6.61e-9. See near-zero-gradient.json.
- Template count0 cannot simply be padded: the old embedder processes padded
  slots through its query path. The graph explicitly gates the full update to0,
  checks old zero-template output/gradients, and preserves Adam grad=None for
  template parameters unused throughout an entire global optimizer batch.
- Sparse bond/contact lists are not consumed in Phase1; dense token_bond_feat is
  required. Consumed fields are copied, and other shape/dtype mismatches abort.
- Full model, Adam and scheduler are restored after validation. FP32 EMA and
  epoch/global_step resume from the same checkpoint.
- Six input-copy unit tests pass. The F/E9 static checks pass for the root trainer;
  full repository style lint still reports annotation/convention issues in this
  standalone trainer and its tests.

## Continuation and provenance

Historical snapshot: jobs16917/16916 below have been superseded. See
[the September24 loader report](../dataloader-20260924/README.md) for the
checkpoint handoff and pending restart validation.

Frozen checkpoint: epoch471, global_step47100.
SHA256: `1e73852fe97622721a2ce9c79df6584720af604b0e72d2b0884a0bfc1201c60c`.

Phase1a **16917** uses `graph_resume_20260924_prefetch`, model/data sources from
`engine2_resume_20260923`, engine55304541, worker4/prefetch8. It resumes the
existing logs and **team_gm/MiniWorld/tapgki9e** with W&B resume=must.
Phase1b **16916** depends on afterok:16917 and requires epoch800/step80000.
Phase1b L768 still uses the previous Fabric path; the L384 graph validation does
not establish L768 graph support.

Attempt16843 failed before optimizer updates on empty-template shape mismatch.
Attempt16915 was canceled during worker8 startup for host-memory use. Analysis
allocation16858 has been returned. Current production status and measured live
latency are in production-summary.json and ENGINE_TRAINING_PROFILE.html.

## Verified completed production epoch

Epoch472 completed and atomically saved a full checkpoint at global_step47200:
436 Adam parameter states and436 FP32 EMA tensors. The complete epoch averaged
**107.77ms/microbatch**, down **56.45ms** from164.22ms (**1.52x**). This includes
all100 updates, data loading and loop/logging costs inside the epoch; it is not
the optimistic91.9ms initial-window measurement. Further epochs continue in
job16917; Phase1b16916 remains queued after successful Phase1a completion.
Production startup independently passed16 distributed validation cases, exact
losses and max per-parameter gradient relative L2 of1.014e-5.

CPU data preparation and rank waits remain a bottleneck; the work does not
claim zero GPU idle time or less than3% idle from these event timings.
