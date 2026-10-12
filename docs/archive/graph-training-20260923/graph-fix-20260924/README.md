# CUDA graph gradient discrepancy: D64 buffer reuse fixed

## Current result

The cause found was a shared-memory input-buffer reuse race in the native D64
Transition backward, used by MSA and template blocks. The two warpgroup leaders
released a ping-pong buffer before every warp had completed the ordinary shared-
memory reads in the LayerNorm/residual dX epilogue. A leader could start the next
TMA load while a slower warp still read the previous tile. dX changed; that
kernel's own parameter gradients remained identical. The error then propagated
upstream into PWA and template gradients. It was not fixed by BF16 GEMM flags.

The fix adds one 256-thread named barrier before releasing the input slot.
It adds no HBM tensor, no copy, and no arithmetic/precision change. D64 gets a
new binary cache identity so an old compiled extension cannot bypass the fix.
Small prior tests had fewer tiles than input CTAs, so they never exercised buffer
reuse; the new test uses the real MSA shape (1024 x384 x64), 30 ordinary/graph
backwards and exact output-gradient comparisons.

Evidence:

- Old kernel under racecheck instrumentation: input-gradient relative variation
  up to0.02179466; every parameter gradient unchanged. Racecheck itself reported
  no hazards, so its summary alone did not detect this synchronization bug.
- Fixed kernel: all gradients bit-identical across30 ordinary and30 graph runs,
  including the same racecheck experiment (0 reported hazards).
- D64 GPU tests:8 passed, including the large persistent-buffer regression.
- Previously failing saved full-model input: validation passes before/after Adam.
- Fresh8-GPU data, random recycle1–4,32 accumulated backwards and empty-template
  cases: all8 ranks pass. Maximum per-parameter relative L2 is2.554e-6 before
  Adam and2.802e-6 after. Losses exactly equal. Original tolerances retained.
- Two diagnostic optimizer updates complete after model/Adam restoration; these
  diagnostic runs did not write production checkpoints or W&B.

Short cached-input CUDA-event benchmark,25 graph replays per recycle, excluding
DataLoader/NCCL/optimizer:

| Recycle | Before (ms) | Fixed (ms) |
|---|---:|---:|
|1|64.501|64.404|
|2|80.764|80.789|
|3|97.090|96.795|
|4|113.171|113.266|

Largest increase is0.094ms (0.084%); no material slowdown in this sample.
See [summary](barrier-fix-summary.json) and [samples](barrier-speed.json).

Production jobs submitted on node02,8 H100 each: phase1a16976 resumes epoch477 /
step47700 with compile + CUDA graphs; phase1b16977 has afterok:16976 and also
uses compile + graphs. Both use the existing run and W&B `team_gm/MiniWorld/tapgki9e`.
The new snapshot freezes source and the validated extension binaries with SHA-256
verification. Each production phase still performs its strict preflight before
an optimizer update. Startup/training status is recorded separately below.

The investigation notes below are historical, including rejected hypotheses.

## Historical findings

This was not established as a graph-only incorrect-gradient bug. Repeated
ordinary executions also disagreed with the first ordinary reference. In job16931,
402/436 parameters had identical error metrics in graph-vs-reference and
ordinary-repeat-vs-reference. The aggregate gradient relative L2 was6.36e-6;
the failing per-parameter maximum was0.002677. Near-zero gradients are sensitive
to cancellation; the global norm must not replace parameter-level checks.

Local gradients before NCCL already differed on one rank. NCCL Ring/Simple did
not fix it. Workspace-only and same-stream-only single-GPU successes did not
survive all eight ranks. They are not sufficient fixes.

Microbatch-level comparison on the saved failing input found the first tracked
MSA LN-bias discrepancy at microbatch24 (zero-based23), recycle4, after an Adam
update. The first23 accumulated values of that tracked gradient matched exactly.
Disabling autograd multithreading still failed. Disabling BF16 reduced-precision
GEMM reduction passed, as did a separate deterministic-algorithms experiment.

This implicates GEMM reduction precision/algorithm-dependent rounding, which
propagates into small MSA and template gradients. It does not identify a single
cuBLAS kernel or prove that every prior mismatch has the same internal kernel.
The earlier claim that fixing workspace alone solved the issue was premature.

## Candidate configuration and limited validation

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, before CUDA handles initialize.
- `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`.
- Run the ordinary reference on the capture stream; order caller/side streams
  explicitly. This also eliminates the AccumulateGrad stream-mismatch warning.
- Warm the same input/recycle sequence before establishing the reference.
- Keep the original per-parameter tolerance and exact loss comparison.

BF16 operands and outputs remain BF16. The precision flag disallows reduced-
precision intermediate GEMM reductions; it does not turn the model into FP32.
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms)
and [NVIDIA cuBLAS reproducibility](https://docs.nvidia.com/cuda/archive/13.0.2/cublas/index.html#results-reproducibility)
describe these controls.

Eight-GPU job16936: all8 ranks,32 accumulated backwards, recycle1–4, real saved
inputs plus empty-template cases, before/after an Adam update. Losses match
exactly. Maximum parameter relative L2:

| Stage | Relative L2 | Percent |
|---|---:|---:|
| Initial |0.000849738|0.084974%|
| After Adam |0.000016461|0.001646%|

The criterion remains relative L2<0.001, with the pre-existing tiny-gradient
absolute-error exception. No threshold was loosened. Two diagnostic optimizer
updates then completed; these used cached input and did not write production
checkpoints or W&B. This is correctness evidence, not full-data throughput.

Eleven CPU tests passed for graph inputs and early workspace configuration;
targeted F/E9 lint passed. Production job16937 performs its own strict preflight
on newly loaded real batches, then resumes epoch477/step47700 with the original
W&B run `team_gm/MiniWorld/tapgki9e`.

See [A/B outcomes](ab-summary.json), [all-rank summary](eight-gpu-summary.json)
and [full rank0 validation](eight-gpu-validation.json).


## Follow-up: the candidate did not generalize

Production preflight16937 failed on newly loaded inputs before training resumed.
Maximum parameter relative L2 was0.007466 (MSA LN-pair bias); template projection
weights also exceeded0.001. Consequently the reduced-precision flag is NOT an
established root cause or a complete fix. All production training remains stopped.

Short benchmark jobs16938/16941: 25 CUDA graph replays per recycle, cached same
input, no loader/NCCL/optimizer. On/off results differ by less than0.2%:
R1 64.397/64.501ms; R2 80.667/80.764ms; R3 97.161/97.090ms;
R4 113.046/113.171ms. No material slowdown demonstrated by this short sample.

Shape-matched standalone probe16943 used synthetic operands. PWA dW BMM
[8,384,32768]@[8,32768,384] selected the same `nvjet_tst_128x80_64x8_1x1_v_bz_TNT`
kernel under both flags, with exactly equal outputs and0.243/0.245ms latency.
The template mask-projection dW shape likewise used identical split-K kernels
and exactly equal outputs (0.0474ms). This does not establish behavior of every
GEMM or every real operand/alignment.

Instrumented PWA backward probes16939/16940/16942 compared dO/V/bmm output and
LN gradients (and a32-backward accumulated comparison). Those instrumented runs
matched and passed. Instrumentation changes allocation and scheduling; the
unmodified failure's first divergent operator has NOT been located. Do not
claim graph correctness, a complete fix, or causal proof from the precision flag.

See [short timings](short-probe-summary.json) and [GEMM kernel names](gemm-probe.json).

## Further localization (jobs 16944-16962)

Neither deterministic-algorithm mode nor preallocated owned `.grad` buffers
resolved failures on fresh inputs. Changing the PWA dW BMM output to FP32 passed
one saved eight-rank dataset and the nine PWA integration tests, but failed the
next fresh dataset. That candidate is NOT deployed.

Profiler job16951 found identical mathematical kernel names, launch geometry,
and order between ordinary execution and graph replay. Profiling itself removed
the reproduction, so this is not proof of equivalence on the failing invocation.

Standalone checkpoint-weight repetition tests (30 ordinary +30 graph runs each):
PWA, TriMul, OPM, Pair Transition, MSA Transition all produced bitwise-identical
input gradients. PWA LN-MSA parameter reductions varied by about7e-7 relative
(atomic FP32 sums); PWA pair gradients were identical. Repeating PWA backward
with three actual model contexts also gave identical input/pair gradients.
Targeted racecheck of one PWA glue3 and one dgv backward launch each reported
0 hazards/errors/warnings; this is limited coverage, not a whole-model sanitizer.

Boundary diagnostic16959 reproduced a discrepancy on saved rank2 input at
microbatch8 (zero-based), R1, empty-template case. All final logits remained
bitwise identical for all32 microbatches. Parameter differences first appeared
at MSA block1 PWA and propagated towards block0 and the input embedder. MSA
block1 Transition and later blocks remained unchanged at this boundary.
The next experiment records PWA input/intermediate/output fingerprints to
localize this backward discrepancy. Training remains stopped pending resolution.

## Production confirmation

Job16976 passed its own fresh8-rank initial/after-Adam validation, restored the
checkpoint, resumed W&B run `team_gm/MiniWorld/tapgki9e`, and reached step47706
with finite losses. Phase1b16977 remains pending on afterok:16976. Its L768
preflight will run when phase1b starts, with epoch800/step80000 asserted.

W&B already contained partial pre-checkpoint-resume logs through step47702.
It rejected the first duplicate step after restoring checkpoint47700; subsequent
steps continued in the same run. No new W&B run was created, and no old W&B
history was deleted. The existing train.log and Slurm log are appended.

See [live-resume verification](production-resume.json).
