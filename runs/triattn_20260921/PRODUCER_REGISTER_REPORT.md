# Core producer and register follow-up, through job 15052

No new serving speedup. Installed SHA256 remains
`9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225`;
all 12 manifest entries were reverified. Last installed NCU measurements
remain L768 **795.904us / SM61.348992%** and L1024
**1751.712us / SM65.249349%** (job14862). SOL90 is unachieved.

The new producer uses native 5D TMA with the original tensor maps, byte counts,
barrier phases and consumer release ownership. Its K/V warps each keep only
one descriptor/data/free-barrier base live. Bias and K/V loops are rolled;
the corrected `b8v2` variants unroll the eight bias slots within each tile.
The mixed variant assigns complete warpgroups 24/168/160/160 registers:
the first consumer caches all four Q fragments, while the other two keep
the installed three-fragment cache. Its actual emitted allocation now fits
the 65,536-register CTA pool without spills.

## Measurements

Core ratios below are candidate/serving; values above 1 are slower. Each
graph comparison replaces exactly one hot node in the actual captured core
and block, preserving its 1,216-byte argument and all other nodes. Full output
clones match bitwise before and after 24 balanced timing rounds in both
orientations. These are sampled-case checks, not broad qualification.

| Job | Candidate | Starting core ratio | Ending core ratio | Decision |
|---|---|---:|---:|---|
|14983|Early bias release with all-load dependency|1.062891|1.064786|Reject|
|14983|Independent K-ready / V-ready barriers|1.001192|1.001952|No core gain|
|15018|Rolled producer, 32/160/160/160|1.011837|1.009889|Reject|
|15018|Rolled producer, 24/168/160/160|1.019909|1.017595|Reject|
|15052|Eight bias slots unrolled, corrected 32-register producer|1.006771|1.007490|Reject|
|15052|Eight bias slots unrolled, corrected mixed registers|1.018478|1.018664|Reject|

Identical reassembled controls are included in every comparison. Their small
timing variation is retained in the JSONs; it is not a serving speedup.

M64/N64 architecture controls also failed initial latency screening:

| Job | Configuration | Candidate hot | Same-job serving hot |
|---|---|---:|---:|
|14988|Two consumers, three scores, two P buffers, full Q|1259.662us|791.771us|
|14994|Three consumers, two scores, two P buffers, full Q|962.842us|789.007us|

Both pass the initial FP64 RMS ceiling but are not bitwise. Job14989 profiles
the two-consumer version at SM38.844488%, with higher long-scoreboard waits
and L2 traffic than installed. Restoring a third consumer helps this family
but does not beat installed. No broad numerical or sanitizer work was
performed for these slower candidates.

Four current-source PV-before-E variants (three/full Q, before/after bias
release) all produce C7512 serialization under all 12 compiler settings.
They were rejected before GPU execution. Packed-P dependency through the
existing bias-empty arrival also fails all 12 compiler settings.

## Invalid first unroll experiment and the fix

The first `b8` builds did **not** contain the intended producer rewrite.
The generator reused its outer replacement variable inside the bias-loop
transformation, so the final producer replacement silently matched nothing.
Consequently:

- `leanprod32currentb8` equalled installed because the producer change was a
  no-op. This does not establish equivalence of the intended rewritten code.
- `leanprod24q168b8` retained a 32-register producer while requesting
  168/160/160 for consumers: **66,560 required, 65,536 available**.
  A zero-spill ptxas summary did not expose this role-budget error.
- Job15033 timed out during the candidate's first core replay, with no
  candidate correctness or timing result. Job15045 reproduced it under
  CUDA-GDB: one consumer retried `USETMAXREG160`, the other two waited at
  prologue named barrier11, and producers waited on empty queues.

The builder now uses a separate checked producer anchor and verifies source
features. Corrected names end in `b8v2`. The actual SASS now shows producer
24 and consumer allocations 168/160 as intended; job15052 completes normally.
No compiler or TMA-phase defect is inferred from the invalid first build.

The graph harness now validates actual SASS DEC/ALLOC values and declared
role multiplicities **before loading the candidate CUDA module**. It rejects
both a mismatch from intended roles and an overflowing actual register pool.
Four regression cases using real installed, valid mixed and deadlock SASS pass:

```bash
python3 runs/triattn_20260921/core_sol90/test_register_role_gate.py
```

## Evidence and scope

- [Producer results and codegen](core_sol90/producer-register-results.json)
- [Release, N64 benchmark and profile results](core_sol90/release-splitfull-n64-results.json)
- [Actual register-role audit](core_sol90/lean-producer-register-audit.json)
- [CUDA-GDB deadlock diagnosis](core_sol90/lean-producer-debug-15045.log)
- [Producer generator](core_sol90/build_lean_producer_ptx.py)
- [TMA producer implementation](core_sol90/lean_producer_body.cuh)
- [Register gate](core_sol90/register_role_gate.py)
- [Corrected graph measurements, starting](core_sol90/lean-producer-b8v2-e0.json)
  and [ending](core_sol90/lean-producer-b8v2-e1.json)

All owned builds and jobs through15052 are terminal. No serving source,
binary, manifest or expected vectors changed. The new gate improves the
experiment workflow; it is not a claim of core speed or SOL improvement.
