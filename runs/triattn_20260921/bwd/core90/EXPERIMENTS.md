# Backward core experiment journal (2026-09-23)

Target: both native dK/dV+bias main kernel and native dQ at SOL90. **Not reached.**
The fixed baseline is the installed package after job 16663, including shared-Z
weight gradients; all six manifests and their files are in `baseline/`.
Do not compare these incremental gains against an older backward package.

## Design and measurements so far

1. Move the BF16 probability/dS operands from shared memory to WGMMA registers.
   dK/dV still retains dS in shared memory for the cross-row bias reduction.
2. Retire resident K/V or Q after their last read; reuse their shared storage for
   TMA output stores. This removes scattered scalar global stores and needs no
   additional HBM intermediate or shared allocation.
3. dQ can use a cooperative 128-thread CTA with one-key-tile-ahead TMA prefetch,
   four CTAs/SM and no register reconfiguration. Removing the dedicated producer
   from dK/dV is slower and has been rejected.
4. dK/dV R8 uses two consumer warpgroups, four rows per WG, one producer WG,
   deterministic FP32 bias partials. The bias partial buffer is halved. The
   main kernel is slower than R4 TMA, but the whole dK/dV+bias path is faster.
   This is a useful latency reduction, **not** a higher SOL result.
5. Bias FP32 partial TMA stores, dQ grouped rows sharing bias, and cooperative
   dQ key-tile sizes 32 and 128 were measured and rejected; see below.

The intermediate dQ/R8 pair in job 16759 improves full backward by 13.0-15.0% and full
forward+backward by 8.7-10.5%, depending on length/direction. Its input and all
parameter gradients pass. See `README.md` and `installation.json` for the final
installation; this journal retains intermediate decisions.

## Evidence index

All numerical reports referenced below are under `../below80/`, except the full
module reports under this directory and bias cancellation under `../bias_fusion/`.

| Candidate | Pilot | NCU | Qualification | Result |
|---|---:|---:|---:|---|
| bias ds_double | 16685 | - | - | Bitwise, slower; reject |
| bias ds_double_vec4 | 16686 | - | - | Bitwise, about 1% gain; superseded |
| bias rs_double_vec4 | 16701 | 16707 | 16716 | Bitwise, 7% faster; superseded |
| bias rs_tma_store | 16725 | 16732 | 16731 | Bitwise, L768 3.576 to 3.092 ms |
| bias rs_coop_tma | 16741 | - | - | Bitwise, slower; reject |
| bias rs_r8_two | 16749 | 16757 | 16756 | L768 3.488 to 2.903 ms; FP32 partials |
| bias rs_partial_tma | 16760 | - | - | Slower than direct FP32 stores; reject |
| dq key128 (shared dS) | 16695 | - | - | Bitwise, slower; reject |
| dq multicast2 | 16706 | - | - | Bitwise, slower; reject |
| dq rs3 | 16705 | 16708 | 16717 | Bitwise, 10% faster; superseded |
| dq rs_coop | 16721 | 16726 | 16727 | Bitwise, 10-12% less time |
| dq rs_tma_store | 16724 | 16730 | - | Bitwise, faster |
| dq rs_coop_tma | 16748 | 16753 | 16752 | Bitwise, 17-23% less time |
| dq rs_coop_k32 | 16758 | - | - | Slower than K64; reject |

dQ `rs_coop_tma` independent FP64 and none/mixed/one-key/all-masked checks at
L64/128: job 16754. Full paired module jobs: 16737 (R4 TMA + dQ TMA), 16759
(R8 + cooperative dQ TMA). Both use the current shared-Z wgrad on both sides.

Source counters: jobs 16702/16703 baseline; 16736/16735 R4/dQ TMA candidates.
Profile reports must be read with their CSV unit row: NCU mixes MB and GB.

Compile-screen rejections: bias R8 four-WG cooperative has 224/228-byte spills;
bias compact static producer has 32-byte spills; dQ four-CTA static producer
has 152/92-byte spills; dQ cooperative K128/three-CTA has spills and serialized
WGMMA. dQ dynamic-register compact producer job 16700 hung at L64 and was
canceled; do not rerun it without resolving the allocation/synchronization.

SOL is the maximum actual bottleneck throughput, not a synthetic ratio to old
latency. Current L768: R8 main 44.18% (2.653 ms), final bias reduction 86.68%
(0.314 ms), cooperative dQ TMA 76.66% (1.045 ms). Fewer memory transactions
can lower SOL while making execution faster; never claim these as SOL90.

## Subsequent trials

| Candidate | Evidence | Decision |
|---|---|---|
| dQ cooperative K128, two CTAs | pilot16764 | 199 registers, no spills; still slower than K64 |
| dQ grouped2/grouped4, one common pipeline | pilots16761/16763 | Sharing bias alone does not offset CTA coupling |
| dQ bias prefetch | pilot16766; direct16806 | 0.5-1.6% slower than rs_coop_tma |
| dQ paired WGMMA commit | pilot16772; direct16808; qualification16809; FP64 16807 | Bitwise; direct comparison within0.22%; no meaningful gain |
| bias bias prefetch | pilot16767 | Slower than selected R8 |
| bias R4 paired WGMMA | pilot16774 | Slower than R8 end to end |
| bias R8 paired WGMMA | pilot16769; qualification16771; NCU16773 | Improves R8 by about1%; selected as next basis |
| dQ ldmatrix | pilot16816; qualification16821; FP64 16820; NCU16819 | Bitwise, 106 registers, no spills; 77.80% SOL |
| dQ ldmatrix, five CTAs | compile rs_ldmatrix5 | 128/64-byte spills; reject |
| bias R8 staged FP32 P, four cooperative WGs | pilot16826 | Zero register spills, but slower; reject |
| bias R8 staged FP32 P, dedicated producer | pilot16825; direct16834 | Slight L384 gain, slower at L768/1024; reject |
| bias R8 two independent producer warps | pilot16824; direct16832; qualification16837; NCU16833 | 2.30-2.56% faster than rs_r8_pair |
| dQ three TMA issuing warps | pilot16844 | Similar to original cooperative TMA; below ldmatrix candidate |
| dQ warp-level buffer releases | pilot16838 | Slower; reject |
| bias R8 warp-level buffer releases | pilot16836 | Small improvement; test with independent producers |
| bias R8 independent producers + warp releases | direct16849; final-build qualification16856 | Numerical pilot passes, but racecheck fails; do not install without explicit warp memory synchronization |
| dQ grouped2, dedicated producer, two CTAs | compile rs_group2_producer | 128/64-byte spills; reject |
| dQ grouped2/grouped4 independent pipelines | pilots16853/16852 | Bitwise, slower than single-WG ldmatrix; reject |

The R8 FP32 staging trials retain exact FP32 probabilities in shared memory,
then reuse the score registers for dP. The first compile used an AoS shared
layout and was not GPU-tested; `grouped.aos.cu` and `build-aos.log` preserve it.
The measured builds use SoA (`x*128+lane`) shared addresses to avoid bank conflicts.

`wgmma.wait_group.sync.aligned` is not a full warpgroup barrier: its `.sync`
scope is a warp. The release experiments therefore replace each blocking WG
barrier with four warp-leader arrivals, and the producer waits for all four
before recycling a buffer. An initial version omitted `__syncwarp()` before
the elected lane's arrival. Final-build racecheck16856 caught missing ordering
between ordinary shared-memory stats reads and TMA overwrite, despite passing
numerical tests. `rs8_tma` is rejected. Adding `__syncwarp()` in
`rs8_tma_syncwarp` still failed racecheck16866. Both variants are rejected;
the final `rs8_tma_barrier` retains the original full warpgroup barrier and
single-leader arrival. Do not treat the four-arrival experiment or its generator
as qualified, and do not waive the sanitizer failure. The safe parent
`rs_r8_parallel_tma` passed L64/L256 sanitizer job16860.
[PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group).

Parallel NVCC builds in isolated PID namespaces can share PID-derived `/tmp`
filenames. One `rs_parallel_tma` build failed with a missing cudafe temporary.
The builders now use `--objdir-as-tempdir`, which puts intermediate files in
the distinct artifact object directories. Final selected artifacts use new
names, isolated builds, fresh qualification and installed-package checks.

## Qualified installation and low-priority continuation, 2026-09-25

The safe final R8 artifact `rs8_tma_barrier` passed qualification16876, including
L64/L256 memcheck/racecheck/synccheck, full-module gradients, independent FP64
and exact-zero bias cancellation. `rs_tma_ldmatrix` passed qualification16864
and FP64/mask job16862. Candidate paired job16878 improved every tested full
backward and forward+backward case. Both are now installed.

Installed-package job17025 passes all six manifests, wgrad opcheck, ten opt-outs,
cold fullgraph compilation, all-gradient paired measurements at all three lengths
and both directions, BF16 autocast and FP16 native-path refusal. Full backward
is 15.44-16.58% shorter than the fixed job16663 baseline; F+B is 10.49-11.38%
shorter. Attribution17026 accounts for all16/17 kernels and all six weight grads.

Per the user's new constraint, jobs17025 onward in this continuation use
`--nodelist=node01 --qos=normal_h100` (QOS priority100 versus cssb_h1001000).
Every core90 batch template now defaults to that node and QoS. Compilation also
runs inside those allocations. Jobs17039/17040 were canceled and replaced by
17042/17043 to add the installed-only profile flag; they provide no measurements.

Fresh source profiles17027/17028 retain all required barriers. At L768, the
dK/dV main kernel is about51% of starting backward and dQ about20%. R8 reports
roughly30% DRAM throughput, one resident CTA and substantial TMA/barrier wait
samples. Long-scoreboard samples occur at barrier polling too; they are not
proof that HBM is saturated. PTXAS reports small local spills, including producer
TMA address state. Follow-up producer32/producer40 variants preserve all reader
barriers and increase the producer register allowance. Their maximum dynamic
budget is128*40+256*232=64512 registers, matching384*168 initially allocated.

The dQ follow-up waits for the older score WGMMA group, computes probability
while the dP group is pending, then waits before reading dP. It preserves the
CTA barrier before either K/V/bias stage is recycled. Numerical and sanitizer
qualification remain mandatory before any candidate adoption.

| Follow-up | Evidence | Current result |
|---|---|---|
| R8 producer32 | pilot17032, qualification17049, full comparison17050 | All outputs bitwise; 0.61/0.78/0.89% shorter isolated dK/dV path at384/768/1024; full backward0.22-0.73% shorter than installed17025 |
| R8 producer40 | pilot17033 | Bitwise; slight L384 regression, smaller large-shape gain than producer32; reject |
| dQ softmax/dP overlap | pilot17034; qualification17055 and FP6417056 | Bitwise; about0.40/0.62/0.62% shorter at384/768/1024; all gates passed and installed |
| Two R4 CTAs per cluster | build/pilot17054 | Bitwise at64/384/768/1024, but about53/59/60% more latency at384/768/1024; reject before further qualification |

Producer32 reduces PTXAS spill stores/loads from52/68 to20/36 bytes; producer40
reduces them to4/20 but is not faster across the target shapes. Lower spill
counts alone are not a performance result. Combined follow-up job17057 compares
producer32 + dQ overlap directly against the frozen installed17025 package;
the original campaign baseline16663 remains separately preserved in `baseline/`.
`checkpoint17025/` preserves every hashed file and all six manifests before any
further installation. The paired harness now records both arms' binary digests.

The cluster experiment retains four resident rows per CTA and four consumer
warpgroups. Only rank0 reads the two CTAs' dS via shared-cluster instructions,
then writes one FP32 R8 partial; dK/dV remain final outputs. The HBM partial size
matches the installed R8 kernel. Double-buffered readiness and read-completion
barriers protect both shared allocations, and a final whole-cluster barrier
keeps both CTAs alive through the last remote access. The experiment tests
whether extra active consumer warps compensate for DSM and cluster waits.
They did not: isolated path times were0.643/4.543/10.507ms compared with installed
R8's0.419/2.851/6.558ms. The partial HBM volume was equal, so the scheduling and
on-chip communication changes provided no benefit. This rejected candidate has
numerical pilot evidence only, not a full sanitizer qualification.

Producer32 and dQ overlap passed combined17057 with bitwise equality for every
input and parameter gradient. Versus installed17025, all full backward cases
improve0.20-0.60% and F+B0.20-0.46%. Both are installed; `installation.json`
records the current artifact hashes. Actual installed-path final17069 and
attribution17070 passed and remeasure against original baseline16663, without
adding independent experiment percentages. Final backward is15.85-16.96% shorter
and F+B10.48-11.66% shorter. NCU17062/17063 measure46.29%/78.09% SOL for
dK/dV main and dQ; SOL90 is still unmet.

Two further dQ experiments change only the TMA L2 eviction hint: prioritize
reused K/V (`rs_kv_last`, job17075) or reused bias (`rs_bias_last`, job17076).
The local CUTLASS4.2 `Copy_Traits::with` API supplies `CacheHintSm90::EVICT_LAST`;
all layouts, math and synchronization are unchanged. Jobs17071/17072 were
canceled before allocation to fix a C++ template/namespace name collision.
They supply no timing evidence. Both replacement experiments are rejected:
K/V retention is about0.46% slower at384, unchanged at768 and0.79% slower at1024;
bias retention is unchanged within0.11% at384/768 and0.24% slower at1024.
All pilot outputs are bitwise equal. Neither supplies a consistent speed gain,
so neither receives further qualification or changes the installed pair.
All jobs owned by this continuation are terminal; final17069 and attribution17070
remain the current installed evidence. The SOL90 target remains open.

## Next continuation from checkpoint17069 (jobs17127 onward)

`checkpoint17069/` freezes all31 manifest-listed files before this continuation.
Every build and GPU experiment again requests node01 and normal_h100. The
machine-readable `continuation17127-results.json` records artifact hashes,
compiler resources and paired isolated-path measurements. Pilot timing does not
by itself qualify a candidate for installation.

Q16 R8 retains the same deterministic FP32 bias scratch while reducing the
score/dP register fragments. Two-consumer, four-consumer and cooperative
four-consumer controls all lose to the installed Q32 kernel. Four consumers
plus shared FP32 probability staging and independent TMA producers also lose.
These candidates preserve all output bits in the four pilot lengths.

dQ query grouping differs from earlier outer-row grouping: adjacent query
tiles of the *same* outer row share K/V. It creates no global partial and keeps
final dQ ownership. Groups2/4 were tested with a dedicated TMA producer and
with cooperative TMA issue. The dedicated G2 build spills and serializes WGMMA;
G4 and both cooperative builds avoid spills but still lose at the target
lengths. Reduced requested K/V traffic is not a measured HBM saving; the
installed dQ already records an83.12% aggregate L2 hit rate. All four controls
are rejected before further qualification.

Three dK/dV consumer WGs own3/3/2 outer rows with three independent TMA producer
warps. The row iteration, bias FP32 association and output layouts match R8.
The third consumer omits the unused ninth row; its Q ring phases advance by
two actual rows. The cross-consumer barrier counts384 threads, and the first
two consumers keep the installed bias output partition. This avoids staging
persistent gradients or probabilities and preserves every shared-reader
barrier. Pilot17146 is bitwise and reduces the target isolated path by
2.84/2.24/1.64% at384/768/1024. The staged-probability control17147 regresses
9.89/13.38/15.20%. Qualification17154, full comparison17155 and NCU17156 follow
the direct three-consumer candidate; their final disposition is recorded below.

The separate async-bias experiment17151 assigns the two otherwise idle
producer warps to deterministic R8 reduction. A two-stage ready/empty protocol
lets consumers proceed to the next tile while the64 reducer threads read the
previous dS. A full64-thread barrier protects all readers before the empty
arrival. Producer56 plus consumer224 registers use64512 registers per CTA.
The FP32 HBM partial size and summation order are unchanged. Register control
17160 instead allocates56/152 registers to the three-consumer design; its
total budget is65536 and must be evaluated independently.

Final measured disposition in this continuation:17160 spills192/128 bytes and
serializes WGMMA, so it is rejected. Three-consumer qualification17154 passes,
but full comparison17155 gives no L1024 improvement; it is not selected.
Async bias17151 reduces isolated384/768/1024 path times2.64/3.43/4.06%, and
qualification17162 passes every FP64/cancellation/module/sanitizer gate. Full
comparison17163 is bitwise for all input/parameter gradients and improves
backward1.21-2.41% and F+B0.73-1.04% against checkpoint17069. NCU17164 records
2.4288ms and49.61% maximum SOL for its main kernel. SOL90 remains unmet.

node01 became fully occupied before follow-up allocation. The package remains
checkpoint17069 at this queue snapshot. Promotion17175 is queued atnormal_h100:
it rechecks adoption gates, installs the async candidate, performs actual
installed-path verification plus full attribution, and refreshes reports.
Validation failure restores the changed prior package/report files. Q prefetch
depth4/8 experiments17167/17168 remain pending afterok:17175; no timing exists
for those two candidates. `CONTINUATION17127.md` and `promotion-17175.json`
(once started) distinguish qualified, queued and installed states. This pending
continuation supersedes the earlier terminal-job statement for17025-17076 only
in the sense that new jobs now exist; those earlier jobs remain terminal.

## node02 continuation and completed installation (2026-09-25)

The user authorized moving the queued work to node02. Jobs17175/17167/17168
were still pending and were moved in place, preserving their dependencies and
normal_h100 QoS. The promotion runtime guard, report node metadata and all
core90 batch defaults now use node02. Job17175 completed successfully there:
all six manifests, wgrad opcheck, ten opt-outs, cold fullgraph, AMP, complete
all-gradient paired timings and attribution pass. `PROMOTION_COMPLETE 17175`
and `promotion-17175.json` state complete are the installed evidence.

The installed async-bias package reduces full backward17.47-18.20% and
F+B11.25-12.53% versus original16663 in that same node02 job. The independent
incremental comparison17163 against checkpoint17069 remains1.21-2.41%/0.73-1.04%;
do not add these percentages. All31 hashed package files are frozen in
checkpoint17175 for subsequent comparisons.

Fresh node02 profiles17190/17191 match installed binary hashes. dK/dV main is
2.424544ms, compute45.79%, memory49.35%; dQ is1.016896ms, compute62.64%,
memory78.62%. The unchanged final bias reducer reaches93.03% memory SOL in this
profile. This does not mean either target core kernel reached SOL90.

Q/dO prefetch-depth4/8 pilots17167/17168 compare against the newly installed
async path. New dQ pilots17193/17194 reuse the current8192-byte bias shared tile
for half of the FP32 probabilities, leaving the other half in registers.
A full CTA barrier completes every ldmatrix bias reader before reuse; the
original end-of-tile barrier still protects stage recycling. No HBM tensor or
shared allocation is added. Serial score/probability/dP scheduling is tested
with launch_bounds compiler targets of four versus five CTAs/SM. These targets
do not cap actual residency; both builds use96 registers and no spills.

Both half-probability dQ controls are rejected:17193 is3.58/4.77% slower at
L768/1024 and17194 is1.73/2.85% slower, despite small L384 changes. Pilot17208
adds dP WGMMA overlap with ex2 on the retained half of the logits; its result
is recorded separately and is not included in those controls' timings.

Q/dO depth4 wins:pilot17167 is bitwise and reduces isolated target path time
13.96/14.04/13.10% at384/768/1024 against installed17175. Depth8 is also faster
than17175 but slower than depth4 at every target length, so it is not selected.
The depth4 build has no local spills. Qualification17203 passes module10,
FP64/masks24, exact cancellation and all six sanitizer runs. Combined17204 is
bitwise for every input/parameter gradient, improving full backward4.88-6.71%
and F+B2.85-4.21% relative to checkpoint17175. NCU17205 records2.070592ms,
compute54.00%, memory55.70% for the main kernel. Adoption gates pass; installed
verification and attribution are assigned to promotion17211 on node02.

Promotion17211 completed on node02 and `rs8_async_q4` is now installed.
Its installed-path checks include all manifests, wgrad opcheck, opt-outs,
cold compile, all-gradient timings, AMP and complete16/17-kernel attribution.
The final installed comparison against original16663 reduces full backward
21.92-23.23% and F+B14.63-16.00%. dQ overlap control17208 is bitwise but
1.47/3.08% slower at768/1024, so it is rejected before further qualification.

The paired harness now accepts an explicitly named frozen baseline checkpoint
and verifies all its file digests. The campaign report rejects any baseline
other than original16663, preventing checkpoint comparisons from being mixed
into that table. Job17212 directly compares the final installation with this
turn's starting checkpoint17069; `NODE02_CONTINUATION.md` records that result.

All13 jobs in this node02 continuation finished COMPLETED/0:0 on node02 at
normal_h100. Direct job17212 measures7.05-7.90% full-backward and3.89-4.91%
F+B reduction from checkpoint17069 to final17211, every gradient bitwise equal.
No jobs from this continuation remain queued or running. See node02-jobs.json
and scheduler-node02.txt; no SOL90 claim is made.
