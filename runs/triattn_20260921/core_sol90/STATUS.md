# Closed unsuccessful: calculation-kernel SOL90

User decision: stop the additional core SM SOL90 search. Preserve the installed
qualified core and completed surrounds/mask/SAFE work. SOL90 remains unachieved;
no further optimization experiments without a new user request. Entries below
are chronological historical records, including superseded active-work notes.

**2026-09-22 counter correction:** The H100 microprobe `xu_probe.cu`, job14280,
measured XU instructions for exp2, BF16 conversion, PRMT, SHR, LOP3, ADD, and
video add/min separately. Only exp2 uses the measured XU counter. BF16
conversion and PRMT both count ZERO. Earlier statements here or in HANDOFF
that BF16 packing shares the XU pipe with exp2 are incorrect. Replacing the
packing operation cannot lower XU work; exp2 itself is the relevant operation.

User explicitly reiterated SOL90 on 2026-09-21. A persistent goal is active.
SOL90 is not achieved. The installed package now includes exact L768 and L1024 specializations,
independent K/V producers, corrected barriers, N40 PV/row-sum fusion with original
V TMA, and constant K/V descriptor offsets for both qualified hot paths.
Installed SHA 9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225.
Latest job14862 extends the exact pipeline to L1024 hot536870912: paired core
reductions 4.163%/3.971% and block 2.444%/2.494%, starting/ending (14851).
L768 hot1073741824 machine words unchanged from4be; its preceding0.45% gain
remains. Final14856: 82 bitwise cases and three fullN1024 strided sanitizers,
five cases each, zero hazards/errors. Installed14862: both directions/lengths,
five masks, bitwise default/optout, expected native flags,17/17 shipped tests.
NCU14862: L768 795.904us / SM61.348992%; L1024 1751.712us / SM65.249349%.
q1024-results.json and ../Q1024_REPORT.md record the current result.
SOL90 remains unachieved. See ../CORE_SOL90_REPORT.md.
Earlier timestamped entries retain their historical state, not current status.

## New standalone family

`build.py VARIANT` generates a unique CUDA namespace and extension per variant
from `overlap_template.cu` / `overlap_v*.cu`, compiles on CPU with the usual
`runs/anthropic_adoption_20260919/env.sh` wrapper. `bench.sbatch` takes exported
`SOL_VARIANT` (default r6n32), `LENGTH` (default768), includes current serving core,
checks actual names and an FP64 row reference (max 1.05x RMS), then three graph
rounds. It loads the existing .so directly, never JIT-builds on GPU.

- r2n64: two consumers, partial producer warp, two-CTA target, 3 KV/bias slots,
  two score fragments with QK one chunk ahead. C7511 + large spills; job14068
  hot3106us vs current840us; passes initial FP64.
- r6n32: six consumers80 regs, full producer24, one CTA / 896threads,
  3 slots, two scores. C7515 silently serialized every WGMMA; job14067
  core entry2029us vs910us, FP64 passes. r4n32/r5n32 likewise C7515, not benchmarked.
- v2: fixed issue sequence (always two commits, harmless phantom QK past end),
  explicit prologue. Still C7514 across loop backedges.
- v3: drain and fence all accumulators every six chunks. Removes C7514/15;
  r6 consumer80 too small, C7512. v4 full unroll fixed L768 likewise C7512.
- v5: remove per-logit infinity checks and dynamic reseeding; seed only the first
  chunk, invalid seeds request SAFE. r4n32v5 has no spills/serialization warning,
  but job14075 hot1991us vs846us: small TMA transactions/protocol are costly.
  r5/r6 still C7512.
- v6: same plus scalar FP32 denominator instead of ones MMA in hot pass only
  (numeric change). r6 still C7512. Not benchmarked.
- v7: load K/V/bias in 64-key tiles, consume32-key chunks, release only after
  both chunks finish. r4n32v7 no warnings/spills; job14090 hot1073us vs842us.
  NCU job14091: hot1.056ms, SM46.28%, tensor25.17%, issue43.70%, occupancy26.28%,
  L2 196642319 sectors /6.293GB. MIO throttle1.34 perissue; long scoreboard1.78.
  First NCU job14076 failed due Python source splitting inside a string; fixed
  roof.py splits at a real newline before the main no_grad block.
- v9: three score fragments, two packed P buffers, two-chunk QK lookahead,
  prefetch next bias after PV commit, one wait2 per body. r4n32v9 C7511 due
  register pressure; job14112 hot1513us vs842us, FP64 passes. Rejected.

## Current independent design: split4

`specialized.cuh` adds a fully separate Tensor MMA warpgroup feeding FOUR
softmax-only warpgroups through shared S/P ping-pong slots per row. One TMA
producer warpgroup. Total768 threads: MMA0..127, softmax128..639, producer640..767.
Shared memory ~214KB: LN64 K/V/bias2 stages, Sfloat and Pbf16 two slots/row.
Registers: MMA224 (inc), SFU56 (DEC from initial80), producer32 (dec), total61440.
Original first compile used inc56 on SFU by mistake; NEVER benchmark that build.
The corrected compile is build_split4_fixed.log (running at this note).
Host launches specialized::attention for hot, the existing standalone v7
attention<true> for SAFE. Same grid/fix-list and global TMA descriptor types.
Correctness/performance NOT YET TESTED. The custom probability shared layout
is K_SW64 64x32; SSA PV/ones MMA reads it. score_ready/prob_ready each count4
(one arrival per warp after syncwarp/fences). Slot reuse waits for prior PV.

## Relevant compiler evidence

NVIDIA confirmed conservative WGMMA loop dependency analysis, with mixed wait1
and outstanding groups sometimes silently serialized. The six-chunk drain works
around the control issue here; remaining warnings are actual register shortage.
https://forums.developer.nvidia.com/t/ptxas-mysterious-warning-for-wgmma-mma-async-instruction-serialization/340610
https://github.com/NVIDIA/cutlass/discussions/1645

Only CUDA12.9 ptxas is installed. All experiments use unique C++ namespaces to
avoid GNU UNIQUE launch-configuration statics colliding between libraries.


## Follow-up measurements and active builds (22:31 KST)

No new winner. Installed main remains ~845us / SM57.70%.

- split4 fixed dealloc build completed without warnings. Job14126 FP64 RMS
  .14979 vs.0002888 baseline: wrong output. First query/first columns look right;
  debugjob14130 uniformV=1 passes but others fail. Job14129 hot2.017ms;
  specialized split not viable, not installed. SASS saved split4.sass.
- tile1n64: single-score serial M64/N64,Rows1,Stages2,producer32threads,
  consumer112/prod24 dynamic regs,target4residentCTA. Job14133 hung at candidate;
  cancelled. Partial warpgroup setmaxnreg is suspect. Do not relaunch this binary.
- tile1n64static: same but no dynamic registers. Used78regs, zero spills.
  Job14141 passes FP64, hot1315us vs847us. Reject: poor bias reuse/traffic.
- tile5n32: serial single score,Rows5,LN128,N32,Stages2,768threads,
  consumer88/prod24. Compiles clean but >228KB shared, launch fails job14136.
- tile5n32x64: reduce LN to64. Compiles clean80regs, initial FP64 passes;
  job14142 hot1026us vs839us. Still slower.
- tile5n32gd: LN128, same Rows5, global .ca bias reads (drops SMEM bias ring
  and TMA bias traffic; source occupancy_global.cu). No warnings/spills;
  job14155 passes FP64, hot1352us vs844us. Slower from exposed global loads.
- tile5n32gp: gd plus one-score QK-next/PV-current pipeline. Top wait1 retires
  QK; SFU overlaps priorPV, wait0 before packedP reuse, next QK thenPV.
  Four-step drain avoids ptxas loop serialization. Compiled clean80regs.
  Benchmark job14158 (verify actual job id from log; submitted just before note).
- direct: original M128/R3 broadcast kernel, hot bias from global .ca cache,
  same TMA QKV. SAFE original. build_direct.py copies prototype with unique
  namespace/wrapper. C7512 + heavy spill (counter-based global addresses).
  Job14151 bitwise equal but hot2091us vs838us. Reject.
- directv: asm volatile/memory barrier on global loads; identical C7512/spill;
  no GPU test warranted. Build CPU session84721 may still be running SAFE.
- directj: replace direct_chunk++ with explicit per-period tile offset and
  compile-time c/h; reduces hot spill to164store/156load bytes, still C7512.
  Build CPU session48050 may still running. No GPU test yet.
- tile3n128 and tile2n128: ACTIVE CPU BUILDS (sessions58615/27730), use serial
  one-score M64/N128/LN128/Stages2,consumer160 or224. Larger MMA and64 exp/thread
  improve SFU ILP and reduce per-logit barrier/instruction count. Not measured.

bench.py accepts direct* modules at build/triattn_sol_<variant>/*.so and loads
matching Python wrapper with _EXT injected. All original standalone modes use
build_<variant>/*.so. No new library is loaded into serving dispatch.


## Latest state (22:45 KST; continued work)

All measured candidates still slower than installed. Further results:
- tile5n32gp job14158 hot1366us vs840: global bias remains slow, reject.
- directj completed: much less spill164/156 but still C7512; not benchmarked.
- tile3n128 job14161 hot1144us vs844, FP64 passes (~.00028527); job14164 NCU
  SM43.11%, XU43.11%, tensor23.40%, issue29.99%, L2~6.93GB/61.3%.
  MIO throttle1.71, long-scoreboard2.21, wait2.20. Larger MMA alone not enough.
- tile2n128 job14162 hot1353us vs847, same initial FP64 pass. Reject.
- tile3n128ring job14169 hot1179us vs849. Named E-token ring still slower.
- tile5n32b1 job14165 hot1120us vs843, FP64 passes. One shared32KB bias tile
  separate from two K/V slots, bias released after last QK reads. Too exposed.
- tile5n32b4 job14166 hot964us vs839 (FP64 passes). Source occupancy_b4.cu:
  separate TMA producer warps (leader Consumers loadsQKV, Consumers+32 bias),
  LN128KV2slots, bias4 independent8KB slots, released after each QK issue.
  NCU job14167 hot971.104us SM51.12%, tensor27.83%, issue48.43%, occupancy33.50%,
  L2 189278735 sectors (~6.06GB), L2 63.38%. Still slower than installed.
- tile5n32bp job14168 (b4+single-score pipeline) C7514/17 forced serialization,
  slower ~1.1ms. No need more tests.
- tile5n32b2 b4+TWO score pipeline, consumer96/prod24. Clean compile, but job14170
  hung at candidate. CANCELLED. Reason: dynamic register budget exceeds initial
  CTA pool. Used80*768=61440; requested5*96*128+24*128=64512. PhysicalSM capacity
  does not enlarge CTA pool. DO NOT launch tile5n32b2 or tile5n32bc (same96budget).
  Primary NVIDIA confirmation:
  https://forums.developer.nvidia.com/t/sm90-setmaxnreg-will-change-occupancy-dynamically/302068
  PTX requires allwarps within WG execute same setmaxnreg. Partialproducer WG
  in earlier tile1n64 should remain rejected. Valid5consumer budget is88/prod24.
- tile5n32b2l and tile5n32bcl compiled atconsumer88; both C7512. No GPU tests yet.
  bcl = two-CTA query-tile cluster (adjacentqt have sameKV), K/V multicast.
  rank0 loads allK, rank1 allV, bothMCmask3. TK built withMC op but tilecount1
  to retain fullbox; eachCTA expectsallKVbytes. Empty barriers countRows*4*2;
  eachwarp arriveslocally and remotely. Bias remainsperCTA, noMC. Cluster enter
  andexit holdSMEM. SAFE isunclustered andusesMCmask1. NOT CORRECTNESS TESTED.
- tile5n32bu / tile5n32bcu builds active (CPU54467/91572): b2l/bcl but WGindex
  madeuniform with __reduce_max_sync, SSoperanddescriptor slices useget_slice(0),
  outputcoordinate slice remainsget_slice(t). Hope toreduce actualregister need.
- tile4n32bu / tile4n32bcu builds active (CPU session ids in tool history): same
  uniform/cluster designs withRows4,consumer112/prod32,Threads640,initial96 gives
  61440regs pool =512*112+128*32, valid. More registerroom withoutserialization.

Important: for real SM90 (~540us atsameexp count), M64 doubledKV L2 traffic is
close to a hard bandwidthlimit. K/V multicast toquery-pairedCTA cuts that traffic
inhalf; this differs from the older failed bias-multicast prototype.

No new serving files changed, no new winner installed, no finalclaim ofSOL90.


## Toolchain comparison active (22:52 KST)

- tile5n32bu/bcu still C7512 atconsumer88; no GPU tests. Uniform descriptors
  do not eliminate the register bottleneck.
- tile4n32bu /bcu clean96initial regs, no spills. Both pass FP64;
  jobs14171/14172 hot1173us/1217us vs~840. K/V multicast correctness passed
  initial check, but its extra cluster/barrier overhead outweighs savings.
- tile5n32bs /bcs use FP32 scalar denominator (unroundedP) instead ofonesMMA,
  trying tosave registers; still C7512. Jobs14173/14174 initialFP64passes at
  .0002937 (baseline.0002888), hot1102/1157us. Rejected.

Downloaded official NVIDIA `nvidia-cuda-nvcc==13.4.92` wheel (~47.7MB) fromPyPI,
verified published SHA256 a1f3bfb27299e060b444d5df1f4bcd762501326cf0fd3141ed61756816ab9a0e.
Extracted only toolchain/ptxas-13.4.92. Versionconfirmed13.4.92 (2026-09-01 build).
The existing Python hasNO pip, so downloadusedstdliburllib afterautoapproval.
Source: https://pypi.org/project/nvidia-cuda-nvcc/13.4.92/

`toolchain.py::use_ptxas()` makes a local CUDA_HOME shim (all12.9 tools/headers,
onlyptxas13.4) atcore_sol90/toolchain/cuda_home_134. Doesnotchange globaltoolkit.
`build.py VARIANT_c134` reads originalvarianttemplate but usesunique fullartifact
name andnewptxas. `build_base.py base134` copies the installedbroadcastprototype
source withunique namespace/name andbuilds usingnewptxas. bench.py acceptsbase*
like direct* wrappers. Bothcurrentlybuilding:
- base134 CPU84104, logbuild_base134.log
- tile5n32bu_c134 CPU68465, logbuild_tile5n32bu_c134.log
Afterbuild, inspectC751/spills andactualshimtoolpath; runbenchforboth. Themain
hypothesis iswhethernewassembler avoidsforcedserialization and/or improves
existingkernel scheduling. No servingchanges. SOL90 goalstillactive/unfulfilled.


## 23:06 KST: exp2 pipe balancing

CUDA13.4 ptxas builds finished but both runtime jobs fail with `device kernel
image is invalid`: 14175 tile5n32bu_c134, 14176 base134 (even stage_bias).
No timings and no serving changes. GPU driver compatibility not yet established.

New basepoly41/basepoly42/basepoly41n builds use exact broadcast source and
unique namespaces, replacing 25% or50% of hot exp2 with range-reduced degree4
polynomial and integer exponent reconstruction. SAFE remains nativeex2. Degree4
CPU maxrelativeerror on[-.5,.5] is7.3e-6. basepoly41n removes redundant clamp
from polynomial argument while keeping final underflow/overflow selects.
All have initialRMS/FP64 andrealnamespace gates in bench.py. Not measuredyet.
FMA/SFU balancing is also used in official FA4 (SM100 packedf32x2 there; this
experiment uses scalarFP32 onHopper):
https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/softmax.py


## 23:14 KST: new measurements and swizzle correction

- basepoly41/42/41n all passFP64 (~.0002888), nohotspills. Jobs14177/78/79:
  hot916/1095/905us versus839/843/846. FMA/SFU substitution slower; reject.
- basew4 exactbaseline withR4 andtwoScoreWschedule,consumer112: C7512 and
 176store/152loadspillbytes. SAFE520/936. Notbenchmarked.
- warp6 usesTMA producerWG+6consumerWG withwarp-level SM80 BF16mma.sync.
  896threads,Used72,consumer80/prod24 valid64512registerpool, zerospills.
  Initialjob14180wrong; debug14181/82 locatedSWIZZLE UNIT ERROR: directcall
  SQ{}(coord)/SK{}(coord) usesbyte-addressswizzle on elementoffset. GMMAflagged
  layouts requiremake_tensor(pointer,layout) or as_position_independent_swizzle_layout.
  Fixeduniquevariantwarp6f plus1024Bexternalignment. Job14183passesFP64
  .000288830vs.000288835. Hot1723us vs840, so reject onperformance. NCU14184:
  SM68.54% (ISSUE),XU28.43%,tensor20.65%,issue68.54%,occupancy39.66%,
  L2 34.72%. HigherSOL whileTWICE slower; NOT aSOL90 orperformancewin.
- split4 hadSAMEbug inmanualSP{}(q,k) probabilitystore. Fixedinunique split4f
  (as_position_independent_swizzle_layout;extern1024alignment), buildingnow.
  Thisalsochangesbankconflicts, sooldwrongoutput timingcannotrejectcorrected
  structuraldesign. No installedsourcechanged.


## 23:27 KST: single-score128 and reduced-fence baseline

- split4f job14185 nowBITWISEEQUAL toinstalled but1909us vs848; NCU14186
 1.900ms SM34.83%issue,XU26.03%,tensor14.07%,L233.54%,MIOstall2.55. Reject.
- m128_pipe.cu newminimalM128/N32/LN128,8half-bias4KB? ActualN32half8KB,
 8slots64KB,total~225KB shared atRows4. TMAproducerQKV+separatebiasproducerwarp,
 oneScore/Pbuffer,twoOutputhalfaccumulators. ScalarunroundedPdenominator,SAFE
 runningmax, hotfallbackonextremes. m128r4 originalbuild wasinterrupted (handle
 38214 missing), hasWRONGbiasdescriptorhalfstrideM*N; DO NOT launchit.
 Correctstride64*N in m128r4f. m128r4f Used96,consumer112/prod32,clean/nospill.
 job14191 initialFP64 .000293701 vs.000288835 (1.7%higher<5%gate), hot906us vs845.
 NCU14193 909us SM53.86%,issue53.23%,tensor26.68%,L247.08%,wait1.39. Stillreject.
 m128r3 job14192hot1020vs843, sameaccuracy;compilerhot56/80spillbytes.
- m128_two.cu shrinkscomputeN16, LN64, maintainsM128queryreuse and2Scorebuffers.
 QK-next beforeE-current, PV-prev overlapsE, wait1 beforePbufferreuse. Scalar
 denominator andindependent8bias-halfslots. m128n16r5 consumer88,Used80 CLEAN;
 m128n16r6 consumer80,Used72 C7512 -> notbenchmarking6. Jobs14200/14201 expected
 for5consumerbench/profile (verify toolhistory).
- baseprepv/baseprefence areexactbroadcastsource, unchangedR3/M128/N32. Move
 pack/PV(k-1) BEFOREexp(k);preservetwoWGMMAcommitsperbodyandallwaitcounts.
 baseprepv retains2arrives/body; baseprefence packs+operandfencesPbeforeQK
 arrive, thenissuesPVwithNOsecondhardwarefence. SAFEbodykeepsoriginalsequence.
 Bothcompiledcleanhot128regs; SASSbaseprefencehot WARPGROUP.ARRIVE39->23,
 DEPBAR22same,HGMMA116same,EX2 288same,F2FP160same,FFMA296same. NotyetGPUtested.
 Benchjobs14198baseprepv,14199baseprefence. No servingchanges ornewwinner.


## 23:35 KST: latest outcomes and FP24 transport

- baseprepv job14198bitwiseequal,hot864vs842; reject. baseprefence14199bitwise
 equal,hot841.64vs842.15, no realwin. NCU14206 844.512us SM57.60%,issue37.01%,
 tensor32.66%,L258.66%,compute-memory64.86%,L1tex66.06%. Stallsperissue:
 MIO1.856,longscore1.724,wait1.706,shortscore.776,mathpipe.068. Removing16
 hardwarefences/period didnotremove remaining shared-memory-relatedwaiting.
- m128n16r5 cleancompilepassesinitialFP64 .0002950vs.0002888 (2.1%higher),
 hot993vs844 job14200. NCU14201 completed. m128n16r6C7512 notbenchmarked.
- m64r6 derivedfromsingleScoreM128, foldedonehalfM64 andLN64 with6consumerWG,
 scalarDen,8bias8KBslots. C7512 despiteoneScoreandconsumer80; SAFEspills192/80.
 job14213FP64 .000293701passesbuthot1425vs848. Reject. m64r5configexistsunbuilt.
- NEWbase24 usescompress_bias.py::transform. Biasstaging nowalloc7168words/
 original4096nominalcolumn. Physicalper(bh,qt): alloriginalFP32columnsfirst
 (nkc*4096), thenallcompressed24bitcolumns(nkc*3072). Originalsource retained
 forSAFE; HOTloads24bitFP32top24bits, decodesintegerPRMT/shift/mask. TMA bias
 halfbox256x6floats(6144B) vsSAFE256x8(8192B), sourcehalfstride1536/2048,
 qstride7168*nkc. SharedkSlotElems3072hot/4096SAFE. Stagepacking usesquadshuffle,
 packs4valuesinto3uint32and arranges16values/thread into3LDS128vectors.
 CompilerbuildactiveCPU42450,build_base24.log. NoaccuracyorGPUtimingyet.
 No newservingfileschanged, installedbest remains~845usSM57.7%. Goalnotachieved.


## 23:47 KST: diagnostic evidence and Q-register3-score pipeline

- base24 hotcleancompile, CPU400000randombitpatternsdecodeexactlytop24bits.
 Job14223FP64RMS .000289122vs.000288835 passes (0.1%higher), buthot873.52vs843.81;
 stage19.94vs13.87us. NCU14224 hot875.616us,SM55.83%,issue46.95%,L2sectors
 135983465 (~4.35GB) vsbaseline~4.96GB,compute-memory55.84%. Reject.
- m64r6p2(shorter2stepperiod)stillC7512,notbenchmarked. m128n16r6p2 CLEAN
 atconsumer80/initial72, nohot/Safe spills. Job14226passesFP64 .000295025
 buthot1043.67vs842.10. NCU14227 1.049888ms,SM70.44%ISSUE,XU46.46%,tensor22.92%.
 Reject: higherSMmetricbutslower. m128n16r6p4 C7512again,notbenchmarked.
 Thisisolatesunrollregisterpressure; shortperiodcanremoveC7512 butextra waits
 and smallMMA tiles stillpreventspeedwin.
- DIAGNOSTICONLY baseablatenobias andbaseablatereplay. DoNOTinstall/useasvalid
 outputs: roof.py explicitlylabels themincorrectablations. SAFEunchanged, hot
 kNoBias=!kSafe or kReplay=!kSafe. Bothcleancompile. NCU14229NoBiashot807.168us,
 SM60.28%,issue41.15%,tensor34.18%,L2~4.93GB. NCU14230Replay(noTMAafterfirstfill)
 hot843.360us,SM57.31%,issue34.96%,tensor32.49%,L2~1.56GB. ThereforeglobalTMA
 transportisNOTdominant; removingallbiasLDS only~4.5% faster. NeedreduceonSM
 tensoroperandtraffic/scheduling,notglobalbandwidthcompression.
- NEWmake_three.py generatesm128_three.cu(controlSSQ) andm128_three_qr.cu.
 M128/N32/LN128,Rows3,3Scorebuffers,2Pbuffers,2Outputhalves, scalarDen.
 CurrentchunkE->pack->PV, QK(k+2) issuedbeforeE, nextbiasprefetchedafterPV;
 topwait2 retiresQK(k),PV(k-2), leavesQK(k+1),PV(k-1). Sixchunkperiod
 matches3Score/2P ring; drainatloopbackedge. Prologue establishes4groups
 PV0,QK2,PV1,QK3 andprefetchesS4; tailguardsconsumeallrealchunks.
 QregistervariantloadsQonce viaCuTe LDSM (two64x32RS-Afragments,16registers
 total) andusesRSQK, aimingtosave4KBsharedreadper64x32chunk. FewerScore
 bufferspayforQregisters toavoid theolderfourScore+Qregs regoverflow.
 Buildsm128threeCPU60792,m128threeqrCPU61436 activeatnote. Bothconsumer160
 producer32,512threads. Basememory/QKvariantpendingtest. No servingchanges.


## 23:56 KST: 3-score outcomes and shape specialization

- m128three andm128threeqr compilewithoutC751 butHOT spills. Firstlogentryis
 SAFE(clean),secondisHOT. Control80Bstack116store160load; QR80Bstack204store
 256load. InitialQregisterloadlayoutcorrect: both14241/14242 passFP64 at
 .000293701vs.000288835; hot1141.86/1144.29us vs852.39/837.41. NCUQR14243
 1.148928ms,SM51.22%issue,tensor20.80%,compute-memory83.30% (spillsinflateL1).
 JobswerebrieflypendingbehindOTHERS' b7-glu jobs14236/14238; queuecleared
 naturally, nootherjobsmodified.
- m128threeflat/qrflat precomputefullK/V ringdescriptor tensors outside
 loop,insteadof constructing perchunk. Compileralreadydoes similarwork: same
 controlspill116/160;QRslightlyworse208/260. NoGPUtests warranted.
- CheckedCUDAdebugasserts hypothesis: genericbuilder lacks-DNDEBUG butSASS
 hasNOassertcalls/traps, so no evidenceofdebugassertoverhead. No suchclaim.
- SASSserving usesmaxR143 andNOlocalmemory; m128threeqr maxR157,64LDL/51STL
 instructions. Dynamicreconfiguration presentasUSETMAXREG.TRY_ALLOC.CTAPOOL0xa0.
- NEWm128three768 andm128threeqr768 CPUbuilds89423/78545. Samealgorithms but
 replacep.L with768 andp.scale withfloatconstant0x1.6a09e6p-3f; hostexplicitly
 checksL==768 andfloat(scale)==constant. Removingruntime shape/indexstate
 mayreduce registerpressure. NotyetGPUtested.
 Serving remainsuntouched~845us/SM57.7%. SOL90notachieved.

## 2026-09-22 00:07 KST: specialization results, fused N40 in progress

- `m128three768` made hot spills worse (180 store / 208 load bytes).
  `m128threeqr768` removed all hot spills, but job 14253 still measured
  990.34 us versus serving 845.26 us. Initial FP64 RMS passed at .000293701
  versus .000288835. Reject; no serving changes.
- `baseconst768` keeps the serving pipeline and specializes S=N=768, H=4,
  scale=1/sqrt(32) in the hot pass. SAFE retains runtime parameters. Hot
  compiles clean. Job 14259 was bitwise equal, hot 834.72 versus 840.25 us.
  Graph timings varied substantially across rounds; this is not an established
  end to end win. Nsight job 14262 completed for a separate measurement.
- `baseqr768` adds the existing Q register mode to that specialization.
  Hot has C7512 serialization despite no spills; not benchmarked.
- `fuse_pv40.py` / `basef40` is a new unqualified experiment: fuse PV and the
  ones denominator into one N40 WGMMA instead of N32 plus N8. V is packed in
  groups of eight keys with an appended eight ones columns. A TMA load uses
  a dense (64,5,16) box; GMMA reads the equivalent MN interleaved layout.
  The last four accumulator registers alias the existing denominator view.
  The timed Python wrapper includes packing (ones allocation plus copy).
  Uniform masked row indexing is also adjusted to the packed V storage.
  First compile failed on an unavailable CuTe `_320` alias; corrected to
  `Int<320>`. Also corrected a transformation regex to preserve all commits.
  CPU build session 47847 is active. No GPU launch or correctness claim yet.

Installed best is unchanged, approximately 845 us and SM SOL 57.70%.
SOL90 remains an active, unachieved goal.

## 00:14 KST: fused PV result and next two comparisons

- NCU 14262 measured `baseconst768` at 837.280 us, SM 57.876%, tensor 32.986%,
  occupancy 21.234%, L1tex 67.057%. The SM throughput maximum comes from
  `sm__inst_executed_pipe_xu`, not tensor activity or generic issue throughput.
- `basef40` compiles clean in hot; SAFE has 156/168 byte spills. Job 14266 is
  bitwise equal, but hot 833.259 versus 842.304 us is only a small gain.
  Packing adds 58.34 us fill plus 194.85 us copy, so total core and block are
  slower (1160 versus 950 us core entry). No installation. The N40 fusion
  and packed V layout are now initially validated; not fully qualified.
- `fuse_m64.py`: apply N40 to the smaller M64, two-score, one-P pipeline,
  four consumers at112 regs / LN128 or five at88 regs / LN64, both two KV
  stages. Full square L768 and scale are guarded and compiled constant.
  V packing uses a single CUDA uint4 kernel, included in the core entry.
  First compile found a shared `auto` declaration for different K/V TMA
  slice types; split the declarations and restarted both builds.
- `baserneint` and `baseroundint`: compare exact ties-even BF16 rounding
  through integer add/shift/PRMT versus nearest ties-up integer rounding.
  Both clamp positive NaN payloads before adding so NaN cannot become -0.
  SAFE retains the original conversion. These test moving BF16 packing off
  the XU pipe that also executes exp2. Builds active; no GPU results yet.
  Earlier truncating PRMT flag was numerically worse; neither new variant
  truncates. All remain isolated from serving.

## 00:21 KST: integer packing rejected; early fence and wider keys

- M64 N40 candidates compile clean with no hot/SAFE spills or C751 warnings.
  Initial RMS passes at .00028883038. Job 14268 R4 hot 1082.02 versus845.40 us;
  job 14269 R5 hot 1411.48 versus839.55 us. The CUDA V pack adds118.2 us.
  Both rejected. Increasing active consumers alone did not improve latency.
- CPU conversion check covers228480 positive finite bit patterns including
  half-way rounding boundaries: integer RNE matches Torch BF16 exactly;
  ties-up differs only on ties. Low-payload NaNs can round to Inf, so an
  initial assertion that every NaN stays NaN was too strong and failed.
  Corrected check confirms every NaN remains nonfinite, which preserves
  SAFE detection. No claim of an exact NaN payload conversion is made.
- Jobs14270/14271: both integer packing candidates were bitwise equal on the
  benchmark, but hot1035.04/896.75 versus840.70/840.05 us. Reject both.
  Added integer issue/dependency cost exceeds the saved XU work.
- `baseearlyfence`: differs from earlier `baseprefence`. P(previous) is packed
  before QK's hardware fence, but PV(previous) still commits LAST after
  E(current). The PV helper omits the second hardware fence, whose wait
  otherwise covers E's unrelated in-flight register writes. SAFE unchanged.
  Clean hot build, no spills; GPU job14274 submitted.
- `make_wide.py` / `m128wide2`: M128/N64/LN128, two full consumer WGs at224
  registers, producer32, three score buffers, two P buffers. Restore ones
  WGMMA denominator, and compile L768/default scale constant behind a guard.
  Eight bias half slots total128KB; K/V two-stage ring. CPU42636 active.
  Check ptxas initial register pool before launch; no GPU result yet.

## 00:32 KST: wider keys rejected and instruction attribution corrected

- `baseearlyfence` job14274 bitwise equal, hot835.22 versus840.59 us.
  NCU14276:839.648us,SM58.225%,tensor33.014%. SASS WARPGROUP.ARRIVE39->23,
  DEPBAR22, HGMMA116, EX2288, F2FP160 unchanged. Gain is small, no installation.
- `m128wide2` (consumer224) and `m128wide240` (240/producer24) both retain
  C7512. Initial168*384=64512 pool safely covers both requests (240/24 exactly).
  240 also introduces8-byte hot spills. Neither was sent to GPU.
- `m128widefull` unrolls the fixed L768 main loop completely, removing C7512
  and all hot/SAFE spills at224/32. Job14281 passes initial FP64 RMS .000287134
  versus .000288835, but hot1129.82 versus850.31us. Reject. C7512 can disappear
  after loop unrolling; do not equate the message with a hardware register
  impossibility without examining compiler liveness.
- NCU14277 `baseroundint`:891.552us,SM54.069%,tensor30.658%. Absolute XU work
  estimated from duration barely changed. This motivated the instruction
  probe rather than another assumption about instruction routing.
- `xu_probe.cu` is a synthetic diagnostic, NOT an attention result. Seven
  kernels each execute4096 operations/thread,132 CTAs x256 threads. Job14280
  measured XU sum4325376 for exp2 and ZERO for BF16x2 conversion, PRMT, SHR,
  LOP3, ADD and video add/min. This conclusively rejects the packing/XU premise.
  The raw report is xu_probe.ncu-rep; reduced results are in xu_probe.csv.
- `basefusedroundint` compiled clean (min after add enables VIADDMNMX), but
  not GPU benchmarked: the XU premise is now disproven. Do not install.
- Next bounded comparison: `basepoly31n` / `basepoly32n`, degree3 polynomial
  exp2 for25%/50% of logits, no redundant input clamp, original exp2 for the
  rest and unchanged SAFE. Earlier degree4 was slow. Degree3 reduces the
  arithmetic latency at an accuracy tradeoff that must pass FP64 checks.
  CPU builds active; not tested or installed. Goal still not achieved.

## 00:44 KST: polynomial screens and scaled Q register experiment

- `basepoly31n` /32n hot clean; cubic maximum relative error1.88081e-4,
  RMS4.83242e-5 on fractional range[-.5,.5]. Jobs14284/14285 pass FP64 at
  .000288540/.000288620 versus .000288835, but hot877.85/1000.80 versus
  846.63/844.46us. Reject. Lower arithmetic latency than degree4 still loses.
- `basepoly21n` /22n use a degree2 least squares polynomial with coefficients
  [.9999482435818677,.7015086752699851,.24229444447823467]. Max fractional
  relative error.00376285, RMS.00110576. Both compiled clean in hot. GPU
  jobs14290/14291 are initial accuracy and performance screens, not winners.
- `scaled_q_model.py` job14288 is a numerical model ONLY, not a CUDA timing.
  On the actual prologue inputs, constant shift64 without per-row seeding has
  RMS1.00307x baseline. Pre-scale Q by scale*log2e, round Q/K to FP16, and use
  fixed shift64: RMS1.00998x baseline (.000291718 versus .000288835). The
  result stays within the initial1.05x RMS criterion on this sample.
- New `scaled_q_regs.py` / `basescaledqr`: keep the existing M128/R3 pipeline.
  Hot QK uses FP16 RS operands: Q is LDSM-loaded once and converted/scaled in
  registers; each K stage is converted in-place by its owning consumer WG
  after TMA full, followed by proxy fence and a128-thread named barrier.
  V and P remain BF16. The original BF16 arrays are unchanged in global memory.
  An additional small staged-bias transform folds scale*log2e and shift-64
  into hot bias. Hot E becomes exp2(score), eliminating per-logit FMA and
  Q shared-operand rereads. Fixed-shift hot skips periodic rescale and flags
  denominators outside[2^-84,2^-44] for SAFE. SAFE uses original BF16 Q/K and
  original bias, via its original Args, and is otherwise unchanged.
  CPU build56583 is active; NO GPU correctness/performance evidence yet.
  This changes hot QK precision. It is unqualified, and extreme/cancellation
  cases plus synchronization must be examined before any serving use.

No serving files have changed in this follow-up; SOL90 remains unachieved.

## 00:49 KST: quadratic rejected; scaled Q compilation variants

- Jobs14290/14291 degree2 exp2 approximations pass the initial RMS ceiling
  (.000294055/.000298160), but hot869.32/970.47 versus842.35/838.11us. Reject.
- `basescaledqr` completed with C7512 in hot despite no spills; initial128
  regs and160 consumer budget. SAFE is the original code (184/232 byte
  spill stores/loads). GPU14292 submitted for the first actual correctness
  check of Q/K conversions; verify the submitted job ID in the matching log.
- `basescaledqrloop` reduces only the K conversion loop to unroll1, aiming
  to avoid hoisting16 packed loads/conversions into live registers at a
  boundary with outstanding MMAs. CPU4777 active.
- `basescaledqr768` forces hot to process all six key tiles, including masked
  prefixes/suffixes (masks stay folded into staged bias; all-empty rows still
  skip). S==768 is guarded. Hot n_w=6 and a three-iteration unroll directive
  remove variable stream-length control state. SAFE retains its own masks
  and variable ranges. CPU34509 active. Neither variant has GPU evidence yet.
- The integer-packing variants have no remaining work justified by the XU
  premise. Their routing diagnosis is settled by xu_probe job14280.

## Scaled Q follow-up (after job14292)

- Job14292 validates actual Q-register and K-shared conversion layout. RMS
  .000291694 versus .000288835 agrees with the numerical model; hot971.88
  versus839.21us is slower. Extra hot-bias transform7.42us, SAFE3.02us.
  This candidate is rejected for speed and has not had extreme-case qualification.
- `basescaledqrloop` still C7512, no spills. `basescaledqr768` produces C7514,
  no spills. Both use16 reported named barriers due dynamic per-WG barrier IDs.
  Neither is sent to GPU: compilation did not resolve the relevant issue.
- New `basescaledqglobal` removes K conversion and named barriers from the
  attention CTA. In fwd it converts K once to FP16 with ATen CUDA; hot Args
  point to that tensor, SAFE Args retain original BF16 K. Q remains scaled
  in registers. Both K conversion and hot-bias transform are timed. This
  trades a global K pass for less CTA work and simpler async register state.
  CPU build76928 active. Not tested or installed; SOL90 still unachieved.

## 01:02 KST: combined exact candidate and three-score compilation

- `basescaledqglobal` job14293 passes the same initial RMS .000291694 but
  hot915.43 versus843.64us; timed K conversion makes core1142.70 versus919.21.
  Rejected for speed. Removing shared K conversion does not fix four-score C7512.
- `basefast768` combines shape constants and the early P hardware fence,
  preserving PV as the last commit. Job14294 is bitwise equal; hot825.16
  versus839.63us. Graph core946.47 versus904.09us conflicts with hot timings.
  Job14295 uses alternating paired graph measurements to resolve that conflict.
  Not qualified or installed.
- `m128scaled3` fully unrolls the three-score/RS hot stream and removes
  intermediate drains. C751 warnings disappear, but HOT spills132/300 bytes,
  stack64. SAFE is clean. No GPU run. `m128scaled3drain` restores period
  drains to shorten live ranges while retaining full unrolling.
- New `m64pref5` moves bias reads one body ahead in the two-score M64/R5
  stream and fully unrolls L768. `m64scaled5` additionally uses pre-scaled
  FP16 Q registers/global K conversion and exp2(score). Both are isolated
  prototypes. Compile inspection comes before GPU execution.

## 01:17 KST: repeatable small gain, rejected exp table, SAFE barrier defect

- `basefast768` paired jobs14295/14298: starting core901.88 ->888.18us,
  paired ratio.98451; ending903.99 ->889.49us, ratio.98409. Block paired
  ratios.99039/.98997. Full-shape B2/N768/H4 job14296 passes20 bitwise cases.
  NCU14297:832.576us, SM58.265%,tensor33.208%,occ21.281%. Still FAR fromSOL90.
- `basefastdispatch` keeps flags0 hot unchanged for nonqualifying shapes and
  adds flag1073741824 for N=S768,H4,default scale. SAFE1024 remains shared.
  Full-shape job14303 passes20 bitwise cases. Not installed.
- `m64pref5`, `m64scaled5`, and `m128scaled3drain` compile without C751 but
  spill heavily in hot (344/516,272/456,156/388 bytes store/load). Register
  usage level0 variants still spill (332/492 and144/368); no GPU launches.
- `basetable1`/2 replace25%/50% of exp2 with a1024-entry shared lookup at
  rounded1/1024 log2 input. Both hot compile clean, initial RMS .000288494/
  .000289043 passes; jobs14300/14301 hot955.83/1158.78 versus846.65/840.22us.
  Reject: extra lookup/decode cost exceeds the SFU work saved.
- IMPORTANT correctness discovery: job14299 full-shape sanitizer passes
  racecheck (all5 cases,0 hazards), but synccheck fails in SAFE's persistent
  CTA reset on late_seed; memcheck did not run. Job14302 reproduces the
  same SAFE reset failure on the INSTALLED ORIGINAL broadcast binary with
  full-shape forced_safe,3520 errors. Earlier smallN sanitizer cases never
  exercised repeated SAFE list entries per CTA.
  Source has two issues: kBarReinit=FirstUserBarrier+4=12 is passed to the
  user NamedBarrier API, which adds another8 (effectiveID20 outside0..15).
  Its aligned bar.sync is also used from distinct producer/consumer branches.
  `basefastdispatchsyn` uses user-relativeID4 (effective12) and
  NamedBarrier(...).arrive_and_wait_unaligned() at both reset rendezvous.
  Build42205 active. Must pass full-shape synchronization before installing.
  PTX reference: https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier
- `basescaledproducer` is a NEW isolated prototype: the two previously idle
  producer warps convert shared Q (scaled) and K toFP16 after TMA completion,
  then signal q_converted/k_converted barriers. Consumers use SS QK, no extra
  Q registers, no global Q/K conversion passes. Warp0 retains bias TMA,
  warp1 retains original Q/K/V TMA. Each converter warp fences its own writes
  and arrives once; two arrivals make each converted buffer visible.
  SAFE retains originalBF16 arrays andbias, with the reset barrier correction.
  It inherits fixedshift64 and exp2(score) from scaled_q_regs. Build74297
  active; numerical/performance/synchronization qualification not done.

## 01:37 KST: installed exact improvement; current SOL58.45

- `basefastdispatchsyn` corrects both the reset ID and aligned-barrier use.
  Job14304 fullshape forced_SAFE synccheck0 errors. Jobs14305/14306 pass20
  fullshape and40 generic-layout bitwise cases. Fullshape sanitizer14308
  passes all5 cases in racecheck(0 hazards),synccheck(0 errors),memcheck(0 errors).
- `stage_fast_package.py` builds the final ta_core_broadcast namespace/module
  in fast_package/. Jobs14309/14310 pass20 fullshape bitwise cases and2
  nondefault-scale cases. `install_fast.py` installed it after qualification.
  New .soSHA f44286f39cac8a061c72157b6a6df154cbb82249bc788adea164481d27d489ab.
  The old package is preserved in before_fast_install/. Canonical rebuild
  source is now the installed package, NOT the older core_tiles prototype.
  Incremental source patch: ../codex_core_sol90.patch. Report: ../CORE_SOL90_REPORT.md.
- Installed job14312 confirms default/opt-out equality on5 masks x2directions
  x2lengths, no flash fallback, hot1073741824 at768 and hot0 at1024. All17
  shipped vectors pass. NCU14313:830.176us,SM58.447%,tensor33.312%,occ21.270%,
  DRAM607.746MB,L24.953GB. **SOL90 remains unachieved.**
- `basescaledproducer` job14307 passes initialRMS .000291694 buthot1592.67
  versus846.17us. Shared conversion by the idle producer warps is much slower.
  Only8 bytes hot spills, SASS shows one pointer spill/reload around the
  producer's conversion wait; not a consumer bulk spill. Rejected for speed.
- `basescaledssglobal` job14311 pre-converts Q/K globally to isolate SS hot
  arithmetic from shared conversion and Q-register pressure. Hotclean, RMS
  .000291694; hot812.77 versus844.88us, but fully timed core1622 versus932us.
  Only~32us hot gain would be available to pay for a fused prologue conversion.
  Not a serving candidate and no prologue integration was made.
- `m64twocta` targets2CTAs/SM,2consumers104regs+producer32,384threadsinitial80.
  Safe pool30720 registers/CTA exactly covers the requested dynamic pool.
  ThreeScore/twoP schedule hashot808/808byte spills; noGPU run.
  `m64twoctass` removes hot seed/FMA via pre-scaledFP16 SS but stillspills
  904/840bytes; noGPU run. Both remain rejected compile screens.
- New `three_probability.py`:3Score+3P supports QK2ahead and delayedPV(k-1).
  P(k) packs into the slot whosePV(k-3) retired, while P(k-1) was fenced before
  QK's hardware fence. This saves8 register slots relative to4Score+2P.
  `m128p3qr`/`m128p3ss` (6chunk period) have noC751 but spill128/448 and80/64.
  NoGPU runs. The6chunk period does not align the8chunk bias/KV tile, so its
  repeated modulo/address state remains live.
- `m128p3qr24`/`m128p3ss24` instead use a24chunk period, aligning3Score/P,
  2query halves,8chunkTMA tiles. The hot bound is runtimep.L (hostguard768)
  to retain a two-iteration loop. Start withQK0/1ready, P2zero forone harmless
  dummyPV, seed onlychunks0/1, thenprocess48chunks andflushPV47. Builds23929/
  66300 active. These prototypes are not tested or installed; inspectspills
  and compilerwarnings beforeGPU work.

- 24chunk variants both compile clean inhot/SAFE with0spills andnoC751.
  Jobs14314/14315 pass initialFP64 (.000288830), butQR hot1047.21 versus828.32,
  SS hot925.47 versus826.68us. No installation. Their core deltaRMS7.84e-6
  is nonzero; they are NOT bitwise candidates. ActualRS QK is slower even
  after removing its compiler serialization/spills. Do not assumeQregisters
  will improve this D32 problem merely from fewer shared reads.
  NCU14316 profilesSS24 to distinguish issue/control cost from instruction
  cache or shared-pipe stalls before changing another period.

## 01:55 KST: narrowing synchronization and SFU limits

- NCU14316 p3ss24:924.288us,SM52.589%,tensor30.314%,issue40.560% versus
  installed830.176us,SM58.447%,tensor33.312%,issue37.880%. Absolute issued work
  is about19% higher. no_instruction stall .0375 versus .0675: instruction
  cache is NOT the reason this 24chunk loop is slower. Do not shorten its
  period merely to address a presumed instruction-cache limit.
- `independent_rows.py` pairs each two bias halves into one full barrier
  (init2, waitonlyeven), retains independenthalfempty barriers, and makes
  K/V full+empty barriers perconsumer row. `m128p3ss24row` compiles without
  C751;8Bpointerspill. Job14317 passes initialFP64 .000288830 (same asSS24).
  Hot874.681 versusinstalled827.642us:~5.5% better thanpriorSS24, stillslower
  thaninstalled. Core1000.50 versus906.05; not installed.
- `build_installed_variant.py` uses currentinstalled csrc and qualified
  basefastdispatchsyn Python builder under unique namespaces. New
  `base3producer` gives K toproducerwarp1 andV toproducerwarp2 onlyinfast768;
  Qremainswarp1 andbiaswarp0. ExistingperrowK/Vemptybarriers andfullcount2
  retainownership; SAFE/otherhotpaths keeporiginalproducer. Build50603active.
- `baseablateexp` is DIAGNOSTIC ONLY: substitutes a cheap input-dependent
  positive finite bit transform for main exponentials. It is intentionally
  incorrect and may onlybeused throughroof.py (which labelsablation), never
  installed or presented as a valid speed result. Build19411active. This
  isolates howmuch latency can actually be removed by improving exp2.

## 02:07 KST: a second exact incremental winner under qualification

- `base3producer` compiles withhot0spills/noC751. Job14318 initialbitwise
  equality, hot814.513vs831.979us.16-round paired14321 startingcore897.61->
  884.97us,ratio.985886;blockratio.989482.14323ending897.99->884.97us,
  ratio.985582;block.990628. NCU14322hot821.280us,SM59.196%,tensor33.739%,
  occupancy22.324%. No installation yet; SOL90remainsunachieved.
- `base3producerids` additionally changesconsumer kRingBar0 from8 to0:
  the CUTLASS uint userAPI adds8. Its existingIDs8+3 thereforebecame19,
  outside documented0..15. Use userIDs0..3 (hardware8..11); SAFEresetID4
  (hardware12) alreadycorrect. It compilesclean,hot12barriers,SAFE13.
  Job14324pairedstarting,14325fullB2accuracy,14326fullshape3sanitizers
  are active. Finalnamespaced producer_package build also active.
- `baseablateexp` NCU14319hot804.928us,SM49.077%,SAFE3.968us(nofullSAFE).
  It emittedP~1 andtriggered unnecessaryperiodrescaling; do not use it as
  a clean exp2 latencybound. `baseablateexp64` keeps P~2^-64 via0x1f000000
  floatbits, cleancompile,job14327profilesit. Both are intentionally
  incorrectdiagnostics,notattentionimplementations.
- `m128p3ss24split` separatesK/Vproducers andreturnsK afterlastQKretired,
  startingfromperrow/pairedbias. Compileradds23C7519hardwarefences,noother
  C751orhotspills.14320passesFP64but880.259usversus834.374: noimprovement
  overrow874.68. Notinstalled.
- `m64_probability.py` reducesM128toM64,singleoutput+denominator,onlyseed
  seq0,properM64biascoordinates. `m64p3ss24r4` targets4consumer112regs plus
  producer32 (initial640threads*96=61440,exactpool). It failscompilescreen:
  C7512,hot174/328Bspills. NoGPUrun.
- `four_query_halves.py` is a NEW alternative: M256,R3,N32,LN128,2Score,
  2P,4Output+Den,32chunkperiod. AtstepkinitS(k+1),wait1retiresQK(k)and
  PV(k-2),issueQK(k+1),E(k),packP(k),PV(k). Fourqueryquarteraccumulators
  amortizeQ/K/Vtransport;~212KBsharedfitsoneCTA. Build56467active for
  m256p2ss32r3. Notvalidated; inspectspillsandC751beforeanyGPUrun.

## 02:20 KST: second exact improvement installed; SOL59.22

- `base3producerids` paired14324starting897.223->884.408us,ratio.986030;
  paired14330ending904.071->884.934us,ratio.984950. These are paired16rounds
  withbitwisecore/block. Blockratios.990105/.988724. IDs correction keepsgain.
- Fullshapeprototype14325passes20cases; sanitizer14326passes5patternseach:
  racecheck0hazards,synccheck0errors,memcheck0errors. Finalnamespacepackage
  jobs14329/14331/14332pass20full+40generic+2scale(.25)bitwisecases.
- `install_producer.py` checkedproofs+manifest,backedupoldpackagein
  before_producer_install/,andinstalledproducer_package. NewSHA
  893a0a0a32ef03572936fcb7378abca1358a5af13cb075f85aa36a2806c641fa.
  GenericM1SHAunchanged3c766e0296351d643329c7203f94f25ad3c584c5d400b1ca92b74d35fdb225ba.
- Installedverification14333passesdefault/optout5masks*2directions*2lengths,
  exactcore/block,expectedhot1073741824at768/0at1024,noflashfallback,and17/17
  shippedtests. `serving-producer-*.json` preservespreviousserving-fastfiles.
  NCU14334currentinstalled815.648us,SM59.219836%,tensor33.752238%,occ22.299383%,
  DRAM607.608320MB,L2154855020sectors=4.95536064GB. NOTSOL90.
  Currentreport../CORE_SOL90_REPORT.md,incrementalpatch../codex_core_sol90.patch.
- `profile_installed.sbatch` defaultprefixnowinstalled-producer; override
  PROFILE_NAME ifneeded. verify_fast.sbatch defaultsserving-producer prefix.
- NCU14328sourcecountersonbase3producerids:301.173Msoftwarecountinstructions,
  FFMA57.213M,MUFU56.918M,F2FP29.491M,UMOV25.485M,HGMMA21.529M,LDS14.752M.
  Longscoreboardsamples23466:22169atBRA(primarilyproducerwaits),notconsumer
  loadproof. MIO16310:16040atMUFU.EX2;wait15639:9936atMUFU.EX2.
  Rawsourcecorrelationsbase3producerids-source-sass.csv;opcodecountsfrom
  ncu_report.correlation_ids() inbase3producerids-opcode-summary.json.
  --pagesource cannotshowexecutedopcodecounts;usePythoninstancedmetricAPI.
- `baseablateexp64` NCU14327:789.856us,SM47.695%(issue),tensor35.201%.
  RemovingMUFUwithAND/ORincreasedissuedwork~20%,soitisonlyaloosebound.
  `baseablateexpxor` insteadreplacesex2withsingleXORofnegativezbitsagainst
  0xdd000000:typicalz~-64becomespositiveP~2^-64. DIAGNOSTICONLY,wrongmath.
  Cleannospillcompile;NCU14335hot753.312us,SM41.974%(issue),tensor36.922%.
  Relativeoriginal830us,thisremoves~77uswithoutservingeligibility; replacing
  exp2alonemustnotbeassumedtoyieldSOL90. Noapproximationwasinstalled.
- M256p2ss32r3andm256p2ss32ipbothC7512,hot8/8Bspills (mostlyproducerpointer).
  inplace_exponentials.pyuses8in-placeFMAthen8EX2asmperrow,butdoesnotremove
  ptxasGMMAresourceconstraint. NoGPUrunsfor either.
- Newbasehalfring is NOT theoldfullEtokenring (+22.7% inoriginalHANDOFF).
  ItenablesvalidhardwareIDs8..10andpassesEtokenafterfirst8exponentialshave
  issued,whilethecurrentWGcomputessecondrow. Ringwaitisunchanged;outer
  ring_passdisabledforthefastpath toavoidduplicatearrivals. Samearithmetic,
  onlyschedulingsynchronizationchanges. Build9417active;unqualified.
- Newbasefixeddesc makesonlymk_K/mk_Vperiodoffsetscompiletimezeroatfast768.
  IMPORTANT: leavesrelease_dep'sruntimezeroanddatahazardintact. Thismay
  increasehoisteddescriptorregisterpressure; inspectC751beforeGPU. Build48962.
- Buildernowpinsoldbase3producer/ablationfamilies tobefore_producer_install
  snapshotandnewhalfring/fixeddesc toproducer_package,sofutureinstallation
  doesnotchangeanexistingexperiment'srebuildbaseline.

## 02:25 KST: two scheduling results and deeper-MMA candidate

- basehalfring andbasefixeddesc bothcompiledcleanhot0spills,noC751.
  Half-ringuses16named-barrierresources(dynamicIDs),fixeddesc12; bothfit1CTA.
- Half-ring14336bitwisebut840.102versus821.843us. Earlytokenreducescostof
  oldfullEtokenring,still2.2%slower thaninstalled. Notinstalled.
- Fixeddesc14337bitwise811.173versus815.247us,only0.5%hotgain,withnoisy
  graphcore958versus891us. Job14338pairedstarting willdecidewhetherany
  repeatablegainexists. Do notclaiman improvementfromthissingleprofile.
- NEWfour_probability.py:m128p4ss24 uses3Score+4P,keepingthearithmetical
  registerfootprint124values(includingoutput/den/seed),sameasinstalled4S+2P.
  QKis3chunksahead;wait4 retiresQK(k)andPV(k-4),E(k)thenpackintoP[k%4],
  reinitializeS[k%3]frombias(k+3),issueQK(k+3),thenprefencedPV(k-1).
  Sixoutstandinggroups,24chunkperiod,initialQK0/1/2drainedandP3zero for
  onedummyPV. PerrowK/Vbarriers,separateproducers,andearlyKreturnfromsplit
  family. Build10406active;unqualified. ItaimstohidemoreWGMMAdependency
  latency withoutaddingregisters,nottoincreaseSMmetricbydoingextrawork.

- Fixeddescpaired14338starting890.723->887.606us,ratio.996330,block.997317.
  Only0.37%gain; notinstalledwhilelargerpipelinechangesarebeingexamined.
- m128p4ss24failscompile:C7512,208/200BspillsplusC7519fences. Thislarger
  outstandingwindowstillcoststhecompilerregisterresources,despitethe same
  nominal124valuecount. _ru0buildactive tocheckminimum-registerallocation.
- NEWm64p4ss24s3 reducesMto64,singleO/Den,3consumer160regs toallowthe deeper
  wait4schedule. CruciallyitusesTHREEK/Vstages:withM64eachstagehas4chunks,
  and2stageswoulddeadlock(QK8neededatstep5butV0notreleaseduntilstep7).
  Threestagesgive~226KBshared,underH100per-blocklimit,andthe24chunkperiod
  aligns3Score,4P,3KVstageswith4chunks/stage. Buildactive,unqualified.
- NEWbasestaticseed startsfastnm=-64 andseeded0xf,removinginitialmaxreduction
  whilekeepingexistingperiodrescalingandSAFEfallback. ThisCHANGESBF16P
  roundingandisnotabitwisecandidate; mustpassFP64accuracycriteria before
  anyperformancecomparisonandfurthernumericalqualificationifcompetitive.
  Build98581active. Noapproximationorstaticseedwasinstalled.

## 02:35 KST: deep pipeline rejected; smaller double-buffer candidate

- m128p4ss24_ru0 unchangedC7512/208/200Bspills;noGPU.
- m64p4ss24s3compiles0spills,noC7512 (C7519insertedfences remain).14339
  passesinitialFP64.000288830 buthot1059.195vs820.747us. Three-stage
  correctnessworks; thedeeperpipelineisNOTfasterandwasnotinstalled.
- basestaticseed14340passesinitialFP64.000289767versus.000288835 (RMS+0.323%),
  hot804.052vs815.096us. Nonbitwise,only1.35%hotgain;notinstalled. Its
  numericalpolicyhasnotbeenqualifiedacrossthefullpatternmatrix. Keep
  installedexact815.648us/SOL59.220ascurrentresult.
- NEWtwo_probability_m64.py:m64p2ss8r4 targets4consumer112regs,producer32,
  M64N32LN128Stages2,2Score+2P+1Output/Den,8chunkperiod. wait1retiresQK(k)
  andPV(k-2),nextQK(k+1)issuesbeforeE(k),thenpack/PV(k),andS(k+2)biasloads
  atbodyend. TwoPslotseliminatetheoldsinglePpipeline'ssecondwait1. Perrow
  K/VbarriersandindependentK/Vproducersremain. Fewerbuffersmayfit4consumers
  withoutspills,unliketherejectedM64P3/P4variants. Buildactive,unqualified.

- m64p2ss8r4compilescleanhotandSAFE:0spills,noC751,initial96regs/640threads.
  Job14341benchmarks4consumerM64(twoPslots,8chunkbias-prefetchperiod).

## 02:48 KST: four-consumer result and N64 compile screens

- m64p2ss8r4job14341passesinitialFP64(.000288830)but910.884vs823.127us.
  Fourconsumerwarpgroupsfitwithoutspills,howeverlargerK/Vtrafficandthe
  shorterconsumerpipelineareoverallslower. Notinstalled.
- NEWone_score_n64.py keepsM128/R3withN64(oneScore,twoP),4biashalfslots
  (2pairedfullbarriers),2KVstages,8chunkperiod. ItpacksE(k),reloadsS(k+1),
  fencesPbeforeQK(k+1)'shardwarefence,thenprefencedPV(k)commitslast.
  LargerNreusesQmoreperGMMAwithoutM64'sextraK/Vtraffic. m128n64s1p2still
  producesC7512,8Bpointerspill;noGPU. Nominalregistercountunderestimates
  ptxas'stemporary/operandconstraintcost.
- m128n64s1p1keepsasingleP,wait0afterE(k)beforereusingPfromPV(k-1).
  NoordinaryPoperandfencebeforethatwait. EcanoverlappriorPV;theexplicit
  retirementand16fewerliveregisters may removeC7512. Buildactive.
- IfanyN64kernelbecomescompetitive,itsbiasTMAcanreadtheEXISTINGM1
  32-columnstagingwithoutanextrarepack:reinterpretglobalas
  [256,8,half2,k32,qbh],strides[1,256,2048,4096,nk32*4096],loadbox
  [256,8,1,2,1]atonehalfandtwoadjacentk32blocks. Destinationpacks4096floats
  inN64accumulatororder. ThisisadesignnoteONLY,notimplementedorvalidated.

- m128n64s1p1removesC7512;only8BpointerspillandC7519fencesremain.
  Job14342performsFP64gateandactualtiming. NoN64candidateinstalled.

- m128n64s1p1job14342passesFP64(.000287134vsbaseline.000288835),buthot
  881.044vs819.799us;core988.897vs903.103us. N64oneP/oneScoreis slower,
  despite slightlylowerreferenceRMS. Rejected;installedproducer815.648us
  andSM59.220remainunchanged.

## 03:02 KST: QK ablation and TMA-only N40 V layout

- baseablateqk diagnostic removes only fast QK GEMMs while retaining WGMMA
  commits and dependent consumption of shared-bias registers before release.
  Deliberately incorrect, never eligible as attention. Hot0spills/noC751.
- NEW f40_tma.py/basef40tma: fuse PV and row sums into N40 without global
  packing. TMA logical axes [D%8,key,D/8,H,N] transpose the original strided
  V into shared [D/8][key][D%8]; fifth 8-column block is immutable ones,
  filled once before the initial async-proxy fence. Each V TMA still moves
  exactly128*32*2 bytes. WGMMA B descriptor uses independent N-group/K-group
  strides, so the ones tail need not be interspersed every8 keys.
  Prototype explicitly rejects B!=1 (five TMA dimensions already used).
  SAFE and uniform-row arithmetic preserved; not yet compiled/validated.
  TMA global strides checked against CUDA Driver API documentation:
  https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html

- QK ablation14343hot757.888us,SM64.051(SFU),tensor19.912%,issue47.116%.
  Removing QK alone onlysaves~58us versusinstalled815.648. This is a
  deliberately incorrect isolation diagnostic, not a candidate result.
- basestream48 keeps the installed4Score/2P ordering across intermediate
  period boundaries. Fast path expands the at-most3 periods statically,
  waits2 at subsequent period starts, and drains only on range completion.
  Removes two intermediate power-of-two rescale checks; final finite/sum
  validation and SAFE recompute remain. Accuracy is not yet qualified.
- m64n16twocta and m64n16twoctaqr shrink priorM64twoCTA candidate fromN32
  toN16 to fit2consumers104regs+producer32 perCTA(launch80regs,384threads),
  targeting2residentCTAs. ThreeScore/twoP schedule; qr additionally retains
  the8-registerBF16Q fragment to avoidextraQ sharedreads fromnarrowerN.
  No FP16conversion or preprocessing; compile screens pending.

## 03:10 KST: measurements of the new families

- basef40tma14344 isbitwiseexact oninitialcore/block checks. TMA-only V
  transpose and immutableoneswork, buthot864.064vs813.041us. NoextraVprep
  exists, yetthissharedlayoutisoverallslower. Notinstalled.
- basef40tmasw32 uses D16groups/32Bswizzle andphysicalD48sharedpadding to
  retainatleast16contiguousBF16NvaluesperGMMAcorematrix. VTMAmovesonlyD32;
  lastD16blockisimmutableones,ofwhichN40reads8. StillB1only. Buildpending.
- basestream4814345bitwiseinitialcheck,0spills/noC751, buthot971.798vs
  825.900us. Removingperioddrainsandfullyexpanding3periods isworse. Rejected.
- m64n16twocta14346 andm64n16twoctaqr14347 bothcompile0spills/noC751,
  initial80regs/384threads,2CTAlaunchbound. BothpassinitialFP64RMS.000289692
  versus.000288835, buthot1073.821vs807.712 and1066.773vs818.612us. Clean
  twoCTAresourcefitdoesnotproduceafasterkernel. Rejected;QRSdoesnotrescueit.

- basef40tmasw32firstcompilehitunsupportedSwizzle<1,3,3>. Correctedto
  Swizzle<1,4,3>: smem_ptr_flag stores byte-domain swizzling, even when
  associatedlinearLayoutusesBF16elements. Rebuilding.
- NEW f40_descriptor.py/basef40desc preserves the ORIGINALSW64V TMA and
  allocation. N40PV's virtual secondN32group points to an8KBsharedones
  tile afterallVstages using the Major-MN descriptor leadingbyteoffset.
  The selected ringstage subtractsitsVoffset fromthatNgroupstride; key
  chunk offsets remain, selecting the correspondingkeys insharedones.
  Only~7.5KBadditionalshared, noexternalVformat/repack/padding. Supports
  originalstrides andBsize. ThisdescriptorconstructionrequiresGPUvalidation.

- basef40tmasw32rebuild and firstbasef40desccompile fail inCuTe's static
  shape-divisibility/pointer-flag checks. NoGPUattempts. N40cannotbe tiled
  directly fromN48/N64nestedN16/N32groups using thatCuTeconstruction.
- basef40descrevised: keep ORIGINALVt layout aswell, construct the original
  N32Bdescriptor, patchleadingoffset, feeditsone-registerfragment to N40PV.
  Bfragmentisstillonesmemdescriptor; noN40virtualtiling needed. Buildpending.
- basephasepv/basephaselds retainallarithmetic andcommits but staggerE's
  placement betweenconsumerWG roles, withoutnewbarriers. WG0runsbaseline
  E-before-PV;otherWGsrunEafterPV (phasepv), orWG1afterPV/WG2afterbiasinit
  (phaselds). Hot-onlybranches; SAFEandgeneric unchanged. Compilepending.

## 03:20 KST: phase staggering rejected; descriptor fusion runs

- basephasepv14348bitwisehot868.088vs815.097us; basephaselds14349bitwise
  hot914.981vs824.172us. Both0spills/noC751. Differentper-WGEplacement
  addsbranch/schedulingcostwithoutanoverallbenefit. Neitherinstalled.
- Revisedbasef40desccompiles0spills/noC751hot. Job14350checks itsoriginal
  VTMAdescriptorwithseparateconstantonesNgroup againsttheactualservingcore.
- basef40tmasw32updatedwithsmem_ptr_flag_bits<16>andN32Bdescriptorcreation
  overvirtualN64(physicalD48stagepitch). ThisavoidsCuTeN40divisibility while
  retainingN16groups. Rebuildpending;itremainsaB1-onlyexperimentalpath.
- NEWbaseablatepvdiagnosticremovesonlyoutputPVwhilekeepingP*ones,soall
  exponentialsandprobabilitypackingremainliveandcommitcountstaysunchanged.
  Intentionallyoutputszeros;notanattentioncandidate. Buildpending.

- basef40desc14350bitwisecore/block,hot804.845vs814.624us. Paired14351
  starting890.845->881.743us,ratio.989769725,block.992762885. Ending14353
  889.118->880.168us,ratio.989836084,block.993033963. ~1.02%repeatablegain.
- NCU14352basef40desc805.728us,SM59.842244%,tensor34.106979%,issue38.218274%,
  occupancy22.339768%. StillfarfromSOL90. FullB2shape14354passes20bitwise
  casescontiguous/stridedwithallmaskpatternsandforcedSAFE.
- Newmerge_qk_pv_group.py(basegroup/basef40group): QK(k+2)stillissuesbefore
  E(k),PV(k-1)afterE, butdeferQKcommituntilPVandcommitbothasONEgroup.
  wait1thenretiresQK(k)/PV(k-3), thesamefrontier asoldtwo-groupwait2.
  Prologueanddrainsunchanged. No arithmeticorbarrierreleaseschanged.
  Distinctfromold scheduleW, whichissuedPVbeforeE. Compilepending.

- Finalnamespacebuildf40_package(basef40desc)started;notinstalled.
- Jobs14355sw32bench,14356fullsanitizers,14357generic40casecheck,
  14358L1024startingpair,14359PVablationNCU. IMPORTANT14358was submitted
  beforepair.sbatchgainedPAIR_SUFFIX; itsoutputwilloverwrite
  basef40desc-paired-0.json. Original768proofsavedas
  basef40desc-paired-L768-0.json. On14358completionrenameitsjson to
  basef40desc-paired-L1024-0.jsonandrestore768jsonfromthatsavedcopy.
  FutureL1024pairedjobsMUSTusePAIR_SUFFIX=-L1024.

- Correction:14358finishedbeforethecopyattempt. ItsL1024jsonwasmovedto
  basef40desc-paired-L1024-0.json. The768startingjsonwas reconstructed
  exactlyfrompair-14351.log'scompletecore/blockarrays(16roundseach)and
  RESULTsummary, withrecovered_frommetadata; restoredtooriginalpathand
  basef40desc-paired-L768-0.json. No measurementdatawasinventedorlost.
- basef40tmasw3214355bitwise,hot841.339vs818.784us. Correctbutslower;
  preferdescriptor-onlyfusion withoriginalSW64V TMA. Rejected.
- Genericbasef40desc14357passes40bitwisecases atL768/1024, layouts/masks.

- L1024starting14358basef40descpairedcore2003.672->1957.146us,
  pairedratio.987269242(~1.27%);block.988505156. Wholemedianratio differs
  frompairedratio becauseclockvariation; reportthepairedmetricconsistently.
  L1024ending14360usesPAIR_SUFFIX=-L1024 andcannotoverwriteL768proofs.
- PV-onlyablation14359754.464us,SM64.173724%,tensor20.615393%,issue38.263837%.
  LikeQK-onlyandEX2-onlyablations,itdoesnotremoveamajorityofthelatency.
  Wrongzerooutputsbydesign;noteligibleforperformanceclaims.

- basef40desc14356fullN768B1stridedsanitizersallpass:racecheck0hazards,
  synccheck0errors,memcheck0errors;5patterns eachincludingforcedSAFE.
- L1024ending14360pairedcore1992.314->1952.518us,ratio.984344496(~1.57%);
  block.992605856. BothL1024directions improve,notjusttheL768specialization.
- install_f40.py preparedbutNOTRUN. Itrequiresfinalnamespace20+40+2case
  proofs,theabove3sanitizers,and<.99corepairedratiosinbothlengths/directions,
  thenverifiesmanifest+current893aSHAandbacksupbeforeinstall. Packagebuild
  andthetwo merged-groupcompileexperimentsremainactive.

## 03:34 KST: final N40 package qualified; merged group timing

- f40_packagefinalnamespace14363/14364/14365passes20full+40generic+2scale
  bitwisecases. Readyforinstallafterfinalcandidatecomparison.
- basegroup14361bitwise,hot819.878vs822.111us(~0.27%),butgraph865.932vs
  891.097us; graphdisagreeswithhotprofile, needs16pairedrounds14366.
- basef40group14362bitwise,hot805.891vs815.076us,essentiallysameashot
  basef40desc804.845. Graph941vs891noisy;16pairedrounds14367willcheck
  whethermergingcommitsaddsanythingbeyonddescriptor-onlyfusion.
- Bothmergedvariantscompile0spills/noC751fast;noinstallationyet.

- Paired14366basegroupcore ratio1.000086/block1.000725: noimprovement.
  Paired14367basef40groupcore.989731761/block.992746186: identicalwithin
  noise tobasef40desc. Merginggroupsaddsnogain;discardbothgroupchanges.
- Qualifieddescriptor-onlyf40_package selectedforinstallation. Installer
  nowatomicallyreplacesfiles(especially.so), preservingtheoldinodeforany
  processalreadyusingit. install_f40.pylaunched;resultpending.

## 03:38 KST: N40 descriptor fusion installed and verified

- install-f40.log: installedSHA6fe43326e87c87bb5512b39b66437493123e6e68cdc6799bdcc7659f6173f694.
  Previous893aartifactandallsourcesinbefore_f40_install;atomicreplacement.
- InstalledNCU14369:811.648us,SM59.943772%,tensor34.164844%,issue38.283036%,
  occupancy22.310606%,DRAM462.517760+145.080064MB,L2154991798sectors.
  ThisistheCURRENTmeasurement, distinctfromprototype805.728us.
- Installed14368:allfourlength/directionJSONsbitwise, expectedhot1073/0,
  noflashfallback,17/17shippedtests. Manifestallhashesrecheckedmatches.
- UpdatedCORE_SOL90_REPORT,HANDOFF,cumulative321linepatchrelativebefore_fast.
  profile_installed/verify_fastdefaultsnowinstalled-f40/serving-f40, preserving
  olderproducerandfastprofiles. SOL90UNACHIEVED,goalactive.
- Allownedbuilds andGPUjobs14343..14369 finished. Unrelatedtraining13228
  leftalone. NoCPUcompilersactiveattheendofthisqualificationbatch.
- build_installed_variant.py experimentsarePINNEDto producer_package(orolder
  before_producer_install), despiteitsname; theydonotautomaticallyusethe
  newlyinstalledN40source. FutureN40-basedworkmustusef40_package/current.

## 03:42 KST: next N40-based register experiment

- NEWbuild_f40_variant.py pinsf40_package(6fe43326). basef40ip4 groups4
  in-placeFFMAthen4EX2; basef40qrip4/qrip8 additionallykeepQinBF16registers
  (existingQRSoption,fast1073only). Intendedtoreducetemporaryregisterpressure
  enoughforQRSwiththeinstalled4Score/2Ppipeline, previouslyC7512.
- fmafFTZsemanticsverifiedbycompilingtinyprobe withsameCUDA--use_fast_math:
  /tmp/triattn-fmaf-probe.ptx contains fma.rn.ftz.f32. InlinePTXusesidentical
  FMAandEX2operations; noreorderedarithmeticwithinavalue,noprecisionchange.
- Job14370(build_f40.sbatch)compiles3candidatessequentiallyonallocated8CPUs
  andscreensresourcesbeforebenchmarking. Only0spill/noC7512/4/5hotvariants
  getinitialFP64/bitwisechecksandtiming. Noadditionalcandidatenowinstalled.
- NewbatchisindependentofcompletedN40installation. Currentdefaultremains
  811.648us/SM59.943772%,17/17tests;SOL90isnotreached.

- Countercross-check(currentvsablations):time*LSUwavefrontpeakfraction is
  ~553.5usinstalledN40,363.1uswithQKremoved,488.6uswithoutputPVremoved,
  551.4uswithEX2removed. L2normalizedworkremains~494–497us inallcases.
  This supportsreducingQK sharedreads(QRS) asadistinctdirection; eliminating
  EX2doesnotreducetheshared-pipework. Theseare diagnosticworkcomparisons,
  notaformalSOLboundorredefinitionofthegoal; clocksdifferacrossjobs.

- basef40ip4job14370:bitwise,818.009vs809.591us;core939.316vs879.572us
  (three-roundgraphnoisy). Noimprovementfromin-placegroupingalone;notinstalled.
- SASSsavedinstalled-f40-hot.sass/basef40ip4-hot.sass. Both296FFMA/292MUFU
  staticinstructions;total3504vs3512. FFMAcontiguousrunschange(max16->10),
  soasmdoesaffectscheduling,butthebaselinealreadyhasmanyin-placeFFMAs.
  cuobjdumpprints3not-foundwarningsforotherfatbins;thesehotdumpsarenonempty
  andcontaintheselected1073function. QRcombinationsstillcompiling.

- basef40qrip4hot0spillsbut16C7519fencesandC7512(insufficientresources,
  WGMMAserialized);screencorrectlyskipsGPUtiming. Stillnotaclean4S/2P QRS
  pipeline. basef40qrip8remainingin14370batch.

- basef40qrip8also0spillsbutC7512,16injectedC7519fences.14370batchendswith
  onlyip4timed(slower);neitherQRSvarianteligible. Nochangesinstalled.
- NEWm256_two_consumers.py:m256ss2pref/m256qr2prefuse2consumers224regs,
  producer32,384threads, M256N32LN128(twoS/twoP,fourO/Den),8biashalfslots.
  PrefetchS(k+2)atbodyend;QRholdsallfour64x32Qhalves(32GPR)inregisters.
  LargerquerytilesreduceCTAswhiletwoWGshavemoreavailableRFspace. L768only.
  SeparateonesMMA/genericpreparerretained,so comparefullcoreoverhead aswell.
  Job14371builds/screensboth;noserializing/spillingkernelwillbebenchmarked.
- FutureIFthisfamilyiscompetitive:4S/2P QK2-aheadpipeline and/orN40descriptor
  fusionmayimproveit. Neitherisimplementedhere;don'tassumeinheritedN40.

- AddedM256initialregisterpoolguardbeforeGPUlaunch(inbench.pyaswellbecause
  14371alreadycapturedolderSBATCHscript). Bothhot+SAFEmustreserve>=160regs
  initially at384threads, covering2*224+producer32=61440physicalregs.
  0spillsalonedoesnotproveadynamicregisterreallocationissafe. Thisavoids
  repeatingtheearlierpartialproducer/underreserved-pooldeadlocks.

## 03:57 KST: larger two-consumer tiles rejected

- Job14371m256ss2prefandm256qr2prefbothcompile0spills/noC751,initial168regs
  inhot+SAFE, satisfyingthe61440-registerpoolguard. BothpassFP64RMS
  .000288830vsbaseline.000288835. SS951.347vs815.368us;QRS952.147vs810.578us.
  CleanQRSandlargerquerytilesdidnotimprovethistwo-scorepipeline. Neither
  installed;N40default811.648us/SOL59.943772remainscurrent.
- PERSISTENT_KV_DESIGN.md recordsadistinctnextarchitecture: R2/M128CTA
  keepsall768K/Vinsharedandloopsoversixquerytiles, two8KBbiasslots,
  QRS4Score/2Ppipeline,consumer224/prod32, andseparateP*onesdenominator.
  RemovesrepeatedK/Vtrafficandringprotocolwork. DESIGNONLY,notimplemented.

## 2026-09-22 04:12 KST — persistent K/V, first prototype

`persistkv2` / job14372 is implemented in `persistent_kv_body.cuh`, with
`persistent_kv.py` reusing prepare/uniform/forward scaffolding. R2/M128,
384 threads, consumer224/producer32, all six K/V boxes held immutable while
the CTA processes six query tiles. Separate denominator MMA, four scores,
two P buffers, two TMA bias half slots. Q reuse is released after QK0/1 have
consumed all Q registers; phantom bias chunks48/49 never consume next-query
slots. Per-CTA fix flags and full-query SAFE recompute are present.

Both hot/SAFE reserve168 initial registers, zero spills. Hot has four C7519
injected fences, no C7512/14/15. Initial L768 comparison is bitwise equal,
including full block; FP64 RMS0.00028883485479432215. But hot1575.022us vs
813.424us installed, whole core1595.488us vs880.555us. NOT installed, NOT
qualified beyond the initial case. Job14373 profiles the regression before
selecting another change. Current installed6fe/811.648us/SM59.943772 remains.

## 2026-09-22 04:17 KST — persistent K/V bias supply diagnosis

NCU14373: `persistkv2`1577.824us, SM30.751299%, tensor17.526647%,
occupancy15.248792%; L2sectors67154928 vs installed154991798 (56.67% lower).
DRAM read488.550MB/write143.288MB. Aggregate long-scoreboard/issue ratio
6.053 vs installed2.023; this alone does not identify the stalled instruction.

`persistkv2b4`14374 aliases the consumed Q shared buffer as bias slots2/3.
A separate query-end barrier prevents the next Q load from overwriting live
bias; producer waits for all Q register consumption before writing alias slots.
Initial case is bitwise equal, zero spills, four C7519 fences. Hot1222.058us
vs816.908us; whole core1244.630us vs883.700us. Rejected, no installation or
full qualification. The improvement over the two-slot prototype motivates
isolating bias-supply latency while retaining Q/K/V TMA.

Job14375 builds `persistkv2gb` (direct float4 global bias loads into score
registers) and `persistkv2cp` (warp0 cp.async stages two shared bias slots).
Both still use TMA for Q/K/V. These are candidates, not qualified changes.

## 2026-09-22 04:34 KST — two-score / three-P pipeline

`persistent_s2p3_summary.json` records jobs14372/74/75/76/77/78/79 and their
actual compiler/measurement evidence. None is installed.

-14375 persistent direct-global bias1347.635us vs813.728; warp cp.async bias
  2958.142us vs812.147. Both initial full outputs bitwise equal. Rejected.
-`two_score_three_p.py` implements QK(k+1), E/pack(k), PV(k-1), wait1.
  Three P buffers protect the in-flight PV(k-2) source while freeing P(k-3).
  Two score buffers reduce register pressure. Initial SAFE/uniform paths remain.
-14376 SS896.589us vs806.559; QRS883.467us vs810.496. Zero spills,128 initial
  registers, no C7512/14/15, but24 C7519 fences. Initial FP64 RMS.000288830381
  vs baseline.000288834855; not bitwise (RMSdelta7.837e-6). Rejected.
-14377 packing after PV did not remove the24 extra fences: SS902.354us,
  QRS880.856us. Rejected.
-14378 `generic_full_producers.py` adds separate Q/K, V, bias warps and one
  16KB TMA for two query halves. QRS883.711us; late-pack877.940us. Still slower
  than their installed baselines808.403/806.186. Rejected.
-14379 R2/N64/224-register QRS has C7512 despite zero spills; M64/R4/112-reg
  QRS has C7512 and spills. Both correctly rejected before GPU launch.

`dump_variant_ptx.py` produces PTX from the original build.ninja flags without
changing the extension. `m128s2p3qr-hot.sass` shows some P(k-1) F2FP operations
scheduled after the QK fence; the compiler then adds another fence before PV.
Descriptor generation alone is not a reason to add this fence: NVIDIA PTX ISA
limits this register-ordering requirement to accumulators and A fragments:
https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
Job14380 tests two explicit P-before-QK dependencies (zero-valued score add,
and dependent warp sync). These are unqualified candidates.

A fresh tiny CUDA12.9/sm90a compile in `half_exp_probe.{cu,cubin,sass}` also
corrects an older handoff premise: one PTX ex2.approx.f16x2 becomes TWO
MUFU.EX2.F16 instructions plus PRMT. No throughput gain can be assumed from
PTX vector syntax. No new half-precision attention candidate was built.

Current installed package remains6fe/811.648us/SM59.943772%, SOL90 unachieved.


## 2026-09-22 05:07 KST — descriptor constants installed and verified

`install_fixeddesc.py` installed SHA6d52ebfc4b43ca8fcbeac7de165d4572cf30d9cd2efe6ee6f63508425dc284fe.
Job14397 finished bitwise core/block and both directions L768/1024, no silent
fallback;17/17 shipped tests PASS. Job14398 installed NCU:804.416us,
SM60.630086%, tensor34.556008%, occupancy22.349430%, issue37.601392%.
SOL90 is NOT reached.

Paired L768 jobs14384/85 improve core1.06%/1.01%; L1024 job14391 is
unchanged within noise. Full20+generic40+2 custom-scale SAFE cases and
14390 racecheck/synccheck/memcheck passed. Backups in before_fixeddesc_install.
Partial-Q composites were not installed: three-way14396 compared18 balanced
rounds; composite/fixeddesc median.9976526, bootstrap95% CI[.9967246,1.0040124].
Its additional gain is not established. Current patch and manifest match6d52.

Next bounded experiment follows PROJECTED_ATTENTION_DESIGN.md: verify exact
BF16 normalization and producer Q/K/V projection before attention integration.
No projection speedup is claimed.


## 2026-09-22 05:35 KST — projected Q/K/V proof and integration screening

Projection implementation now exists in projected/. Scalar and direct STSM
proofs each pass10 bitwise Q/K/V/G/bias cases at L768. A static per-head KV
weight pack is required because concatenated [32,128] SW128 tiles have a
different K64 stride from [64,128]. STSM96 uses90 registers, zero spills,
no C7512; memcheck/racecheck/synccheck at L384, both orientations, pass.

Fused R2 pipeline, two X row buffers, two KV stages, eight bias halves, SAFE
per-CTA flags and a rounded-V all-masked fallback have been implemented.
Initial builds are resource screens only, not GPU attention benchmarks:
14404p64 spills1440/1448B,14406p96 spills244/252B,14407p112 spills168/172B
hot,14408p128ss spills112/116B hot. All initial168 registers. Onlyp64 has
C7512; all hot variants have4 C7519. None is installed or performance-qualified.

SASS in fused-p96.sass shows64-bit descriptor spills during projection.
14409p96z prevents descriptors being retained across projection calls using
an immutable shared-zero dependency;14410p96loop keeps the K16 loop rolled.
Both are compile candidates. projected/resource-screen.json records results.
Current serving remains6d52:804.416us,SM60.630086%,SOL90 not reached.


## 2026-09-22 06:08 KST: projection and R4 follow-up

Projected Q/K/V was implemented, initially checked, and rejected. Standalone
STSM96 job14405 passed10 bitwise projection cases and all three sanitizers
in both orientations atL384. Fused p96z/p96loop/p128sszm128/cl4 whole-block
latencies were2040.788/2096.739/2033.851/2355.786us versus paired installed
1395.448/1368.450/1389.713/1359.585us. NCU14416 p96z hadSM34.653835%,
1534.496us and8.561676224GB L2 traffic, more than installed4.975161312GB.
See PROJECTED_ATTENTION_DESIGN.md and projected/*json; none installed.

Half EX2 throughput microbenchmark14421 gave no gain from packed FP16:
N8 F32/F16 medians1.091984/1.101840ms; N16 2.144560/2.141584ms;
N32 4.262208/4.328512ms. Do not assume ex2.approx.f16x2 doubles H100 throughput.

Compact2KB N40 ones job14423 initially preserved core/block bitwise; hot
799.2512us versus802.7618us was near parity, graph warmup drift and SAFE
spills prevent a speedup claim. It remains an isolated dependency proof.

R4 single-producer-warp544-thread resource estimate was corrected: SM90
launch validation rounds17 warps to20, so120 regs cannot fit. Job14424
compiled96regs with spills/C7520 and was not launched. Cooperative512-thread
m128r4coop2 job14425 compiled116regs, no spills but C7520 serialization; also
not launched. Explicit rescale/repack fence variant14426 is pending screening.
Current installed SHA6d52 and804.416us/SM60.630086% remain unchanged.


## 2026-09-22 06:25 KST: cooperative pipeline screens and CTA traversal

-14426 m128r4coop2f1: fencing after rescale/repack alone left C7520.
-14427 m128r4coop2f2: unconditional PV operand fence/WG.AR removed ALL C751
warnings; hot118regs/SAFE84, no spills. Initial core/block bitwise, FP64 RMS
.000288834855 equals installed. Hot931.411us vs795.276us, core1011.828 vs883.096,
block1477.031 vs1371.593. REJECTED for latency, not installed/full-qualified.
-14428 coop3s4qr:384 cooperative threads,4 scores/2P, Q in registers,
compact N40 ones. Hot162regs/no spills but C7512; no GPU launch.
-14429 coop3s4qrz: shared-zero descriptor lifetime dependency also162regs/C7512.
-14431 coop3s4qrfull: all48 steps statically expanded; C7512 remains,168regs,
32B stack/28 stores/88 loads. Thus simple loop-boundary unrolling does not
resolve this cooperative QRS register constraint. No GPU launch.
-14430 coop5s2p1: M128/R5/LN64/3 KV stages,640 cooperative threads,96register
limit. Two S/one P: wait1 retires PV(k-1) before repacking P(k), while QK(k+1)
remains in flight. Shared226KB plus barriers fits; tail pair rows are clamped
for loads and excluded from stores. Hot C7512,192B stack,2020/1740B spills.
-14432 coop5s2p1fullip4: static48 steps plus in-place groups of4 FFMA/EX2;
C7512,192B stack,5456/5340B spills. Both R5 variants rejected before launch.

Job14433 builds basegrid from the installed6d52 source without changing its
arithmetic/TMA producer/consumer pipeline. A hot-only grid-order parameter
permutes the same6144 CTAs: query tiles interleaved over2^k row groups, k0..8,
or head/row/query order9. Batches are kept separate; non-L768 and SAFE mapping
are unchanged. bench_grid_order.py verifies full core/block bitwise and native
namespace, then captures11 variants and compares22 balanced rotated/reversed
rounds after warmup. Build and measurements pending; not a speedup claim.


## 2026-09-22 06:43 KST: grid and notification experiments rejected

-14433 basegrid: all10 CTA permutations had bitwise initial core/block output,
clean hot128regs/no spills. A broad11-graph22-round sweep appeared to save
5-6% core time for row-group tiles8/16, but CUPTI hot differed only~0.2%.
-14435 narrowed the capture set to serving/grid0/grid3/grid4,22 balanced
rounds, both orientations. The apparent gain DID NOT reproduce: grid3/4
core/serving ratios were1.00410/1.00281 starting and1.00380/1.00354 ending.
Block ratios1.00293/1.00222 and1.00285/1.00195. Runtimegrid0 itself costs
~0.8%; large-capture results cannot be used as evidence of a serving gain.
No grid variant installed. Do not claim5-6% improvement from14433.

-14434 basebiaswgrelease replaces12 bias-half empty notifications with3,
but only AFTER an explicit128-thread named barrier per consumer WG
(userIDs5..7=>hardware13..15). Initial core/block bitwise, hot814.846us versus
798.689us. Core graph934.812vs872.860us; whole-block medians drifted and are
not evidence of improvement. Rejected, no full qualification/installation.
-PTX8.8 explicitly says WGMMA .sync waits for all threads in the WARP;
.aligned requires all warps execute the instruction but does not provide
a cross-warp rendezvous. Thus one WG-leader notification cannot replace
four warp arrivals without explicit inter-warp synchronization. Primary:
https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma

-14436 paired16KB bias TMA initially failed CuTe CopyAtom rank mismatch.
The three-dimensional destination partition must flatten FOUR modes.
-14437 basebiaspairtma fixes that rank. Hot128regs/no spills/no C751 warnings.
One TMA copies both64-query halves, full barrier count1, producer waits
only the even empty slot; consumers retain per-warp arrivals but release
only after the odd QK, freeing the pair. Initial core/block bitwise, hot
808.170us vs803.181us, core896.872vs872.855us. No speedup; not installed.

-14438 baseepackqr compresses each completed E in-place into8 BF16-pair
registers, clears the upper8 score registers, and copies packed P next step.
Period rescaling acts on rounded P, so it is a NUMERICAL experiment, not an
exact arithmetic claim. SAFE unchanged. Intended to fit full QRS with4Score.
Hot still C7512 and15 C7519 (no spills); no GPU launch/accuracy evidence.
See early_packed_score.py. Do not assume the early-rounding path is qualified.

-14439 NEW m128r2s1p1twocta: M128/N32/LN128/R2,320 threads (2 full consumers,
2 independent producer warps), static96-register ceiling, no setmaxnreg.
One producer warp owns Q/K/V, the other bias. Shared3 bias halves,2 KV
stages, compact2KB N40 ones:109568B expected after alignment, fits two CTAs.
Hot QK(k) precedes PV(k-1); wait1 retires QK, E(k) overlaps old PV; wait0
retires old P, then pack P(k), refill S(k+1), issue QK(k+1) and PV(k) last.
Only one S and one P slot; exact seed64, running-max SAFE and uniform fallback
remain. Period rescale occurs after all current PVs retire. Compiler screen
requires <=96 registers/no spills/no serialization; host ALSO asserts
cudaOccupancyMaxActiveBlocksPerMultiprocessor>=2. Build/results pending.
Current installed SHA6d52 remains804.416us/SM60.630086%; SOL90 unachieved.


## 2026-09-22 06:51 KST: two-CTA resource follow-up

14439 failed in the source generator (leading-space substring mismatch),
corrected and locally validated before resubmission. No CUDA launch occurred.
-14440 m128r2s1p1twocta (16-step periods): hot94regs,160B stack,
1020/508B spills, C7520. Rejected by screen before launch.
-14441 m128r2s1p1twoctap6: six-step periods align the3-slot bias ring, reducing
its modulo/phase calculations. Hot96regs,128B stack,488/488B spills, C7520.
-14442 m128r2s1p1twoctap6f: explicit PV operand/WG.AR fence changes warning
to C7512 but leaves96regs/128B stack/488+488 spills. No GPU launches.

14443 m128r2s1p1twoctafsc is building. It retains320 threads, two producer
warps and two consumer WGs, M128/N32/LN128/R2 with two KV stages. It uses N32
PV and a scalar denominator over the rounded BF16 P, replacing N40 fusion.
Removing shared ones lets the bias ring grow to FOUR8KB halves (power-of-two
addressing), total115712B expected shared. Denominator uses explicit rn FP32
pair sums and quad XOR reductions, so its accumulation order differs from
installed Tensor Core sums; no bitwise claim. The FP64 numerical ceiling and
runtime two-resident-CTA assertion are mandatory. Hot must compile <=96 regs,
zero spills, no C7512/4/5 or C7520. No result or installation yet.

## 2026-09-22 07:03 KST: two-CTA measured rejection

14443 fsc scalar-den build removed all C751 warnings, but hot96regs has
16B stack/12B stores/12B loads; SAFE74regs/spill-free. Strict screen rejected.
14444 explicitly bounded this tiny-spill exception: SASS has3 STL and3 LDL
static sites. Runtime two-resident-CTA occupancy assertion PASSED. Initial
FP64 first-pair RMS .000288834855 equals installed; full core delta RMS
2.298304e-6 and block8.903987e-6, thus not bitwise. Despite two CTA residency,
hot929.7316 vs804.6338us, core1106.3011 vs895.4451us, block1553.9217 vs1373.07495us.
REJECTED. Two resident CTAs alone do not make this short pipeline faster.
14445 fscmma: separate N8 denominator WGMMA plus N32 PV,512B ones, four bias
slots. Hot94regs,16B stack/160B stores/144B loads,C7512; SAFE82regs/spill-free.
Resource screen rejected before GPU execution. Installed6d52 unchanged.

Next bounded numerical screen: FP16 P/V and FP16 PV accumulation, preserving
BF16 QK. pv_half_model.py rounds each K16 partial accumulator, compares four
pair rows/both orientations/four masks against FP64 and installed. It is an
error MODEL only; success would not qualify actual WGMMA or exceptional-input
handling. No FP16 PV CUDA implementation exists yet; no accuracy/speed claim.

14446 FP16 PV error model: four pair rows, both directions, ordinary all/
periodic masks give one-accumulator RMS0.9643..0.9753x installed FP64 RMS.
Two split accumulators0.8927..0.9015x, but would spend more registers.
V conversion exact on these sampled ordinary inputs. Model replaces failed
seed rows with native output, so one/none masks are NOT hot accuracy evidence.
Initial all-masked FP64 reference was incorrectly zero; script corrected to
uniform mean (the contract), not yet rerun. Do not cite the all-masked number.

14447 builds basepvhalfqr: original producer_package plus constant descriptors,
full Q RS, FP16 PV accumulator (8regs versus16 per64query-half), FP16 P and V,
FP32 denominator via separate N8 MMA, shift0/den range[2^-4,2^12] with exact
power-of-two scaling. QK remains BF16/FP32. One conversion CTA per pair-row/head
marks any inexact V conversion for original BF16/FP32 SAFE recomputation. SAFE,
generic and fully masked uniform semantics retain original V. All conversion
and fallback costs count in core/block timing. This is unqualified experimental
code; installed6d52 untouched. Compile screen/actual CUDA accuracy pending.

14448 expanded FP16 PV model reveals a disqualifying case for one accumulator:
constant V=1 produces RMS.000434/.000391 while baseline is effectively exact;
positive V=1+.05*V has1.1087/1.1082x baseline RMS, over1.05 ceiling. Two split
accumulators preserve constant V in this sample and positive-V ratios1.0294/
1.0290, but lose the register saving. One-accumulator design is NOT installable.
14447 also failed CUDA compilation: cutlass::half_t has no operator*=float at
hot rescale. No actual FP16 PV CUDA accuracy/performance result exists.

Next candidate uses LOSSLESS bias compression: staged bias originally derives
from BF16 prologue values times inv_scale. Encode that original BF16 value,
reconstruct with the SAME FP32 inv_scale multiplication before QK. Every staged
entry is checked for exact reconstruction; nonrepresentable q-tiles request the
original SAFE path with full FP32 bias. This differs from the old HANDOFF input
bias-dtype experiment, which left the staged buffer FP32 and changed no traffic.
Initial prototype counts a separate compression pass; installed6d52 unchanged.

## 2026-09-22 07:20 KST: lossless compression rejected; native R2 control

14449 basebiascompress: hot128regs/zero spills/no C751 warnings. Initial
core/block bitwise equal, FP64 RMS unchanged. SAFE196/212B spills match the
preceding N40 compile (not introduced by compression). Hot870.7698vs803.5450us,
encode_bias104.614us, core1005.608vs886.759us, block1481.256vs1381.008us.
Even eliminating the entire extra compression pass cannot rescue the hot
regression. Rejected; no installation. BF16-code -> FP32 multiply per consumer
cost outweighs this transfer reduction. All-masked model reference was rerun
correctly in14448; those rows still use substituted native SAFE, not half hot.

Historical review confirms3Score/3P and2Score/3P were already measured slower,
so do not repeat those standalone pipelines. New native R2 control instead
preserves the INSTALLED4Score/2P/N40/two-group ordering and independent TMA
producers, but uses two full224-register consumer WGs plus full32-reg producer.
384-thread launch pool168*384=64512 safely covers224*256+32*128=61440.
Full Q RS applies only fast; generic/SAFE use SS. R2 is used consistently in
SAFE list decoding and Python fix-list capacity; all hardcoded third-row reads
are removed. This trades bias reuse/CTA count for sufficient registers and Q
operand shared-read reduction, with no numeric approximation. Not the older
persistent-KV or M256 two-consumer variants. Build/measurement pending.

Native R2 job14452 submitted. Independent basefixedoffsetqr removes hot
seeding/per-row nm/rescaling entirely, uses literal log2 offset-64, and sends
nonfinite outputs or final denominator outside[2^-84,2^-44] to the original
SAFE path. Full Q RS plus4Score/2P, R3/prod32/cons160 remain. This can remove
live softmax-offset registers; unlike14340 static seed, the hot offset never
changes. BF16 probability rounding changes, so it is a numerical experiment
requiring full FP64 checks if fast. No lower precision QK/PV/accumulators or
exp approximation is introduced. Current source/binary installation unchanged.

14452 native R2 QRS: hot168regs,zero spills/no C751, adequate CTA pool.
Initial full core/block BITWISE, FP64 RMS unchanged. Hot894.1876vs803.8366us,
core981.965vs878.541us, block1426.093vs1371.426us. Rejected.
14456 fixed-offset full QRS: still C7512 despite zero spills/128 initial regs.
Resource screen stopped before GPU execution. Removing nm/seed registers alone
is insufficient; no numeric or speed qualification. No SS control run.

Next Q-layout control preserves arithmetic and all pipeline scheduling: only
fast Q shared layout is SW128 with physical D64; TMA's D64 box zero-fills beyond
logical D32, WGMMA reads the first32 columns. Q SMEM grows24KB (fits), K/V and
bias unchanged. It tests Q operand shared-bank mapping without Q RS registers.
Also collecting source counters from the actual installed6d52 binary to avoid
relying solely on the pre-N40 source profile. Installed main remains804.416us,
SM60.630086%. These are experiments, not claimed improvements.

14464 installed6d52 SourceCounters collected (12 passes), exported full SASS
source table. Actual sampled MIO stalls17684/17915 (~98.7%) are on MUFU;
LDS contributes75 samples. Measured shared excessive wavefront count is ZERO.
Long-scoreboard samples mostly sit on mbarrier TRYWAIT predicate branches,
including producer K/V waits and startup Q/K waits. These samples must not be
presented as proportional kernel latency or assumed global-memory pressure.
Actual issued-ratio stalls: longSB2.0898, MIO1.9529, wait1.6120, GMMA.4955.

New baseexpsplit8 retains installed arithmetic, buffers, producer roles,
commits/waits and SAFE. Hot steady body computes row0's8 exponentials, issues
prefenced PV(k-1), prefetches bias(k+3), then computes row1's8 exponentials.
This spreads the MUFU burst and gives bias loads more time before the next QK
fence. No extra sync, approximation or Q RS. Need inspect actual SASS scheduling
and latency; compiler may move loads, so source order alone is not evidence.
Q SW128 job14463 is still the independent exact memory-layout control.

14463 Q SW128: clean hot128 regs/no spills/C751; initial core/block BITWISE.
Hot806.5028vs797.2348us. Three-round core drifts981->899us; block difference
is not improvement evidence. Rejected, no installation.
14468 EX2 split8: clean hot128/no spills/C751, initial core/block BITWISE.
Hot805.7638vs803.9174us (near parity), graph core917.767vs872.836us with drift;
no win. SASS inspection explains why: installed compiler ALREADY interleaves
HGMMA, bias LDS and late MUFU within E. New code does not establish the hoped
for additional overlap. Installed3448 instructions versus3472 candidate, no
change installed. Do not interpret source E-before-PV as actual SASS ordering.

Next baseqbootstrap uses the source-profile startup-Q wait evidence. Hot Q's
three8KB TMA transactions are issued by the three consumer WG leaders before
derive(), rather than serially by K's producer warp. The Q barrier init becomes
3 with one expect8KB/arrive per consumer; all current waits remain. Producer
warp1 can start K sooner, and mask metadata loads overlap Q startup. Every
steady-state MMA/EX2, buffer, and K/V/bias producer role is unchanged. Generic
and SAFE retain the original Q producer/barrier. No performance claim yet.

Additional bounded resource experiment basemixedqr176 keeps R3 and the full
installed4S/2P/N40 pipeline: consumer WG0 gets176 regs and full Q RS; WG1/2
get152 regs and ordinary Q SS. Producer remains32. The same initial512*128
pool exactly covers(32+176+152+152)*128=65536. Separate compile-time consumer
lambda instantiations, selected uniformly per FULL WG, preserve valid dynamic
register allocation; no partial producer realloc. Generic/SAFE retain all
SS160 consumers. This does not claim that SS152 or QR176 will compile clean;
strict spill/serialization screen gates GPU execution. Q bootstrap14476
remains separate. No installation change or SOL90 claim.

14476 Q bootstrap: clean hot128/zero spills/no C751, initial core/block
BITWISE. Hot804.3842vs805.3246us (~0.1%, near parity); graph core915.345vs
873.024us with monotonic candidate warming, not evidence of a win. No install.

basewgpoly3 is a DISTINCT SFU/FMA balance test: one complete consumer WG uses
cubic exp2 (same coefficients/domain as14284), while the other TWO use MUFU.
Older polynomial tests interleaved25%/50% polynomial logits inside EVERY warp;
they were slower. Whole-WG specialization aims to let separate warps use the
underutilized FMA pipe without placing both dependency chains in every warp.
All three preserve4S/2P/N40, producer32/consumer160, masks/seeding/rescale/SAFE.
Only fast WG0 changes probability rounding. Existing cubic max relative error
1.88081e-4/RMS4.83242e-5 is not full attention qualification. Compile screen
and actual FP64/performance gates are required; no expected outputs changed.
A separate integer-bit range-reduction polynomial was considered on CPU, but
its cubic error1.38295e-3 is worse and it was NOT implemented or GPU tested.

14483 mixed-register QR176/SS152: hot128/no spills but C7512 remains. Rejected
before GPU; which consumer specialization causes serialization is undiagnosed.
14490 spatial WG cubic: hot128/no spills/no C751, initial FP64 gate passes
(.0002888883 vs .0002888349). Hot1021.4706vs797.476us, core1103.4195vs886.7661,
block1585.4515vs1372.2896. Rejected; no installation or full qualification.

New m128s2p2qrn64 reduces the earlier R2/N64 QR two-score/three-P design to
two P buffers. After E(k), a second wait1 retires PV(k-2) before recycling P(k),
while QK(k+1) remains in flight. V empty arrives after that retirement. An8-step
period aligns score/P, bias and KV rings. This saves16 probability registers;
it preserves the original N64 arithmetic and SAFE, but does NOT imply bitwise
equality with installed N32. Strict compile resource screen precedes FP64 and
actual latency. Installed6d52 remains unchanged, SOL90 unachieved.

14494 N64 two-P: both hot/SAFE168 regs, zero spills/no C751/C7520, initial FP64
passes(.0002871343 vs .0002888349). Hot1140.8634vs794.4378us, core1192.105vs
880.0035, block1673.0557vs1379.3959. Rejected. Resource savings did not improve
runtime; no independent-producer follow-up warranted at this deficit.

baseshufexp1 replaces25% of hot exp2 with a32-entry table held ONE value per
warp lane, selected by SHFL, then first-order interpolation. Old basetable1/2
used1024-entry SHARED lookup and failed; this design removes the random LDS.
Each lane's table value is computed once per CTA. CPU1,000,001-point sample
[-100,20] relative max5.91306e-5/RMS2.62330e-5 (32bins); 16bins gives worse
2.37921e-4. Probability rounding changes. GPU correctness/speed not yet known.

baseqrk0 retains the first K16 slice of BOTH Q query halves (8 packed regs),
so every QK issues RS(k0) followed by SS(k1), preserving reduction order and
balancing shared-Q traffic across all chunks. Earlier partial_q_registers kept
ONE complete M64 half, making alternating chunks entirely RS or SS; its paired
gain was noise. Unused Q k1 register slices are not fenced, allowing DCE of
their startup copies. Same4S/2P/N40/producer/consumer budgets and SAFE. No result
or bitwise qualification yet; compile screen gates GPU use.

14499 shuffle-table25%: clean hot128/zero spills/no C751, initial FP64 gate
passes(.0002887290vs.0002888349). Hot884.9024vs801.5672us; graph core candidate
1044->1032->959us is warming, not a win. Rejected. Shared-table traffic removal
does not offset shuffle/range-reduction/interpolation overhead. No install.

14503 baseqrk0: C7512 remains, hot128/zero spills, rejected before GPU.
Historical audit correction: partial_q_kblock.py ALREADY implemented this
K16 split for older6fe. Its plain build had C7512; basef40qrk0ip4 removed it
and paired14393 vs current6d52 showed core.9948561/block.9966909 (starting
only). The prior half-query composite is a DIFFERENT candidate. The newly
created duplicate helper was removed; build_qr_k_slice now reuses the original.
Next baseqrk0ip4fixed combines that exact4-value FFMA/EX2 grouping and K16 QR
with current constant descriptors. No prior measurement of this exact
combination found. Need both-direction paired validation if competitive.

Screen-only candidate_hot_index.json now indexes109 initial comparisons;
it is NOT qualification and historical baselines vary. Specialized::attention
is included to avoid accidentally recording split4f's short SAFE kernel as hot.
Audit found basef40smallones14423(initial799.25vs802.76us, exact) never received
a paired follow-up. pair_small_ones.sbatch checks BOTH directions with16
balanced rounds, unchanged current6d52 serving. No gain claim from3-capture data.

14511 small-ones pairing stopped before GPU because its older builder does
not print the newer screen's BUILT marker. The job now verifies completed
link command, actual ELF, current serving SHA, and the SAME no-spill/no-C751
resource conditions directly. Pair.py still asserts full core/block bitwise.

14509 baseqrk0ip4fixed compiles hot128/no spills/no serialization (C7519
extra fences remain), initial full core/block BITWISE. Hot792.2688vs797.6704us
(~0.68%); three-round graph drift means not a qualified win. Paired BOTH
directions next. Small-ones14512 first direction core.9953691/block.9973937;
second direction pending. No installation yet.

14512 small-ones paired BOTH directions: starting core.9951534/block.9956012;
ending core.9953691/block.9973937. Full captured outputs bitwise equal.
14513 QR K16/ip4/fixed paired BOTH: starting core.9951766/block.9982451;
ending core.9950558/block.9971930. Full captured outputs bitwise equal.
Each reduces core roughly0.5%; new baseqrk0ip4small combines them. Independent
gains cannot be added without measuring the combination. No install yet.

14514 FP16 PV+FP16 denominator model fails1.05x ceiling: normal periodic
1.05375/1.05545, all1.06936/1.06240, positive/negative V1.19699..1.20303.
ConstantV=1 is exact, but that alone is insufficient. NO CUDA implementation
of FP16 denominator pursued. One/none masks substitute native SAFE/uniform
output and do not qualify hot arithmetic. Results pv-half-den-model.json.

14516 composite initial core/block BITWISE; hot789.4498vs795.0148us.
14518 BOTH-direction16-pair composite core ratios.9907964/.9905786;
block.9947833/.9936338. Prototype full20 and generic40 bitwise PASS.
Custom-scale harness then failed before executing CUDA because its broad
CHECK_SCALE string replacement appended '-scale' to the .so path. Fixed
check_base.py to change only JSON output suffixes; accuracy criteria unchanged.
Prototype sanitizer was not reached. Final-namespace qsmall_package stage14520
will rebuild and repeat20+40+2scale+3sanitizers; installation waits for this.

14519 centered-V FP16-PV model: subtract each V row/head/dimension's mean,
accumulate centered FP16 V with FP16 PV and FP32 denominator, add mean back.
Four sampled pair rows, both directions, seven mask/V patterns. This changes
V precision (conversion is not exact); model-only, NOT a CUDA qualification.
Results pv-center-model.json. Further implementation requires full numeric
coverage and accounting for the centering/conversion pass, not just hot timing.

Composite14518 paired-bootstrap95% intervals (10k resamples): starting core
[.9902814,.9910882], block[.9925822,.9979970]; ending core[.9896154,.9908047],
block[.9888591,.9944303]. Point estimates are paired medians, not ratios of medians.
L1024 both directions14521 bitwise; core ratios.9957960/.9967165, block
.9951916/.9944003. Large clock variation exists in1024 rounds, so no precise
large gain claim; no measured regression.

Centered-V model14519 hot normal RMS ratios.97775..98962, constantV exact,
positive/negativeV approximately1.00000. Prototype basepvcenterqr14523 reuses
producer_package's pre-N40 full4S/2P schedule with full Q RS, FP16-PV output
accumulators and separate FP32 denominator. A per-row/head CUDA pass computes
FP32 mean per dimension and converts centered V to FP16; the epilogue adds
mean back after normalization. Nonfinite/FP16 overflow conversion requests
original BF16/FP32 SAFE; ordinary inexact centering conversion is intentional.
The prototype fixes old14447's unsupported half*=float with explicit half casts.
This is a different-numerics experiment, NOT installed/qualified. Conversion
pass time must count in full-core/block comparisons. Full numeric cases still
required if it is competitive. qsmall_package qualification14520 is separate.

14520 final namespace qsmall_package completed62 bitwise cases and15 sanitizer
cases: racecheck0 hazards, synccheck0 errors, memcheck0 errors. Installer gates
these artifacts, both-direction768 gains, both-direction1024 nonregression,
manifest hashes, and exact previous6d52. Installed97634810 via atomic replacement;
backup before_qsmall_install. Generic M1 SHA3c766e02 is unchanged. Postinstall
serving14524 and NCU14525 are now pending, do not label older profile current.
Cumulative codex_core_sol90.patch regenerated from before_fast_install.

14524 installed verification COMPLETE: bitwise core/block both directions,
L768/1024, five masks, correct hot flags/no flash fallback,17/17 shipped tests.
14525 installed NCU:793.536us/SM61.041971%, tensor34.790760%, occupancy22.264219%,
issue37.997760%, DRAMread462.530560MB/write145.006336MB, L2sectors154744499.
SOL90 NOT reached. Reports/HANDOFF/current profile defaults updated.

14523 centered-V FP16-PV: clean hot128/no spills/no C751; initial FP64 RMS
.0002820340 vs .0002888349 passes, but hot861.9440vs799.4540us, conversion
227.2508us. Whole core1144.1936vs890.1340, block1634.6445vs1368.3235. Rejected.
Even eliminating conversion alone would not rescue its hot slowdown. No full
numeric qualification or installation. All owned jobs through14525 completed.

## Next goal turn: deeper bias ring (installed97634810 revalidated)
Previous turn is PROGRESS: qualified and installed Q-K16/small-ones package,
current793.536us/SM61.041971%; SOL90 still unachieved, active goal retained.

14529 compares basebias4unroll (control) and basebias6unroll. Both copy the
CURRENT installed source. Six full M128xN32 bias slots need32KB extra shared
memory but preserve all Q/K/V buffers and producer32/consumer160 budgets.
The hot bias producer iterates stream-relative half indices over12 slots;
consumer full phases and exact half-slot releases use that same sequence.
The existing three16-chunk periods are expanded with compile-time p=0,1,2,
so modulo-six addresses/phases fold to constants. All original exits for
n_tiles1..6 remain; unused past-end bias lookahead is zeroed rather than reading
never-filled extra slots. QK/E/PV order, all numeric operations, masks and
SAFE are unchanged. Four-slot control uses identical expansion/producer-loop
structure, distinguishing ring depth from unrolling. No stagger change yet.
Resource screen precedes bitwise/FP64 and measured timing. Not installed.


## 2026-09-22: installed-qsmall follow-up evidence
Previous goal turn is PROGRESS: current-source counter collection and bounded
CUDA experiments completed; measurements rule out deeper bias buffering and
simple exp grouping/cache reversal. Installed97634810 remains unchanged.

14529 expanded4-slot control hot827.8314vs799.4272us, 6-slot815.6970vs783.9678us;
initial core/block BITWISE, no hot spills/serialization. Both rejected.
14532 narrows the past-end guard to only uninitialized6-slot physical halves
8..11 when n_w==1 (4-slot control needs no guard). Still BITWISE on initial
full-stream case:4slot809.3380vs797.5458us;6slot809.7244vs794.1734us. Reject.
No full short-mask/sanitizer qualification of these slower candidates.

14530 installed partial-Q exp groups2/8 both clean128 and initial BITWISE.
Hot795.8666vs795.3604 /795.8384vs795.4606us: parity, no paired qualification
or installation. Existing groups4 retained.14535 caches SECOND K16 Q slice,
SS(k0) then RS(k1), preserving accumulation order: clean128/BITWISE but
799.7742vs799.1486us, parity. Cached first slice retained.

14536 current976 SourceCounters saved installed-qsmall-source.ncu-rep and
CSV/SASS. MIO samples16791,16581 (~98.75%) on MUFU.EX2; shared excessive
wavefronts remain zero. Short-scoreboard7297 samples,4718 at WARPGROUP.ARRIVE
(vs3418/7158 in older fixeddesc capture). Current executed instructions
283412895 vs282361390 previously. Separate sampled profiles are not paired
latency proof; source run778.816us does NOT replace qualified installed
SpeedOfLight793.536us/SM61.041971%. qsmall-stall-comparison.json aggregates
instruction attribution; sample shares are NOT wall-clock fractions.

14539 retries full-Q RS on current fixed-descriptor/small-ones source.
baseqsmallfull:128initial/no spills, but C7512; rejected before GPU.
baseqsmallmixed168:producer24 + fullQR168 + partialQR160 + partialQR160
exactly fits65536 initial pool; no C7512 but hot16B stack/76B stores+loads,
rejected before GPU. Which role spills is not yet diagnosed. Generic/SAFE
kept producer32/cons160. No producer partial-WG setmaxnreg is used.

14554 NEW N48/M128/LN96/R3 retains3 consumer WGs160 and fullproducerWG32.
Two-score/two-P or three-P schedules derive from isolated N64 family, but
use full Q RS and N40 fused PV+Den, original BF16QKV and FP32 accumulators.
N48 saves QK issue frequency relativeN32 without N64's two-consumer drop.
Three-P variant emits exactly32 chunks (24+8), never48; lastP is slot1.
Two-P retains its explicit secondwait1 before reusing P. Both use2 KV stages
and8 bias half-slots. Different softmax seed chunk width makes this a numeric
experiment, NOT a bitwise claim. Resource+FP64+latency screens precede any
further qualification. Job14554 now building; not installed.


14554 N48 two-P compiles128/no spills but C7512; rejected before GPU.
N48 three-P compiles128/no spills/no serialization and initial FP64 RMS
.00028749958vs.000288834855 passes, but hot1060.0186vs799.1876us;
whole core1094.2950vs865.5450, block1571.7603vs1381.8185. REJECTED.
No full numerical qualification. N48 itself is not proved inherently bad,
but this two-score/three-P/three-consumer/N40 pipeline is clearly slower.

14559 bounded follow-up to mixed168: SASS CFG traversal from both consumer
USETMAXREG allocations proves ZERO reachable STL/LDL there; all20 static
spill instructions reachable only from producer24. Stack16B, aggregate76B
stores/76B loads. screen_mixed_producer_spill.py records this narrow exception
without altering ordinary resource gates. Initial full core/block BITWISE,
but hot796.8380vs791.6164us; reject. No serving change, no claim that producer
spills are free. SASS and mixed168-producer-spill-screen.json retained.

14565 NEW m128r4coop3p2ip4v2: four cooperative consumer WGs,512 threads,
M128/N32,3 scores/2 P,N40 output, Q SS, no dynamic register repartition.
Derived from old correct m128r4coop2f2, but bias init(k+2) occurs a body
before QK(k+2), E(k) stays in score storage until next body, and P(k-1)
recycles retired P(k-3) after the sole wait1. QK is one chunk ahead. This
removes the old two-score design's second wait and exposes less LDS latency.
Period24 aligns3-score/2-P/8-half-bias/2KV rings; drained rescale pendingE23/47
in sc[2]. TMA thread0 load_bias(k+8) follows the current QK retirement;
K/V refill follows all WG stage releases. Exact in-place FFMA/EX2 groups4
control live registers. SAFE unchanged except array size, still scalar exp.
First zero-P dummy PV commit is essential so step1 wait1 retires QK1;
initial job14564 was cancelled after noticing this missing commit BEFORE
qualification, v2 unique namespace14565 fixes it. Check build resource gate
before any performance claims. Installed976 remains793.536us/SM61.041971%.

14565 cooperative3-score v2 compiles hot114/SAFE84, zero spills but C7512;
rejected before GPU. New follow-ups14567(q2) and14568(ru0) are isolated
namespaces m128r4coop3p2ip4v2q2 /...ru0. q2 keeps the same3Score/2P storage
but primesQK0/1 and uses wait2; each body packs oldE(k-1), initializes the
freed score to bias(k+2), issuesQK(k+2), computesE(k), then commitsPV(k-1).
Thus it trades bias LDS lead time for deeper QK overlap, no extra buffers.
The first dummy PV remains. ru0 keeps v2 math/schedule and changes only
ptxas register-usage-level=0. Both must pass the unchanged no-spill/no-C7512
resource gate before GPU timing. Jobs14567/14568 pending at this note;
all owned earlier jobs through14565 terminal. No installation change.
CORE_SOL90_REPORT now explicitly records installed ptxas's16 extra C7519
fences, correcting its older source-level 'without another fence' phrasing.

## Producer SFU helper and intervening rejected candidates

14567 cooperative3-score q2: hot122/SAFE84, zero spills, C7512; rejected.
14568 same v2 register-usage-level0: hot114/SAFE82, zero spills, C7512;
rejected. Neither reached GPU qualification.

14570 m128r4coop3p2ip4v2epack compacts E into8 BF16-pair registers early.
Clean resources/no serialization; initial full core/block BITWISE and FP64
pass, but hot1015.6012vs793.8100us, core1083.7795vs884.3671,
block1561.3095vs1373.9375. REJECTED. Early rounding before periodic rescale
is a numerical experiment, not a universal bitwise claim. 14571 q2epack
still C7512 plus16B stack/32B stores/16B loads; rejected before GPU.

14574 baserangeexp4f4/f8 uses a bounded quartic on x/4+16 in[-1.5,.5],
coefficients scaled2^-16, two squares, native EX2 outside the domain.
CPU real max relative error .0006610979 is NOT CUDA qualification.
f4 hot128B stack/384B stores/432B loads; f8 128B/408B/432B. Rejected.
14575 baseqsmallepackfull combines full Q RS with early BF16 E compaction;
rescaling would force SAFE. Hot144B stack/784B stores/672B loads, rejected
before numerical or performance qualification.

14590 baseproducerexp4v3 is a distinct producer-assisted SFU experiment.
Three original consumer WGs retain MMA and12 of16 EX2 values/thread; each
TMA producer warp services four EX2 values/thread for its corresponding
warp in all three consumers. Two request/response slots per consumer warp,
each with explicit ready/done phase barriers. Producers service queues
while waiting for TMA empty slots AND after finishing TMA, avoiding cyclic
waits. Lane0 acquire is broadcast then __syncwarp before other lanes read.
Partial Q register cache removed to recover8 registers; full-WG budgets
remain32 producer/160 consumer. Extra shared storage24960B. No arithmetic
approximation. Generic/SAFE retain their original producer protocol.
14585/14588 were cancelled during compilation for pre-benchmark fixes:
helper drain must precede the hot-loop break; acquire needs warp visibility.
14590 hot128/no spills but C7520 compiler serialization; rejected before GPU.
SASS retained as baseproducerexp4v3-hot.sass. No hot LDL/STL exists.

14596 baseproducerexp4v4 removes the dynamic completed-result branch using
the known pipeline position: prologue collects E1; body0 follows prologue
or drain so needs no collection; bodies1..15 collect E(k-1); drain collects
pending E before rescaling; final pack uses that already-collected result.
Each response is collected once, no reload can overwrite rescaled E.
Build/resource/initial correctness+latency job pending at this note.
Installed97634810 remains unchanged,793.536us/SM61.041971%; SOL90 not reached.

14596 static helper completion still C7520, now64B stack/384B stores/320B
loads; rejected before GPU. 14600 v5 instead uses v3 completion plus an
explicit uniform PV fence. Hot128/no spills but C7511 serialization.
The previous screen regex missed C7511 and erroneously allowed its initial
benchmark: full core/block BITWISE, hot1937.3546vs799.3974us, core1900.4599
vs871.9904, block2418.7479vs1355.3401. REJECTED. Screen now rejects
C7511/2/4/5/7/C7520 and the explicit serialization warning text. Archived
v3/v4/v5 logs all fail the corrected gate. No performance qualification.

14599 compiler-only controls on installed976: ptxas-O2 and register-usage-
level0 both compile128/no spills, initial full outputs BITWISE. All6944 SASS
machine words of the o2 hot kernel exactly match installed, not merely
instruction counts; its CUPTI791.124vs795.4436us is measurement variation,
NOT a code speedup. ru0 differs (7424 machine words,3712 instructions vs3472)
and hot797.1622vs793.8180us gives no initial improvement. Neither advanced
to paired qualification. codegen-and-helper-screen.json records the exact
comparison and rejection-gate verification. No installation change.

14606 factored_bias_exp_model tests a distinct arithmetic idea: precompute
exp(bias-rowmax)*2^-64 shared across pair rows, then polynomial exp(QK).
FP64 MODEL ONLY, original BF16 Q/K/V, four pair rows/both directions and
Q/K strengths1/2/4. Native factorization model RMS1.0029..1.0051x atstrength1.
But low-degree models fail the1.05 ceiling on ordinary inputs: cubic[-1,1]
RMS1.1181/1.1197x, cubic[-2,2]12.464/12.641x, quartic[-2,2]2.681/2.717x.
Only~68%/95% of normal logits lie inside those domains. A per-lane native
fallback outside the interval would also retain MUFU issue in most warps.
No actual factored CUDA implementation or speed claim; these fits rejected.

14609 baseproducerexp4v6 removes the helper-result writes to WGMMA score
accumulators entirely. pack_chunk waits for the result once and converts it
directly into the first two BF16 P registers. Remaining P entries use the
ordinary score values. Period drains remember two exact row scaling factors
for the outstanding helper result, applied on collection; no conditional
completed counter and no drain-time response reload. Original v3 producer
service protocol and32/160 full-WG budgets remain. Build+corrected resource
screen+initial FP64/bitwise/latency pending; not installed or qualified.

14609 direct-packed helper completion still has C7520 and large hot spills:
128B stack/1072B stores/952B loads. Corrected resource gate rejects it before
GPU. No accuracy or latency evidence for v6.

14617 texture_exp_probe is a NEW independent hardware experiment, not an
attention implementation. A1D float texture with linear interpolation stores
2^x over[-126,127] at1/128 spacing. Max relative error vsnativeEX2 over2^20
points1.92417e-5, RMS8.40819e-6, ordinary[-80,-40] max1.70611e-5. Four modes
use0/25/50/100% texture exponentiation in a384-thread,16-independent-chain
microkernel; dynamic190KB and runtime occupancy assertion enforce1CTA/SM.
Measured30.752/77.683/152.464/301.325us: texture is decisively slower, so no
attention integration. This is microkernel evidence, not an attention bound.
14614 was the initial probe;14617 adds explicit shared initialization sync
and the occupancy assertion, and is the evidence of record. CUDA texture
linear filtering is hardware low-precision interpolation (official guide):
https://docs.nvidia.com/cuda/archive/12.8.1/pdf/CUDA_C_Programming_Guide.pdf

14620 baseproducerexp4v7 moves BF16 conversion into the producer, returns
two packed uint registers instead of four FP32 registers. Response storage
halved; generic/SAFE unchanged. Any hot periodic rescale forces original SAFE
recomputation, rather than assuming early BF16 rounding commutes in extremes.
Thus not universally bitwise-qualified; full FP64/fallback checks would be
required if fast. Producer EX2 itself remains original native arithmetic.

14622 baseproducerexp4v8 additionally collapses the helper ready/done barriers
from per-warp to per-consumer-WG. Each barrier has FOUR arrivals, one from
each warp leader; producer warps still process disjoint32-thread slices.
All128 consumer threads wait on the SAME done address before WGMMA, and
all input slices are published before a producer sees ready. This preserves
phase/slot ownership and tests whether the previous per-warp completion
control caused compiler divergence/extra register lifetimes. Both v7/v8
builds pending at this note. Installed hash97634810 reverified unchanged.

14620 packed-response v7: C7520,128B stack/1076B stores/972B loads; rejected.
SASS CFG attribution finds ALL412 static spill instructions reachable from
consumer160 and ZERO from producer32. Lower producer register pressure will
not repair this consumer bottleneck. producerexp4v7-spill-attribution.json
and baseproducerexp4v7-hot.sass retain the evidence.
14622 WG-wide packed v8: C7511,80B stack/336B stores/296B loads; rejected.
14625 WG-wide FP32 v9 keeps v3's once-only collection/rescaling and has zero
spills, but C7511 insufficient WGMMA register resources; rejected before GPU.
WG-wide barriers remove C7520 in these two builds, but do not by themselves
make the four-score helper fit the asynchronous MMA register allocation.

14630 baseproducerexp3score is the next structural change, implemented in
producer_three_score.py on the exact FP32 WG-wide helper protocol. Hot uses
three scores/two P, still three consumer WGs160 and producerWG32. PackE(k-1),
reuse its score immediately for bias(k+2)+QK(k+2), computeE(k), issuePV(k-1)
last. One wait2 per step preserves QK2-ahead; bias no longer prefetches a
body ahead, saving16 consumer score registers. Shared TMA rings unchanged.
Prologue primes QK2 in S2 and QK3 in recycled S0 after packing E0. Full static
positions2..47 keep modulo3/shared descriptors constant. Original rescale
points17/33 and final7/15/23/31/39/47 are retained. Short streams finish and
release the final V stage through the matching static branch; helper request
count remains exactly8*n_tiles. Generic/SAFE retain their original schedules.
Build/resource/initial correctness+latency pending; no qualification/install.

14630 three-score helper removes all C7511/2/4/5/7/C7520 serialization
warnings. Hot128initial,32B stack/36B stores/108B loads remains, so ordinary
resource gate rejects before GPU. SASS CFG attributes all36 static LDL/STL
to consumer160, zero to producer32. Several values are stored once early
and reloaded in later chunks/drains. Files baseproducerexp3score-hot.sass and
producerexp3score-spill-attribution.json record this resource progress,
NOT runtime qualification or a performance gain.
14636 baseproducerexp3scoreru0 uses identical three-score math/schedule with
ptxas register-usage-level0 to reduce aggressive long-lived hoisting. Pending
at this note; installed97634810/SOL61.041971 unchanged.

14637 is a bounded DIAGNOSTIC benchmark of the existing14630 three-score
build, explicitly allowing its known32B stack/36B stores/108B loads and36
consumer-only spill sites. It still requires zero MMA-serialization warnings.
screen_three_score_spill.py pins those counts; the ordinary no-spill gate is
UNCHANGED. Measuring this bounded case determines whether the architecture
is worth further register work; it cannot qualify installation. Initial
full-output/FP64/native-dispatch checks remain mandatory. Pending at this note.

14637 completed: three-score helper initial full core/block BITWISE and
FP64 identical, but hot1822.4162vs795.1790us, core1785.3833vs869.7715,
block2310.4089vs1351.4055. REJECTED. Removing compiler serialization was
necessary but did not make the producer request/response architecture fast.
14636 register-usage-level0 reduces remaining spill to8B stack/28B stores/
28B loads; ordinary resource gate rejects. The much slower measured parent
does not justify another spill exception or deeper helper tuning.

14638 basenative3scoreqr reuses the now initially correctness-tested static
three-score/two-P schedule, removes ALL helper queues/computation, and uses
the saved score registers for FULL Q RS. Independent original TMA producer
warps and native consumer EX2 remain. Both Q query halves/K16 slices are
loaded and fenced once after Q TMA; QK keeps k0 then k1 order. Every original
periodic rescale point and short-stream exit is retained. This is distinct
from older three-score/three-P and four-score/full-Q attempts. No change to
generic/SAFE or installed package. Build/resource/initial timing pending.

14638 native three-score/full-Q still C7512 and8B stack/4B stores/4B loads;
resource screen rejects before GPU. No runtime claim for this variant.
All owned jobs through14638 are now terminal. This continuation added no
serving change or qualified speedup. Installed97634810 remains793.536us,
SM61.041971%, with its earlier bitwise/sanitizer/serving proofs intact.
SOL90 goal remains active and unachieved. No expected vectors were changed.


14645 basenative3scoreqrdep adds an actual dependency from all eight packed P
words to the next bias LDS address (bitwise AND with host-proven params.zero).
It removes C7512: hot128, zero stack/spills and no serialization warnings.
Initial full core/block BITWISE and FP64 identical, but hot849.0212 versus
798.4192us, core960.0883 versus870.8557; rejected. This proves a compiler
ordering improvement, not a speed gain. Dependency is formed before PV issue,
so no in-flight WGMMA A registers are read. 14646 identical native3scoreqr with
register-usage-level0 still C7512 and is rejected before GPU.

14649 basenative3scorelatebias moves bias addition after full-RS QK on the
three-score schedule. QK starts ScaleOut::Zero; bias is added with FP32 RN
before seeding/exponentiation. Bias empty barriers are returned only after an
explicit warp-wide dependency covering all four LDS.128 results. Clean hot
compile; initial FP64 RMS0.000288862082 versus0.000288834855 (within1.05),
NOT bitwise. Hot888.0686 versus798.6220us, core977.7526 versus870.2586;
rejected. No full numerical qualification or installation.

14654 baseqfullorderedbias applies the P-to-bias LDS dependency to installed
four-score/two-P scheduling, retaining its one-body bias lead time and using
full Q RS. Hot128/zero spills, but C7512 remains. Strict gate rejects before
GPU. All owned jobs through14654 terminal. Installed97634810 unchanged;
793.536us and SM61.041971 remain the current qualified NCU results.


14662/14663 early-even-pack initial builds failed PTX type checking: bf16x2
cvt requires a b32 destination, not an f32 register. Corrected using scoped b32
PTX temporaries plus mov.b32 to the score storage; unique v2 namespaces keep
failed artifacts separate. 14672 baseevenpackpartialv2 and14673 fullv2 both
C7511. Hot spills respectively152B stack/408 stores/452 loads and152B stack/
472 stores/808 loads. Rejected before GPU. Both fuse4 FFMA,4 EX2,2 BF16 pairs,
retain packed pairs in even score slots, and force SAFE instead of rescaling
rounded P. This still worsens compiler resource allocation; no speed claim.

14671 m64r2s2p2qrf40 is a new M64/R2 two-score/two-P two-CTA design from the
initially correctness-tested M64/R4 two-P prototype. Four bias slots replace
eight; full Q caches8 registers;24 static chunks; independent K/V producers;
N40 PV/den uses original V TMA plus2KB ones descriptor (no V repack).
Hot80 initial registers, consumer104/producer32 (30720-register CTA pool),
zero stack/spills and no serialization warnings. Runtime two-resident-CTA
assertion PASSED. Initial FP64 RMS0.000288830381 versus0.000288834855, core
not bitwise (different existing M64 seed expression), all finite. Hot989.7916
versus789.5422us, core1058.6768 versus875.3094. REJECTED. More resident CTAs
with full Q RS is still insufficient to offset this M64 schedule's costs.

14676 half_bias_transport_model is a NEW error model, four pair rows and both
directions, BF16 Q/K/V and BF16 probability rounding, bias strengths.25/1/4/16.
Scaled-bias FP16 transport passes ordinary strength1 (~1.0122x starting RMS)
but fails strength4 (~1.2490x) and16 (~2.0265x). Do not implement universally
rounded scaled FP16 bias. Raw bias FP16 agrees nearly with exact-control model
because the current prologue's float32 bias values originate from BF16.
This observation motivates lossless raw-BF16 transport, not a model speed win.

14681/14682 baserawbf16biaspartial/full are new CUDA candidates using raw
BF16-representable bias bits before multiplication by inverse scale. Stage
retains original scaled FP32 for generic/SAFE and writes packed raw16 data
plus one exact-representability flag per staged key column. Masked/padding
values remain -Inf. Non-BF16 live bias marks the whole query/head tile for
SAFE recomputation; this fallback can differ in rounding from original hot
and requires numerical qualification (not an all-input bitwise claim).
Fast bias TMA/LDS bytes halve. Two LDS.128 preload eight packed words into
future score storage; QK expands them and applies the SAME host-rounded
inverse scale with FP32 FTZ RN multiply before the original QK chain. Full
variant uses the saved bias register slots for full Q RS. Four-score/two-P
schedule, native EX2, N40 PV, and independent producers otherwise retained.
Build/resource/initial accuracy and latency pending; no serving change.


14681 raw-BF16 partial-Q transport: clean hot128/zero stack/spills/no WGMMA
serialization. Initial FULL core/block BITWISE, FP64 identical; SAFE remains
only3.2064us (no full fallback). Hot878.9370 versus796.5760us, core940.8224
versus864.9552; rejected. Bias stage17.6892 versus13.8046us also adds overhead.
SASS evidence raw-bias-static-instructions.json: hot3472->4088 static
instructions, LDS.12884->42, FMUL.FTZ196->516. These are STATIC counts, not
executed work; reduced LDS traffic does not by itself pay for restoration.
14682 full-Q raw bias hasC7512 despite hot128/zero spills; no GPU launch.

14685 baserawbf16biasearly moves raw expansion/scaling from issue_qk into the
previous body's init_chunk, keeping partial Q. Clean hot128/no spills/no
serialization; initial full core/block BITWISE, FP64 identical. Hot869.0112
versus792.9474us, core956.7293 versus865.1389, stage18.0094 versus13.7406us.
Rejected. Earlier reconstruction does not make this compressed transport
competitive. No full numerical/sanitizer qualification or installation of
any raw-bias candidate. Integer layout proof separately exhausts all2048
packed words and restores each of256 original16-element fragments in order.

All owned jobs through14685 terminal. This continuation tested three-score
ordering/late bias, early probability packing, two-CTA M64 N40/full-Q, and
lossless raw-BF16 bias transport. No new serving speedup qualified. Installed
SHA97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b remains
793.536us/SM61.041971%; SOL90 goal ACTIVE and not reached. Original expected
vectors unchanged. Producer-helper and raw-bias families now have measured
negative results even after removing serialization or moving conversions;
do not repeat them just by changing register-budget knobs.


Next continuation classified prior turn as PROGRESS: new measured resource/
latency evidence excluded early packed E, two-CTA M64 and raw bias transport.
Installed97634810 reverified at turn start. No owned jobs were live initially.

New shared_probability.py uses same-consumer shared P with THREE4KB slots per
consumer WG, retaining original four-score/QK2-ahead and producer32/consumer160
pipeline. SS N40 PV reads original V+ones descriptors. P packing writes exact
BF16 pairs in position-independent K_SW64 layout. After stores, explicit
fence.proxy.async.shared::cta and bar.sync hardware8+cwg (128 threads), BOTH
with memory clobbers, publish all warp slices. Slot j%3 was last read by
PV(j-3); the PREVIOUS P(j-1) publication barrier follows every warp's wait
that retires PV(j-3), so it also proves all old reads finished before stores
for P(j). First three slots have no old readers. Final tail has the same
prior-barrier protection. No register-A reads of in-flight MMA and no cross-WG
helper exchanges. P storage adds36KB, and removes hot two-P register operands.
Generic/SAFE retain RS P and empty-base optimization avoids extra shared bytes.
Partial/full Q controls test whether saved registers make full Q viable.

14699/14700 source generation failed ambiguous SharedStorage match (Pipe and
Traits both define it). Fixed precise Q-field marker and validated both
transforms. 14702/14703 were cancelled during compilation after finding
CUTLASS fence_view_async_shared/NamedBarrier::sync have no compiler memory
clobber. No GPU result. 14708/14709 unique v2 namespaces include explicit
proxy/barrier clobbers and are building. Resource gate then initial FP64 and
whole-output check precede timing; full race/sync/memory checks required if
competitive. This is an unqualified new architecture, not an improvement.

14708/14709/14716 all-shared-P results: partial v2 hot1161.8350 vs799.3732us;
full-Q v2 hot1090.0504 vs794.9558us; full-Q STSM v2 hot1055.3912 vs799.3654us.
All clean hot128/zero stack/spills/no WGMMA serialization, initial full core/
block BITWISE and FP64 identical. Full Q can compile clean when register P
is removed, but shared P traffic/synchronization makes all three slower.
Rejected; no broad qualification or installation.

14720 mixed v2 BUILD FAILED before GPU: generic Traits<0> has no shared P
member. Combined if constexpr(kFast && hh==1) in generic lambda left hh
substitution pending, so nvcc checked that nondependent member for SAFE.
Fixed by nesting outer kFast guard before hh guard; v3 fresh namespace.
Mixed design keeps even P in GPR, puts odd P in two shared slots (24KB),
uses full Q RS, restores even-P fences. Previous odd publication j-2 follows
all warps' waits retiring PV(j-4), protecting the reused odd slot. One STSM/
publication barrier per odd chunk; compile-time slot arithmetic removes %3.
No performance claim; v3 build/resource/initial numerical screen pending.

14727 basesharedp3mixedv3 compiled clean hot128/zero stack/spills/no WGMMA
serialization. Initial FULL core/block BITWISE, FP64 identical. Hot882.5522
versus794.2146us, core995.1792 versus868.4822; rejected. Halving shared-P
publication frequency and retaining even P in GPR improves on all-shared
versions, but is still slower than installed. Generic/SAFE existing spill
counts are separate from the hot resource screen. No installation.

14730 coop4sharedp3qr is a NEW R4 cooperative3-score/QK2-ahead full-Q/shared-P
candidate. Each of4 WGs synchronously loads its full Q into16 registers,
then WG bar.sync proves all Q readers finished. Q's8KB/WG becomes P slots0/1;
extra4KB/WG supplies slot2. Bias ring8->6 saves16KB to keep shared capacity.
Six bias slots bootstrap/replenish with seq+6 and phase seq/6; unchanged KV
ring uses two128-wide stages and replenishes after last PV retirement.
24-step period divides bias6/P3/KV8. P publication uses STSM then explicit
proxy fence and WG bar.sync, both memory clobbers. Prior publication follows
wait retiring the reused slot. Dummy P(-1) uses slot2, real P0 slot0. SAFE
retains Q shared and register P, using the same six-slot bias producer.
Build/resource/initial accuracy pending; no performance/qualification claim.

14730 coop4sharedp3qr and14732 prefenced control both C7512 plus32B stack,
208B stores/400B loads; hot126. Rejected before GPU. Removing hot PV's
redundant WG.AR restores compiler-injected C7519 but does not repair the
allocation. SAFE82/zero spills. No speed result or installation.

NEW coop4pairedp6qr groups both64-row query halves. TWO score fragments
instead of three; full Q RS; QK(pair j+1) commits before PV(pair j), so the
next wait1 retires QK while prior PV can overlap both current exponentials.
P uses THREE pairs (six4KB tiles per WG); one STSM publication barrier per
pair. Prior pair publication follows waits retiring pair j-3 and therefore
protects storage reuse at j. First three pair slots have no prior readers.
Q aliases the first8KB/WG, extra16KB/WG holds other four tiles. LN64/three KV
stages plus four bias slots preserve the232448B shared ceiling. Bias ring
advances two sequences per hot step, using original seq-major staged data.
KV ring is primed for three64-wide tiles and replenishes after pair PV
retirement. Drain every12 pairs retains original24-chunk rescale period;
no already-exponentiated P is pending, so only outputs/nm are rescaled.
SAFE keeps serialized original math with the adjusted KV/bias geometry.
Generation validated; build/resource/initial numerical check pending.

14738 coop4pairedp6qr: hot118/zero stack/spills, but C7514 accumulator-read
serialization. Rejected before GPU. The paired calls still had the second
half's warpgroup fence and helper bookkeeping inside an uncommitted QK
pair. SASS shows inserted wait0 between individual HGMMA instructions.
V2 uses explicit pair helpers: prepare descriptors/fence both scores FIRST,
then four consecutive QK MMAs and one commit, then bias releases. PV pair
prepares both shared-P descriptors and shared V descriptor before its four
MMAs/one commit. This addresses the instruction ordering cause; build gate
still rejects any C7514. First R2 version was only generated, never queued.
R2 v2 is a separate two-CTA control:256 threads, two WGs, two64-wide KV stages,
two bias slots, same three shared P pairs. <=116224B static shared assertion
and runtime occupancy>=2 assertion. Same no-spill/serialization gate applies.
New namespace per changed candidate; build/initial numerical results pending.

14745/14746 pair v2 R4/R2 both zero stack/spills (hot117/120 respectively),
but C7514 persists. The fence-between-half explanation was a hypothesis,
NOT a confirmed root cause; moving bookkeeping outside did not resolve it.
Neither variant passed the compiler gate or ran on GPU; the two-CTA runtime
assertion has NOT yet run. 14749 R4 v3 removes the last-iteration conditional
QK group: an unused final QK on zero-seeded scores/live stage0 keeps exactly
two committed groups per body. PTX retained for pipeline-stage analysis.

14749 R4 pair v3 still C7514, hot116/zero spills. Keeping a dummy final QK
does not alone repair stage analysis. Preserved PTX prologue's four QK MMAs
has no score-reading operation before its commit; descriptor-only integer
arithmetic appears between k slices. The loop header still merges initial
wait0 with later wait1, a possible conservative stage-analysis cause.
R4 v4 adds one zero-P pair in free shared slots4/5 before first real pair.
QK0 followed by this zero PV establishes two initial groups and allows
UNCONDITIONAL wait1 in every body. First reuse of slots4/5 at pair2 is protected
by pair1's publication after retiring dummy PV. Final dummy QK retained.
Zero-P group adds no nonzero math; it is a bounded compiler-control experiment.
No timings or qualification yet; all prior pair versions rejected pre-GPU.

14757 coop4pairedp6qrv4 removes C7514: hot114/zero stack/spills. The initial
QK0+zero-P PV pair enables unconditional wait1, which was the effective
compiler fix. Initial FULL core/block BITWISE, FP64 identical. Hot1241.4158
versus799.3158us, core1262.0438 versus872.3772; rejected. Two scores cannot
issue next QK until both current E/P packs finish, losing QK/SFU overlap.
14764 coop2pairedp6qr2ctav4: clean hot116/zero spills, runtime two-resident-CTA
assertion PASSED, initial FULL core/block BITWISE/FP64 identical. Hot1106.4528
versus793.1374us, core1168.6656 versus877.6550; rejected. No installation or
broad qualification for either paired version. SM occupancy alone is not a win.

NEW m64coop2s4p3qr reduces query M128->M64, freeing one20-register output
half and8 full-Q registers. FOUR score buffers retain original QK2-ahead and
one-body bias lead: packP(k-1), QK(k+2), E(k), PV(k-1), initBias(k+3).
Three shared P slots per WG; Q aliases slot0 after synchronous load/barrier,
extra8KB/WG supplies slots1/2. Same previous-publication slot ownership proof.
R2/256 cooperative threads, LN64/three KV stages/four bias slots, static
shared<=116224B plus runtime residentCTA>=2 gate. Full Q RS, native exact
FMA/EX2, N40 SS PV on original V+ones descriptor.24 key chunks, period12
(aligns score4/P3/KV6); all outstanding groups drain before row rescaling,
including pending E for the one query half. SAFE retains shared Q/register P.
Build/resource/initial accuracy pending. Installed976 rehashed unchanged.

14770 m64coop2s4p3qr: clean hot118/SAFE62, zero stack/spills/no serialization.
Runtime two-CTA assertion and initial FULL core/block BITWISE/FP64 passed.
Hot1495.5142 versus789.2448us; core1557.1207 versus878.2089. Rejected.
14774 kv4b2 reallocates shared capacity: four64-wide KV stages, two bias
slots. Bootstrap bias0/1 then refill2 after QK0 release; hot body k loads
bias(k+3) after QK(k+1) released that slot in preceding body. SAFE refills
seq+2 after consumption. Initial resource/occupancy/bitwise/FP64 checks all
pass, but hot1410.2592 versus793.4702us, core1473.0208 versus892.1488. Rejected.
TMA lead helps this schedule modestly; neither shared-P M64 variant is useful.

14777 m64coop2s4p2sskv4 is a NEW register-P control preserving four scores and
QK2-ahead/bias one-body lead. M64 has one20-register output, and removes the
full-Q8-register cache; this may fit two8-register P buffers within128/thread.
SS QK reads original Q. RS PV keeps P in GPR, removing ALL hot P shared stores/
publication barriers. P(k-1) slot(k-1)%2 last belonged to P(k-3), retired by
this body's wait2 in each warp before its fence/overwrite. Pack/fence P precedes
QK's hardware fence; PV stays last committed. Extra16KB formerly used by P
funds a fourth64-wide KV stage; four bias slots remain. Source generated and
strict resource/occupancy/initial accuracy checks pending. No serving changes.

14777 register-P M64 control: clean hot126/SAFE62, zero stack/spills and no
serialization, runtime two-resident-CTA assertion PASSED. Initial FULL
core/block BITWISE and FP64 identical. Hot1160.6390 versus797.6190us;
core1272.3689 versus885.0858. Rejected. Removing shared-P publication is an
improvement over the new M64 shared-P family, not over the installed core.

This continuation produced new same-consumer shared-P, mixed-P, cooperative
four-score/three-score, paired two-score, and M64 four-score controls. Nine
measured variants all passed initial full-output bitwise/FP64 and all lost
on speed. Earlier compilation failures were rejected before GPU. Pair C7514
was eliminated by uniform initial group counts, which is useful compiler
control evidence but NOT a speed gain. Do not replay this shared-P family or
standalone3Score/3P (already negative14314/14315) via budget/period tweaks.
All owned jobs through14777 terminal. Summary artifact is
shared-probability-results.json, regenerated by summarise_shared_probability.py.
CORE_SOL90_REPORT.md and HANDOFF.md updated with measured negative outcomes.
Installed SHA97634810cdba1e91fe4aecb8bdfa610dd050dd39e7504ffb8babee8c46d4077b
reverified unchanged. Existing793.536us/SM61.041971% remains installed result;
SOL90 goal remains ACTIVE/unachieved. No expected vectors or serving files
changed, and no new broad sanitizer qualification was warranted for slower
candidates. Generic M1 binary and codex_core_sol90.patch remain unchanged.

Next continuation: previous turn classified PROGRESS, since it established
nine slower numerical-valid controls and the paired-loop C7514 group-count
fix. Installed976 hash/authoritative source retained; no owned jobs live at
start. Other jobs14779/14780/14781 and training13228 belong to other work.

14782 basepackfencedep is a NEW installed-family ordering control. Actual
installed SASS has P F2FP conversions on both sides of QK's WG.AR (e.g.
2030 fence,2040/2060/2080 conversions), then ptxas inserts a second PV fence.
All eight packed P words are OR-reduced, ANDed with existing Params.zero,
and bit-XORed into future QK score[0] before its existing operand/WG fence.
Params.zero is a.q.size(3)>>40 in launch_m1.cuh, exactly0 for guardedL768.
Thus bits/operations remain numerically identical while a real dependency
forces conversions before the score write and hardware fence. No in-flight
register is touched: reused P was retired by wait2; future score has bias
loads and has not yet entered its next QK. Generic/SAFE retain original code.
Added integer instructions may outweigh fence savings; inspect SASS and
strict no-spill/no-serialization screen before initial FP64/full output and
latency. Keep PTX. No performance or qualification claim yet.

14782 basepackfencedep: hot128/no stack/spills but C7512 plus all16 C7519
remain. Rejected before GPU; serialized SASS WG.AR78/WG.WAIT78. No timing.
P-to-score integer/float bitcasts introduce another accumulator version in
PTX; the relation to allocation is a hypothesis, not a measured root cause.
Next basepackfenceqdep instead XORs the zero dependence into ONE packed Q
cache word before QK's existing hardware fence. Body k's wait2 has retired
QK(k), which uses the same query half as futureQK(k+2). Only the opposite
QK(k+1) may still read Q, so modifying this half's A word is safe even before
considering the invariant zero. Default dependency0 prologue calls eliminate
the bit update. This avoids modifying/bitcasting a FP32 accumulator bank.
Same OR/AND overhead and strict compiler/initial accuracy gate; no gain claim.

14787 basepackfenceqdep also retains all16 C7519 plus C7512. Hot128, zero
stack/spills. Rejected before GPU; no correctness or latency measurement.
The zero dependence through a cached Q word did not repair compiler scheduling.

14796 half2_exp_model is a new HFMA2 polynomial numerical/cost model. Cubic
and quartic coefficients fit exp2 fractional range[-.5,.5] with FP16-rounded
Horner stages; exponent reconstruction remains FP32, preserving the hot
2^-64 probability scale. Actual SASS contains HFMA2/HFMA2.MMA, no spills.
Four pair rows, both orientations, Q/K strengths1/2/4, original BF16 inputs
and BF16 P rounding: every model finite, approximate RMS ratios<=1.01174.
This is NOT an actual QK/attention implementation or full qualification.
Register microprobe native/cubic/quartic respectively36.877/97.304/96.029us
starting and37.102/97.429/96.186us ending. Cubic ~2.63x native; no integration
justified by this initial cost evidence. Microprobe includes range conversion,
reconstruction, exceptional-input predicates, and common feedback operations;
its ratio is not an attention-kernel slowdown measurement. Coefficient fit,
two numerical JSONs and half2-exp-probe.sass retain reproducible evidence.

NEW m128compactprod3p2dep: four complete consumer WGs at threads0..511,
plus one ordinary TMA producer warp512..543. STATIC120 register cap gives
65280 registers/CTA; no partial-WG setmaxnreg. M128/N32/LN128, three scores,
two register P slots, N40 PV/den, shared Q, two KV stages/eight half-bias slots.
Producer polls both empty queues without blocking on either; ready slots use
the original TMA loader and transaction barriers. Consumer code performs no
TMA refill, preserving QK2-ahead while removing cooperative producer stalls.
P(k-1) is OR-reduced before its PV issue and ANDed with host-proven Params.zero
to order freed-score reuse/bias loads. Previous P ownership is retired by
wait2. Strict no-serialization/no-spill/<=120 register gate plus runtime
resident-CTA check precede initial numerical and timing checks. Source generated
and reviewed; not installed or qualified. Installed976 remains unchanged.

14800 compact producer REJECTED before GPU: ptxas overrides maxrregcount120
with entry-specific96 from thread count. Hot91 actual regs,192B stack,
4608B spill stores/3392B loads and C7512. SAFE82/no spills. The initial
544*120=65280 budget ABOVE IS INVALID: local CUDA12.9 cuda_occupancy.h
lines1496..1500 compute regsAssumedPerCTA using warps rounded up to the
number of SM subpartitions.17 warps rounds to20;20*32*120=76800>65536.
96 gives61440. This refutes the proposed static partial-producer workaround;
do not rerun it with only register-budget tweaks. No runtime/speed evidence.
Official occupancy discussion also confirms per-subpartition allocation:
https://forums.developer.nvidia.com/t/optimisation-of-occupancy-summary-table/265684

14802 basepackfencewarpdep is a last bounded ordering control: OR all packed
P words and use ~(OR & Params.zero) as __syncwarp's mask before the existing
QK fence. Every lane's mask is0xffffffff. This adds a data-dependent warp
barrier without rewriting either Q or score MMA registers. Reused P ownership
was retired at wait2. Generic/SAFE are unchanged. Inspect actual SASS/fence
counts and strict resource/initial numerical/time gates; no gain claim yet.

14802 terminal/rejected: hot128/zero stack/spills but13 C7519 and C7520
remain. Actual SASS has78 WARPGROUP.ARRIVE and78 WARPGROUP.DEPBAR, versus
installed39/22; compiler serialized all78 HGMMA instructions. First body's
all8 F2FP are now before data-dependent WARPSYNC and QK WG.AR, so conversion
ordering alone is NOT sufficient to remove the injected PV fence/compiler
dependence. No GPU numerical or latency run was permitted by the strict gate.
basepackfencewarpdep-hot.sass and kept PTX retain the evidence. Do not repeat
P-to-score/Q/warp-mask zero-dependency controls via small scheduling tweaks.

All owned jobs through14802 terminal. This continuation added no qualified
speedup. Half2 HFMA2 numerical model passed its initial limit, but its register
microprobe is~2.6x slower; compact producer's naive register budget was refuted
by actual compilation and the local CUDA occupancy implementation. Three
P-order dependency placements did not fix WGMMA codegen. Summary artifact
half2-and-fence-results.json is generated by summarise_half2_and_fences.py.
Installed976 SHA reverified; existing793.536us/SM61.041971% remains current.
No serving code, expected vectors, or installed binary changed. SOL90 ACTIVE
and unachieved; no broader tests were warranted for these rejected candidates.

Next goal continuation: previous turn classified PROGRESS, because new numeric/
micro-cost evidence rejected HFMA2, actual compilation refuted the static
544-thread register budget, and SASS refuted P-conversion ordering alone as
the PV-fence fix. Installed976 rehashed; owned jobs14802 and earlier terminal.

14811 builds baseqthreehalf0/1: cache THREE of four Q K16 fragments (12GPR)
instead of installed two (8GPR). One M64 half uses RS(k0)/RS(k1), the other
retains RS(k0)/SS(k1); two choices test the asymmetry of the current loop.
Arithmetic order, all four scores/two P, TMA producers and masks unchanged.
The fourth/full Q fragment previously failed C7512, but three is distinct.
half0 clean hot128/no spills/no serialization, initial full core/block
BITWISE and FP64 identical. CUPTI793.2672 vs795.8106us is a small difference;
three graph rounds showed major warmup/drift and do NOT prove a speedup.
Balanced paired measurements in both directions submitted; half1 still building.

14817 baseqfullscalarden is a NEW numerical candidate retaining installed
four-score/two-P producer/consumer scheduling. Hot uses N32 PV and per-thread
FP32 sums of the SAME rounded BF16 P. Four column-owner lanes reduce partial
denominators only at rescale/output. The duplicate N40 sum columns disappear,
saving four live registers across the two query halves, and V descriptor no
longer computes an offset to ones. Full Q RS uses both K16 slices for both
halves. Scalar P reads all precede its new asynchronous PV issue; existing
wait2 retires its old use. Rescale decisions use the reduced denominator,
then scale local partial sums exactly. Generic/SAFE remain N40+original sums.
This changes denominator association, so it is NOT a bitwise guarantee and
must pass FP64 checks before timing; resource gate first. No serving changes.

14811 half1 also clean128/no spills/no serialization and initial full outputs
BITWISE, hot789.2900 versus791.1760us. Both three-quarter variants have40 QK
HGMMA in static SASS:30 RS/10 SS, compared to installed20/20. WG.AR39 and
MUFU288 remain; this change removes another half of installed shared-Q reads.

14820/14821 half0 standard16-pair runs: core ratios.995346/.995324, block
.997382/.997801. Ending core samples showed large clock variation.14827 uses
72 balanced rounds,8 subrounds,2 graph replays/sample across serving/half0/half1.
half0 core.995384/.995588, block.997478/.997400; bootstrap95 intervals fully
below parity. half1 core.998501/.998652, weaker than half0, so half0 selected.
Audit found captured block outputs can share the preallocated output buffer:
pair.py and q_three_interleaved.py now CLONE each replay's output before the
next graph overwrites it. Earlier timing data remains timing evidence, but
its captured block equality alone was insufficient; initial bench.py already
cloned outputs.14833/14834 rerun with independent snapshots, both lengths and
directions, recording q-three-interleaved-cloned-* and -L1024-cloned-*.

14817 scalar denominator/full Q: C7512,48B stack,112B stores/116B loads,
hot128. Rejected before GPU.14826 baseqfullwarpden replaces scalar additions
by two SM80 BF16 m16n8k16 MMAs/warp with ones. Keeps only two denominator
registers per query half; duplicate columns are temporary outputs, no shuffle
reduction needed. Clean128/no spills/no serialization, initial full core/block
BITWISE/FP64 identical, but831.1804 versus791.8066us; reject current schedule.
14835 baseqfullwarpdensplit issues the first denominator K16 MMA after QK and
before E, second after E/before PV. This preserves the exact denominator chain
while giving the first MMA result a full E phase before the second reads it.
All P reads precede new PV; old P use retired at wait2. A new overlap control,
not a performance claim. Generic/SAFE unchanged; pending strict screening.

14832 qthree_package built in FINAL ta_core_broadcast/triattn_broadcast name:
SHA4be5b7cdd291150436b9b048318b6cac7977b0db82b9a4f748520d398cfb4c32.
compare_q_three_package.py confirms all6928 hot SASS machine words EXACTLY
match measured half0 prototype. Generic flag0's7024 and SAFE1024's13472 words
EXACTLY match installed976. Twenty full-shape,40 generic and2 custom-scale
bitwise checks passed; full-shape three-sanitizer suite still running.
Not installed yet. Installer prepared with strict resource,62-case, all three
sanitizer, paired/permutation-bootstrap, manifest and exact-codegen gates.

14832 and14835 terminal COMPLETED0. 14832 full-shape strided racecheck,
synccheck, memcheck: five cases each, zero hazards/errors. Cases are dense,
prefix, empty_batch, late_seed, forced_safe (not one_key in this subset).
14835 split SM80 denominator: clean128/no spills/no serialization and initial
full core/block BITWISE, FP64 unchanged, but833.1260 versus791.5320us, graph
core975.8368 versus868.3926. REJECTED; no further qualification warranted.

14836 COMPLETED0, installed4be via strict gated install_q_three.py. Backup976
in before_qthree_install. Installed actual default/optout both directions,
L768/1024, five masks BITWISE with expected native flags1073741824/0 and no
flash fallback;17/17 shipped vectors pass. NCU784.896us, SM61.347178%, tensor
34.964713%, occupancy22.276943%, sharedLSU58.470807%, DRAM607.538944MB,
L2sectors155482440=4.975438080GB. SOL90 UNACHIEVED. Only incremental speed
claim is14833's paired core reduction.4585%/.4534%, not isolated NCU ratios.
Default verification/profile prefixes, cumulative patch, report and HANDOFF
updated. Final manifests rehashed. Expected vectors/generic M1 unchanged.

14838 NEW lean producer experiment on installed4be. Producer K/V loops are
specialized outside the tile loop to reduce simultaneous descriptor state;
whole producer WG delays deallocation until after Q issue and mask derive,
before any consumer-dependent wait. Hot empty-stage waits use explicit
mbarrier.test_wait.parity.acquire.cta shared polling plus compiler memory
clobber. This removes try_wait's suspend-path temporary registers, which
accounted for most20 producer-only static spills in old mixed168. Barrier
counts, phases, TMA bytes, consumer release frontier and all math unchanged.
Plain control keeps32/160 budgets and three-fragment Q. Mixed uses full-WG
24 producer +168 full-Q consumer +160/160 three-fragment consumers, exact
65536-register pool. Generic/SAFE retain original paths. Both require strict
zero-spill/no-serialization screen before GPU; no gain claim yet.
Reference contract: CUDA12.9 PTX section9.7.13.15.16, test_wait is nonblocking,
parity and acquire/CTA synchronization match the old wait. Spin issue cost
may offset saved registers; actual generated code/time decide, not source.

14838 plain producer control clean128/zero spills/no serialization and initial
full core/block BITWISE/FP64 unchanged, but806.1936 versus795.4444us; reject.
Static3928 instructions versus installed3464, same288EX2/39WG.AR/22DEPBAR.
Mixed168 still building; producer issue/lifetime reduction alone is not a win.

14848 baseqthree1024 adds a separately guarded hot536870912 for squareN=S1024,
H4/usual scale. It applies the installed three-Q fragments, grouped exact EX2,
independent K/V producers and constant descriptors to the existing L1024 path.
Compile-time shape constant1024 replaces768 and n_ktiles8 replaces6 only in
that new instantiation. All min-clamps remain, including the last partial R3
row group atN1024. Original hot0/SAFE1024/L768 hot1073741824 remain present.
No lean producer or spin waits in this candidate. Separate module/namespace;
no serving change yet. check_base.py now optionally accepts SOL_FULL_LENGTH
1024 for actual full-shape tests, with distinct -full1024 output names.

14838 terminal COMPLETED0; plain producer candidate rejected for latency.
Mixed24/168/160/160 still hasC7512 and16B hot stack,28B stores/20B loads,
so no GPU launch. SASS156WG.AR/156DEPBAR confirms serialization across the
two consumer branches. SAFE's consumer lambda also changed codegen to64B
stack/196B stores/212B loads. Neither lean candidate merits installation.
lean-producer-results.json and both -hot.sass files retain the evidence.
Do not infer that producer spin waits or lower budgets improved throughput.

L1024 qualification queued with afterok14848: it first runs16 paired rounds
in both directions with independently cloned outputs, then requires core
ratio<.9975 and block<1.002 in both before20 fullN1024/B2 bitwise cases,
40 generic and2 fullN1024 custom-scale checks. No sanitizer or installation
is authorized by initial timings alone; final namespaced package would still
need exact SASS equivalence and full-shape sanitizers before publication.

14851 is the queued afterok14848 L1024 paired/numerical qualification job.
The final-package sanitizer gate remains outstanding until a candidate passes
performance and accuracy; this is a qualification requirement, not a new
user-permission requirement.

14848/14851 COMPLETED0. L1024 hot clean128/no spills/no serialization; initial
full core/block BITWISE, FP64 unchanged. Hot1741.4256 versus1820.3146us.
Paired16rounds14851 starting core1966.960->1879.503, ratio.9583745185;
block2859.938->2802.666, ratio.9755635504. Ending core1989.822->1905.414,
ratio.9602863723; block2900.083->2836.358, ratio.9750634617. Both captured
core/block outputs independently cloned and bitwise equal. Prototype20 full
N1024/B2,40 generic,2 fullN1024 custom-scale bitwise cases pass.
CPU machine-word comparison: prototype oldflags0/1024/1073741824 exactly
match installed4be, including L768hot6928 words. q1024-results.json saved.

14856 stage_q1024_package is RUNNING in final namespace. It requires allfour
flags' machine words equal prototype, oldthreeflags equal installed4be,
20fullN1024 +20fullN768 +40generic +2customscale, then fullN1024/B1 strided
racecheck/synccheck/memcheck eachfive cases. install_1024_q.py prepared with
those gates plus both paired performance limits, manifest and allowed-diff
checks. finish_1024_q.sbatch queued afterok14856; it installs only if all
explicit gates pass, then verifiesactualnative1073741824/536870912 both
orientations/L768/1024,17 shipped tests and NCU atbothlengths. No permissions
needed; all work within authorized scope. Installed remains4be until publish.

Final q1024 package SHA9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225;
prototype3837b41d3a902b7776fe98710e0460297a171ef7f8b12153432be7d721a5721e.
82 final-package bitwise cases pass; exact machine-word gates pass for all
four flags. Racecheck has passed dense/prefix/empty_batch/late_seed, still
running forced_safe as of this note. finish job14862 queued afterok14856.
finalize_1024_q.py is prepared but NOT RUN: it verifies installed artifacts,
actual serving JSONs and both NCU exports before updating reports/defaults.
It converts adaptive NCU ns/us/ms and byte/KB/MB/GB units explicitly; do not
assume L1024 duration exports in microseconds. Installed remains4be now.

14856/14862 completed successfully; package9365 installed and actual serving/profiles verified. Documentation, default verifier flags/prefixes and cumulative patch updated. SOL90 remains ACTIVE and unachieved.

This continuation is PROGRESS: qualified and installed another0.45% L768
core improvement and the first exact L1024 specialization,4.16%/3.97% paired
core gain. Owned jobs14832/14835/14836/14838/14848/14851/14856/14862 terminal;
no experiment or verification job from this turn is left pending. Producer
lifetime/spin/mixed-budget controls are rejected, not installed. Current
rebuild source is the installed9365 package; older builders pin historical
hashes and must not be rerun against it blindly. Generic M1 digest3c766e02
reverified. Final report/default flags/manifests checked. SOL90 is active and
unachieved; do not mark the goal complete because these changes are faster.

## Resident K/V, four M64 consumer WGs (jobs14909/14911/14915)

Installed9365 remains unchanged. New independent B1/L768 family retains
K/V for two pair rows over six query tiles, with four M64 consumers and one
full producer WG. Initial640*96 budget funds producer32 + four consumers112.
Q16KiB aliases two bias half-slots, released by16 consumer-warp arrivals
whose address depends on every Q register word. Bias slot release has8
arrivals; query-done has16. RESIDENT_FOUR_NOTES.md records the protocol.

14909 resident4m64s3qr (24 unrolled key chunks) rejected before GPU: hot184B
stack,184B spill stores/loads, C7512. SASS has46STL/46LDL and100HGMMA with
100WARPGROUP.ARRIVE/100WARPGROUP.DEPBAR. Stores precede the query loop and
include hoisted descriptor/address values.
14911 completed both loop controls, each rejected before GPU.
resident4m64s3qrloop6 and resident4m64s2qrloop8 both removed all stack/spills
and compile at initial96 registers, but retain C7512. No timing/accuracy
claim is supported by these builds.
14915 now tests matching loop6dep/two8dep controls: all eight packed-P words
feed opaque host zero in the K/V descriptor address, preventing descriptor
setup from remaining loop-invariant. Compiler gate remains mandatory.

14915 completed: descriptor dependency controls both still C7512, zero
stack/spills; rejected.14920 shared-Q three-score also C7512. Shared-Q
two-score passes compiler/runtime resources,initial full core/blockBITWISE
and unchanged FP64RMS,but1608.151us versus795.629us hot.14924 K-resident/V
streaming two-stage,bias8 hybrid clean andinitialBITWISE,but1003.387us
versus791.574us hot. Both rejected;installed9365 remains unchanged.
RESIDENT_FOUR_NOTES.md has full ownership and measured limitations.

14928 shared-P hybrid completed. Clean96 initial registers,232448B shared,
oneCTA;three scores/two shared P slots preserve initial fullcore/block
BITWISE andFP64RMS. Hot1244.955us versus788.050us, rejected.
resident-four-results.json now records all9 variants, hashes/resources/SASS
andthree initial timing comparisons. RESIDENT_FOUR_REPORT.md andHANDOFF
updated. No installed files changed;9365hash reverified.

This continuation excludes another concrete architecture family but adds
NO QUALIFIED SERVING SPEEDUP. All ownedGPUjobs14909/14911/14915/14920/14924/
14928 are terminal, no benchmark is pending. Six compiler rejections and
three latency rejections remain isolated. SOL90 stays ACTIVE and unachieved;
do not mark complete or blocked based on these negative experiments.

## Latest9365 scaled-FP16 Q/K probe (job14934)

Previous continuation classified PROGRESS through new evidence: nine
four-consumer resident candidates exclude that family, with no serving
speedup. Installed9365 hash reverified and14928 terminal at this turn start.

New numerical candidates baseqscaledcurrentfull/three clone installed9365,
including N40 denominator fusion, independent K/V producers and fixed
descriptors. Separate namespaces/modules only. Fast flags107/536 interpret
preconvertedFP16 Q/K; Q scale=scale*log2e, staged bias=fma(oldbias,c,-64).
Hot EX2 has no per-scoreFMA or seed/rescale, but denominator outside
[2^-84,2^-44] routes originalBF16 operands/bias through SAFE. Global Q/K
conversion and bias conversion are INCLUDED in measured core time.
Noncontiguous Q/K uses the original generic flag0 path.

Converter initially uses exact BF16 roundtrip checks for FP16 encoding;
non-roundtrippable elements become NaNs to force original-data SAFE. This
can be too conservative for small subnormals and must not conceal the hot
path's numerical error. bench.py now reports the actual per-call fix count
for direct modules, synchronizing only in the initial correctness phase,
before timed/profiler phases. Build resource gate precedes all GPU launches.
No precision, performance or installation claim yet.

14934 terminal COMPLETED0. Full-Q scaled fast flags107/536 retain C7512
without spills, rejected before GPU. Three-Q scaled flags compile clean128
and initialRMS .000282642 versus .000288835 passes, but2062/6144 CTAtiles
use SAFE in the sampled call. Hot770.974vs795.297us; conversion250.144us,
bias transform11.776us, SAFE958.846us. Timed core1980.992vs872.780us.
REJECTED. This is only a24us hot gain before paying for conversion; no
prologue integration or relaxed-conversion guard is justified by that result.
Generic0 and SAFE1024 machine words match installed9365 exactly for both
variants. scaled-q-current-results.json and bothfast-flag SASS files saved.

Installed9365 linearSASS inspection finds17 PV pairs with P F2FP writes
after the preceding QK fence. This is structural evidence, NOT an all-path
register-hazard proof and NOT grounds to remove hardware fences.
installed9365-pv-fence-defs.json has addresses/registers/pack definitions.
14947 now builds exact basepprobeboth/basepprobeeven controls. Packed P's
eight words feed a four-LOP3/OR dependency into the address of the real
bias-ready mbarrier test. The returned predicate controls the normal bias
wait, so the probe is useful, not a dummy read. P conversion moves before
that probe and the QK fence. Both adds a valid full-slot probe for the second
half; even changes only the existing first-half probes. Bias half1 cannot
be overwritten before its ownrelease, so the next full-barrier phase cannot
complete before itsoldread. All stores/loads and phases remain unchanged.
Compiler/resource gate andinitialbitwise/FP64/timing checks remain pending.
Installed9365 still unchanged.

14947 cancelled after10:18 because both cc1plus processes remained in D-state
file-read waits; node01 load average3367 with over25000 tasks. No GPU result
was produced. The same isolated build is finishing on cssb-master2 with
MAX_JOBS2; preprocessed generic source completed in seconds there.

Source audit found a defect in the experiment itself: installed kNoProbe is
true for fast flags107/536, so the first P-dependent early probe was compiled
out. The prior note describes intended behavior, not emitted behavior. The
legacy basepprobeboth build is kept as such and MUST NOT be called a real
P-probe dependency experiment. New --active-probe variants basepprobebothv2
and basepprobeevenv2 enable only the bias probe on kFast; K/V probe policy and
generic/SAFE remain unchanged. Verify actual PTX/SASS probe/dependency
emission before GPU qualification. No installed changes.

P-probe builds are now terminal. Legacy basepprobeboth confirms ZERO PTX
probes,39ARRIVE/22DEPBAR; no GPU measurement. Corrected bothv2 emits16 probes
but C7512 and78ARRIVE/78DEPBAR; evenv2 emits8 probes but64B stack,512B spill
stores/576B loads and78ARRIVE/78DEPBAR. Both rejected before GPU. Generic0
and SAFE1024 machine words match installed9365 for allthree builds.
probability-probe-results.json contains the exact resources and codegen.

Offline current PTX register-usage screen has12 configs but only3 unique
machine-code groups: levels4/5/noexp equal installed9365 exactly; levels0-3
are identical3704 instructions; levels6-10 are identical3464 instructions
with different EX2/FMA placement. All clean/no spills; allretain39ARRIVE/
22DEPBAR. This is not performance evidence. Job14964 now compares serving,
reassembled-identical ru5 control, ru0 and ru6 with CUDA Driver graph-node
replacement. It preserves the actual core/block graph and kernel arguments,
replaces exactlyone fast node, checks parameter size/512threads/191488Bshared,
and compares independent full-output clones before/after balanced timing in
both directions. No installed package is modified.

14964 COMPLETED0 in2:14. CUDA graph replacement validated with1216-byte
matching argument, one hot node in five-node core/seven-node block graphs.
Full outputs BITWISE before and after24 balanced rounds, both directions.
Reassembled-identical ru5 core ratio1.000194/1.000210 validates the control;
ru0 core1.005191/1.004870 andru6 1.025604/1.027162 are slower. Block ratios
ru0 1.001228/1.006136,ru6 1.014044/1.015950. Both compiler alternatives
REJECTED. No serving changes. bench_ptxas_graph.py is now a working isolated
way to screen compatible single-kernel cubins without rebuilding host TUs;
production installation still requires a normal final-package build/gates.

A final native full-Q resource check clones current9365, changes only the
second query half's last QK operand from shared to cached BF16 registers,
and fences all Q registers. No scaled-Q/FP16/denominator changes. One hot
translation unit compiled to PTX;12 offline compiler settings are being
screened for a clean nonserialized cubin before any possible GPU comparison.

Native full-Q compiler screen completed: all12 settings retain C7512, no
spills,78ARRIVE/78DEPBAR. None qualifies for GPU execution. The P-probe
builder default now enables the actual fast probe; only the explicit
--legacy-probe-disabled flag recreates the flawed control. Original batch
script now selects v2 and checks emitted probes before any benchmark.

P_PROBE_CODEGEN_REPORT.md and HANDOFF updated through14964. Installed9365
manifest/source/binary hashes reverified. This continuation adds no serving
speedup; it fixes an invalid experiment, excludes the corrected dependencies
and current compiler controls, and validates direct cubin graph comparison.
All owned CPU builds and GPUjobs14947/14964 are terminal; no benchmark is
pending. SOL90 remains ACTIVE and unachieved. Do not mark complete or blocked.

Previous goal turn classified PROGRESS: invalid probe corrected, actual
compiler alternatives timed/rejected, standalone CUDA graph replacement
validated. Reverified installed9365 and14947/14964 terminal this turn.

New releaseorderbias/releaseorderp controls use the existing bias-empty
arrival before QK, with dependency on ALL16 loaded score words, independent
of LDS vectorization. The p control adds ALL8 packed P words to that same
arrival address. No extra full-barrier probe. Bootstrap and generic/SAFE
release policies remain unchanged. Both native hot PTX builds completed.
All12 P-dependent compiler controls stillC7512/78ARRIVE/78DEPBAR, no spills;
rejected before GPU. Bias-only control compiles clean/39ARRIVE/22DEPBAR;
initial actual-graph comparison remains pending.

New splitfullcurrent separates K-ready from V-ready using alreadyallocated
PipeK.full storage. Current independent producer warps both signal the same
PipeKV.full barrier; fast QK consequently waits for V as well. Candidate
K producer signals PipeK.full (init1), V signals PipeKV.full (fastinit1).
QK waits/probes useK-ready; PV0 bootstrap waitsV0, and each later firstPV of
a key tile waitsV with phase(pp for tp1,pp^1 for tp2). Both empty-barrier
ownership/release frontiers are unchanged. No extra shared bytes or changed
math/Params. Generic/SAFE retain combinedinit2. Hot PTX build is running.

14983 COMPLETED0, bothdirections/fulloutputs BITWISE before/after24balanced
rounds. earlybias core1.062891/1.064786, block1.040278/1.039618: rejected.
splitfull core1.001192/1.001952, block0.999962/0.997333: nearparity, no core
win; do not install. Identical control core0.997003/1.000327 quantifies some
noise. Logical first-use/phase coverage for keyranges1..8 is saved in
splitfull-schedule-proof.json; no full sanitizers justified at this deficit.

New m64n64r2s3p2qrf40 derives from the initially-verified M64/R2 fusedN40
prototype. NativeBF16, M64/N64/R2, three32-register score fragments, two
16-register P fragments,20-register fused output/den and8-register cachedQ.
Producer32 + two consumer224 WGs require an initial160-register CTA pool;
oneCTA due~140KiB shared. K/V have independent full AND empty barriers with
twoLN128stages each. Four16KiB bias slots retain paired full barriers and
8owningwarp empty arrivals. Seeding uses first32keys to match installed's
seed domain; native FMA/EX2/PV K16 order unchanged. No periodic rescale:
original prototype final denominator window still requests SAFE.

Three-score schedule: prologueQK0 retired, bias1 prefetched; stepk waits1
(QKk andPVk-3 retired), packsE(k-1), issuesQK(k+1), evaluatesE(k), commits
PV(k-1), prefetchesbias(k+2). Step0 uses zeroP1 dummyPV0 so wait1atstep1
actually retiresQK1; P1 not overwritten untilstep2 hasretired itsread. K
released afterQKk retirement, V afterPVk-3 retirement; finaldrain releases
remainingretiredtiles thenpacks/issues/drainsPV11.12static steps, no tail
phantomQKs. SAFE is original sequential native loop with separateK/V waits.
Build70582 is active on buildhost; compiler/resources/initialnumerics and
latency not yet verified. Not an installation candidate yet.

M64/N64 build completed clean168 initial registers, no spills/serialization;
168*384=64512 supplies required61440 role registers. SASS76HGMMA/25ARRIVE/
14DEPBAR/384EX2.14988 COMPLETED0: initial FP64RMS .000288830381 versus
.000288834855 passes, fullcore notbitwise (deltaRMS7.837e-6). Hot1259.6624us
versus791.7712us; core1304.531 versus860.457us, SAFE45.945us. REJECTED.
No broad numerical/sanitizer or installation claims. Existing prototype
only checks denominator window/missingseed for SAFE, not numerator finite;
that limitation would need fixing before any real qualification, but no
further numerical work is justified for this slower candidate.
One NCU profile is being collected to identify the limiting unit; this is
a diagnosis, not another performance candidate.

14989 NCU COMPLETED0: M64/N64/R2 hot1241.568us, SM/XU38.844488%,
tensor22.743146%,occupancy16.059331%,issue24.145006%, sharedLSU34.175567%.
DRAM608.470272MB similarinstalled607.496MB; L2traffic8.047587136GB versus
4.9584768GB. LongSB4.961616 perissue versusinstalled2.010489; these ratios
are not walltime fractions. Execution/memory units are underutilized.
m64n64-three-profile-summary.json records metrics with proper ms conversion.

Follow-up m64n64r3s2p2qrf40 restores three consumerWGs by using TWO scores
and current-chunk E->pack->PV, removing the32-register thirdscore. FullQ8
retained; threeconsumer160 +producer32 exactlyfundedby512*128. Independent
K/V full/empty, fourN64bias slots, first32-key seed, N40PV/den remain. Twelve
static chunks, no intermediate drain or phantomtailQK; everytopwait1 retires
QKk andPVk-2, so P[k%2] safe towrite andV(k-2) safetorelease. Finaldrain
retiresPV10/11 thenreleaseslastVtile. Numerator finite check also added before
output toclose the inherited prototype's gap. CPUbuild4190 isrunning; no
compiler/runtime/numerical claims yet. This is a new occupancy/ILP control,
not a change to installed9365.

14994 COMPLETED0: M64/N64/R3 passes compiler128/no spills/no serialization,
72HGMMA/24ARRIVE/14DEPBAR. Initial FP64RMS .000288830381 passes the sampled
case, but nonbitwise and hot962.8418vs789.0068us/core1075.115vs873.499us.
REJECTED. Restoring three consumers helps versus R2 but remains clearly
slower than installed. No broad numerical/sanitizer or installation claim.

New leanprod32current and leanprod24q168 directly issue the same 5D TMA
descriptor coordinates, with rolled bias and K/V loops and one selected
K/V descriptor/data/free-barrier base per producer warp. Q/K/V logical
coordinates are D,S,H,row,B; bias is 0,0,2*kc0+seq,qtile,BH. Original
full/empty counts, phases and owning-warp releases remain unchanged.
Generic/SAFE retain the original source. Control uses32+160+160+160; mixed
uses24+168+160+160, both512-thread full-WG register pools exactly65536.
Mixed first consumer caches all four Q fragments, remaining consumers keep
installed three. Control has compiled clean128 with2544 instructions versus
installed3464, same78HGMMA/39ARRIVE/22DEPBAR. Mixed CPU17623 is building.
No runtime or speed claim yet. This revisits the old producer-only spill
limit with a rewritten producer, rather than a register-cap-only change.

15018 COMPLETED0: genuine leanprod32current and leanprod24q168 bothfull
core/block BITWISE before/after24balancedrounds in both directions. Core
ratios1.011837/1.009889 and1.019909/1.017595 respectively: reject latency.
Actual emitted role values are32+3x160 and24+168+2x160; mixed rolled-bias
producer really does run at24/no spills. No serving gain.

IMPORTANT CORRECTION: initial unroll-bias b8 builds are INVALID experiments.
build_lean_producer_ptx reused variable old inside the unroll transformation,
then used that overwritten string as the outer producer replacement anchor.
The producer injection silently did not happen. Thus leanprod32currentb8
equal-installed machine words prove the NO-OP, not an equivalent rewritten
producer. leanprod24q168b8 retained actual producer32 while consumers grew
to168/160/160:66560 required versus65536 available. ptxas zero-spill summary
did not catch this runtime role-budget error.15033 timed out124 after4:03.
15045 CUDA-GDB reproduces the hang: onSM0 the middle consumer WG loops in
USETMAXREG160 while the other two wait at prologue namedbarrier11; producers
wait on bias/K/V empty queues. No compiler bug or TMA phase bug is implied.
lean-producer-register-audit.json records actual emitted values and PCs.

Builder now uses a separate checked producer_anchor and asserts emitted
source features. Corrected unroll variants use unique b8v2 names, CPU53077
and1227 active. GPU graph harness now checks actual SASS DEC/ALLOC values
and the declared role multiplicity/pool BEFORE loading a candidate module.
The offline compiler screen also records emitted register reconfiguration.
Historical bad b8 case now declares24 and will fail the strengthened gate.

Current PV-before-E experiments (three-Q and full-Q, before/after biasrelease)
all12 compiler settings each C7512,78ARRIVE/78DEPBAR,no spills. All four
rejected before GPU. Original producer, waits, commits and arithmetic remain
in those isolated sources; the changed scheduling did not repair allocation.

Corrected b8v2 producer injection is verified in source and actual PTX/SASS.
Control2744 instructions and mixed4864, clean128/no spills/no serialization;
actual roles32+3x160 and24+168+2x160, each65536.15052 COMPLETED0 in26s.
Both candidates complete core/block graphs with full BITWISE outputs before
and after24 balanced rounds, both directions. lean32b8v2 core ratios
1.006770535/1.007490237; lean24q168b8v2 1.018478303/1.018664498. Both rejected.
This valid follow-up fixes the generator/reg-budget error but provides no
serving speedup. No broad numerical/sanitizer work justified for slower builds.

register_role_gate.py is used by the graph harness before CUDA module load.
It compares emitted SASS DEC/ALLOC values with declared producer/consumer
roles and verifies multiplicity-weighted pool against entry128*512/65536.
Four regression cases pass using actual installed/valid-mixed/deadlock SASS,
including rejection of both wrong intended roles and actual overflowing pool.
All owned builds and GPU jobs through15052 are terminal;15033/15045 are
recorded expected-timeout failure evidence, never benchmark successes.
producer-register-results.json and PRODUCER_REGISTER_REPORT.md summarize.
Installed9365 all12 manifest hashes reverified unchanged; last NCU remains
L768 61.348992%,L1024 65.249349%. SOL90 active and unachieved.

Next continuation classifies the previous turn as PROGRESS: valid negative
performance evidence, reproduced/diagnosed generator-induced allocation
deadlock, and a tested emitted-role gate change subsequent GPU admission.
Installed9365 manifest reverified; all previous jobs terminal.

New currentphasetrace is a DIAGNOSTIC of startup versus steady stream cost.
Three fixed L768 CTAs (1,1,0),(2,128,1),(5,255,3), each consumer WG leader,
record coarse clock64 points in a private1440-byte module global. No params
changes; no per-chunk body instrumentation. Entry/register allocation/Q
ready/bootstrap/period drains/finalPV/epilogue mark17 timestamps plusSMID.
Differences are taken within each CTA only. Compiler hot128/no spills/no
serialization,78HGMMA/39ARRIVE/22DEPBAR/288EX2 unchanged;4064 instructions
versus installed3464. Graph full-output equality and instrumentation cost
must pass/be recorded before interpreting sampled cycle fractions.


15063 COMPLETED0: coarse phase trace is full-core/block BITWISE before and
 after24 balanced rounds, both directions. Instrumentation core overhead is
1.0510670984 starting/1.0488359428 ending. Same-CTA clocks of later sampled
CTAs show body72.9-74.0%, startup17.5-18.6%, drains2.8-2.9%, tail5.8-6.0%;
register allocation alone0.8%. These are instrumented fractions with ~5%
overhead, NOT an exact unmodified latency breakdown. They motivate work on
steady stream latency rather than startup-only changes. Aggregated evidence:
current-phase-trace-summary.json.

New N16 family uses native BF16/fullQ, fused N40 PV/den, independent per-row
K/V ready/free barriers, paired bias-ready barriers and four score/two P
buffers. First32-key seed and numerator/denominator SAFE checks retained.
Fully static L768 sequence: QK(k+2), E(k), PV(k-1), bias(k+3), no phantom
final QK; last wait1 accounts for the shortened tail. B1 contiguous prototype.
Initial M64/R5 (32+5x88, initial80) and M128/R4 (32+4x112, initial96)
fail C7512/spill gates. No GPU launch. N16-specific gate now validates emitted
SASS roles and multiplicity against actual initial pool for both hot/SAFE.

M64/R4 (32+4x112, initial96) and M128/R3 (32+3x160, initial128) compile
without serialization or spills, with correct emitted roles/pools.15075 and
15076 COMPLETED0; initial full core AND block outputs exactly equal serving,
and first-row FP64 .00028883485479432215 equals baseline. M64 hot914.3968
versus789.3254us/core1042.6723versus886.1334us; M128 hot853.9268versus
795.0976us/core977.7225versus873.3833us. Reject both on performance. These
are initial three-round comparisons; no broad numerical/sanitizer claims.

M64 prefenced variant removes44 source PV fences, but ptxas reinserts all44
(C7519); actual hot machine words are IDENTICAL (9888 words/96 ARRIVE/49
DEPBAR) to its already-measured explicit-fence counterpart. Skip duplicate
GPU benchmark. M128 prefenced and M128/R4 two-score builds pending. The
latter retains four scores only for bootstrap, copies remaining QK2/3 into
two live stream slots, then issues PV(k) before QK(k+2). wait1 leaves
QK(k+1) outstanding across E(k); two P buffers and delayed V release remain.
Installed9365 all12 manifest entries reverified unchanged.


M128 prefenced also identical hot machine words (12592 words/192ARRIVE,
92 reinserted C7519). n16-prefenced-equivalence.json records both; no GPU.
Two-score M128/R4 first build has no serialization but104B stack/220B spill
stores/296B spill loads, rejected. Fixed K/V base descriptors with constant
stage/column offsets remove ALL hot/SAFE spills and serialization at actual
32+4x112/initial96,61440 registers. Name m128n16r4s2p2qrf40fd, SHAa4d21494,
288HGMMA/192ARRIVE/96DEPBAR/768EX2.15094 completed successfully, initial full
core/block BITWISE and FP64 RMS exactlybaseline. Hot859.5294vs787.5728us,
core990.2368vs881.5667us; reject performance. Four consumers alone do not
make this schedule faster. NCU requested to distinguish execution bottlenecks.
Four-score fixed-descriptor variants with12/8 cached Q registers now building
(m128n16r4s4p2qrf40fdq3/fdq2); other Q fragments use native SS WGMMA. This
trades Q register cache for longer lookahead within the same4-consumer pool.


User requested candid status review, not additional experiment submission.
15097 COMPLETED0: fixed-descriptor two-score R4 profile864.480us, SM56.017004%,
tensor31.346303%, occupancy28.238085%, L2 142639423 sectors=4.564461536GB.
Despite higher occupancy than installed22.296%, both speed and SM utilization
are worse. No GPU jobs remain from this continuation; CPU17222/31265 for
four-score Q3/Q2 controls were still running at review time. No gain claimed.
Installed9365 all12 manifest hashes reverified unchanged. Recent SOL90 search
has yielded zero additional installed performance gain. Previous qualified
improvements remain; there is no proof of a hardware limit or of SOL90 being
reachable with this design. Continuing nearby variants has low demonstrated
return. PHASE_N16_REPORT.md records these results and limits.


Kernel-inventory request: both remaining four-score fdq3/fdq2 CPU builds now
terminal0, but admission rejects both for C7512. Q3 also16B stack/16B spill
stores/768B loads; Q2 no spills. Actual32+4x112/initial96 pools valid. No GPU
launch and no owned active jobs/builds remain. KERNEL_STATUS.md and numerical
kernel-inventory.json distinguish all installed stages and shape branches,
CUPTI component times, separate NCU metrics, and historical improvements.
