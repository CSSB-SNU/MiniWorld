# TriangleAttention kernel optimisation — handoff

> **Wide HBM follow-up qualified19517/19537,2026-09-27:** [Results and explicit entry](large_d_20260926/followup_20260927/REPORT.md). Compact grouped dBias partials are stored BF16 with FP32 local/final reduction; buffer bytes halve. Width256 preserves prior TMA/WGMMA projection fusion; width512 uses cuBLAS beta=1 output accumulation plus a specialized native BF16x2 bias epilogue, removing separate projection dX/add intermediates. Across L384/768 and both directions, new BWD is1.27–1.36x/1.25–1.27x over original Triton for width256/512; F+B is1.21–1.28x/1.18–1.22x. Over the previous wide candidate, width512 BWD time falls9–12% and F+B7–9%; width256 gains are smaller. All paired BWD/F+B intervals are positive. FP64, large-gradient range, full PyTorch gradients, dropout/frozen/SGD, changed-state graph and all three sanitizer gates pass. Read the report for rounding differences; full backward is not bitwise with prior. `selected_next.py` and its `CANDIDATE.json` are the new explicit qualified entry; previous selection remains frozen for comparison. No serving edits or SOL90 claim. All owned jobs are terminal.

> **Wide256/512 qualified,2026-09-27:** [Results, selection and evidence](large_d_20260926/REPORT.md). D denotes pair/total-QKV width256/512 with H4, hence head64/128. Native CUDA/TMA FWD/dQ/grouped-dKDV pass independent FP64, full PyTorch gradients, changed-input/weight/dy/mask CUDA Graphs, dropout/frozen/SGD fixtures and all three sanitizers. Select attention plus projection-dgrad fusion at256; select attention-only CUDA at512 because the wider projection fusion regresses. Full F+B beats the actual original Triton engine by1.19–1.24x at256 and1.08–1.12x at512 across L384/768 and both directions. `large_d_20260926/selected.py` is the explicit qualified entry; `CANDIDATE.json` pins artifacts. No serving change, no wide SOL90 claim; all48 installed training18246 hashes remain unchanged. Jobs19396/19397/19405/19416/19417/19418/19426/19427/19429/19430/19431/19436/19445 are complete. The remaining large costs are grouped dK/dV, dQ, projection GEMMs/additions and bias reduction.

> **C128 campaign closed; wide-channel continuation,2026-09-26:** The user accepted the corrected original-Triton comparison and requested large D256–512 work. C128 installed training18246 and selected inference19157 remain the closed checkpoints; `reuse_qdo` remains an explicit qualified experiment. All comparison work is terminal. New work is isolated in [large_d_20260926](large_d_20260926/DESIGN.md), initially interpreting D as total pair/QKV channels256/512 with four heads (head64/128), pending clarification. Measure actual wide dispatch and full FWD/BWD/F+B before selecting native CUDA/TMA fusion candidates.

> **Baseline identity corrected19328/19337,2026-09-26:** The user's **existing engine is the original Triton implementation**. The previous report incorrectly used that label for our already optimized CUDA checkpoint18246; its claim about the original engine is withdrawn. [Corrected original-Triton comparison](bwd/baseline_comparison_20260926/TRITON_COMPARISON.md) restores the git module/core and verifies actual Triton FWD/dQ/dK/dV dispatch. Current CUDA is1.80–1.86x faster in complete BWD and1.63–1.71x in F+B across384/768/1024 and both directions. These are cumulative gains over original Triton, distinct from the small last dQ increment. PyTorch and Anthropic support status are included. All outputs/gradients pass numerical checks, all48 installed files remain unchanged, and no serving change is made. The1024 three-arm rerun19337 is complete;19328's completed384/768 cells remain valid after its later four-arm1024 OOM. Every owned comparison job is terminal.

> Comparison follow-up19317 is complete: L76848x20 repeat retains every output/gradient and actual graph/dQ checks; small engine/candidate BWD differences remain order-sensitive and its F+B intervals include zero. Earlier19315 stopped on an incomplete post-timing profiler trace and is retained only as partial evidence. The repeat changes profiler replay count only; timing/fixtures remain identical. All comparison jobs are terminal; no new default selection is made.

> **Training baseline comparison19305,2026-09-26:** [PyTorch / Anthropic / installed engine / candidate](bwd/baseline_comparison_20260926/README.md). Same-job complete BWD and F+B at384/768/1024 in both directions, all input/parameter gradients, with actual CUDA Graph and dQ dispatch evidence. Against the existing dense PyTorch einsum/softmax path, `reuse_qdo` is3.60–4.25x faster in BWD and4.35–5.58x in F+B. The pristine Anthropic optimized rows have no backward; stock-library fallback is not substituted. Candidate and checkpoint18246 outputs/gradients are bitwise equal, and all48 installed files remain unchanged. This three-arm timing does not establish a consistent small gain over the installed engine: L768 order-dependent variation is larger than the proposed gain; independent repeat19315 is recorded in the comparison report. The candidate stays explicit and uninstalled.

> **Bounded backward follow-up19266/19268/19277,2026-09-26:** [Results and experiment entry](bwd/reuse_20260926/README.md). Three native CUDA controls tested; `reuse_qdo` retains both dQ operands Q/dO in registers,122 registers and zero spills, with existing TMA staging/barriers. Native dQ takes1.79%/1.83% less time at768/1024. Final64x40 paired complete backward reduces time0.10–0.15% at384,0.28–0.40% at768 and0.35–0.66% at1024; all backward95% intervals exceed zero. Only1024 has clear F+B gains in both directions,0.22–0.36%; shorter-length F+B is mostly inconclusive. Q-only and dK/dV softmax/dP overlap give no consistent whole-workload benefit. Native/full-module bitwise, FP64, dropout/SGD/fullgraph/frozen parameters, changed-input/weight/dy/mask graphs and all three sanitizers at64/384/768/1024 pass. This is a qualified explicit experiment, not a production promotion; all48 training18246 files and the inference19157 selection remain unchanged. All nine owned jobs are complete. No new SOL measurement or SOL90 claim.

> **Inference Q reuse qualified19134/19157,2026-09-26:** [Results and sixteen controls](fwd_inference/CORE_FOLLOWUP_RESULTS.md). Native CUDA `qfull4v2` keeps the whole query tile in registers across the key loop, with single-head CTAs, four query warpgroups, two TMA bias slots and128 registers without spills. The experimental `candidate.load()` changes only L1024; L384 `h4kv_local1`, L768 `hot6t`, front8 and outproj_c1 remain. Actual-default64x40 paired complete FWD reduces L1024 time **2.22% starting /2.23% ending** versus18962, with both95% intervals above zero; a separate native-boundary comparison improves3.67%. Current complete FWD takes **32.00–32.25% /6.73–8.47% /5.87–7.43% less time** than the faster original Anthropic residual form at384/768/1024. L768 Q-reuse gain is inconclusive, so its route stays unchanged. Native bitwise/FP64/retry/changed graphs, memcheck/racecheck/synccheck at128/768/1024 and actual-default full-module/fullgraph checks pass. All48 training18246 files match; no production or training change. All owned jobs are terminal. Retain old1024 with `load('hot4t', out_artifact='outproj_c1')`. Four-head attention CTA work remains closed; SOL90 remains unmet.

> **Inference output fusion qualified18946/18962,2026-09-26:** [Results](fwd_inference/PIPELINE_OUTPUT_RESULTS.md). Native CUDA `outproj_c1` combines output projection, BF16-rounded residual and ending-orientation TMA store. The explicit experimental `candidate.load()` entry uses it with the existing L384 `h4kv_local1`, L768 `hot6t`, L1024 `hot4t` cores. Actual-default64x40 paired complete FWD reduces time **7.09–8.85% /5.82–5.83% /4.32–4.83%** at384/768/1024 relative to the previous selection, and **32.12–32.45% /6.47–8.63% /2.98–5.03%** relative to the faster original Anthropic residual form. NCU output-boundary HBM drops about32–39%. Native FP64/graph, all three sanitizers at128/384/768/1024, full-module FP64/changed graphs/fullgraph and actual kernel attribution pass. No production or training change; `load(out_artifact=None)` preserves the previous selection. Single-head attention pipeline controls remain unselected: smaller tiles and register-lifetime variants fail to beat existing cores and emit WGMMA serialization diagnostics. Four-head attention CTA work stays closed. No SOL90 claim.

> **Bottleneck diagnosis18909, 2026-09-26:** the user explicitly closed four-head **attention CTA** work; the separate selected joint KV projector remains valid. [Analysis and next algorithms](fwd_inference/BOTTLENECK_ANALYSIS.md). Fresh selected-core NCU on node02 shows768/1024 HBM peak10.5%/7.0%, tensor-pipe activity27.9%/28.3%, special-function pipe43.2%/45.1%, and DRAM0.422/0.620GB. Long-length core is not HBM-bandwidth-saturated. Main next hypothesis is genuine QK(t+1) / exp(t) / PV overlap, with register-budget redesign (e.g. two32-key score fragments); previous async4/6 only delayed PV waits and are not this pipeline. A separate output-projection+residual epilogue could remove75.5/302/537MB of real temporary traffic at384/768/1024; modeled savings are not measured gains. Bias reuse is along outer rows, and prior corrected multicast already lost. No kernel/dispatch changes; selected paths preserved. Profiling18909 completed. [Raw metrics](fwd_inference/bottleneck-analysis.json).

> **Head-last bias and four-head attention CTA, experiments18806–18831 rejected, 2026-09-26:** [Results](fwd_inference/CTA_HEADS_RESULTS.md). Implemented direct BF16 `[1,L,L,4]` LN+bias output plus one CTA owning four heads and sharing query Z. Six CUDA/TMA controls cover staging, direct vector/scalar packed reads, in-place shared-memory rearrangement, planar bias, and independently pipelined heads. Best head-last complete FWD is **37–48% slower** than current per-length selection; planar independent-head CTA is still **13–15% slower**. Existing front8 already writes planar bias directly, so no separate global transpose was removed. NCU768: packed core1.914ms versus current streaming core1.104ms at nearly identical0.605GB DRAM; shared-read stalls increase. All180 native fixtures and changed graphs pass; primary/repack candidates pass additional retry graphs, three sanitizers at128 and full-module/fullgraph checks at384. No full-length qualification or promotion claim. Producer-only head-last gains at768/1024 are only0.9–1.9%. Current selection and48 training18246 files remain unchanged. All six owned jobs completed; unrelated jobs preserved. [Design](fwd_inference/CTA_HEADS_DESIGN.md), [JSON](fwd_inference/cta-heads-results.json). SOL90 remains unmet.

> **K/V placement and four-head reuse, qualified18721/18722 and selected-entry18745, 2026-09-26:** [Current inference results](fwd_inference/HEAD4_KV_RESULTS.md). `candidate.load()` now selects L384 `h4kv_local1` (joint four-head KV projection plus row-local tiled attention), L768 `hot6t` (unchanged), and L1024 `hot4t` (four query warpgroups, two bias stages, zero spills). Full FWD improves **8.06–8.10% at384** and **1.35–1.63% at1024** versus prior hot6t. Against original Anthropic: **25.97–26.88% faster at384**, **1.23–2.51% faster at768**, but **0.56–2.35% slower medians at1024**. Fourteen controls include materialized resident KV, one/two/four streaming warpgroups, CTA order, adjacent-head projection, and LN/bias/KV front fusion. Fair head-width ablation shows KV projection3.71–7.29% faster; the earlier2.15–2.34x comparison included a poor head-major grid. Row-local query scheduling cuts768 streaming-core HBM3.932→0.605GB, though extra KV projection still loses at long lengths. FP64/retry/graph, selected-shape memcheck/racecheck/synccheck and actual-default fullgraph/full-module checks pass;64x40 paired graphs, both directions, node02. Production dispatch and48 training18246 files unchanged. All owned experiments terminal; unrelated jobs preserved. No SOL90 or unrestricted all-masked parity claim. [Design](fwd_inference/HEAD4_KV_DESIGN.md), [JSON](fwd_inference/head4-results.json).

> **QKV inference algorithm advanced, qualified18586/18598, 2026-09-26:** [Current results](fwd_inference/QKV_ALGORITHM_RESULTS.md) select `hot6t + front8` in the explicit `candidate.load()` experiment entry. QKV stay on chip; the normal path computes both P*V and P*1 on Tensor Cores and retries unsafe tiles with stable softmax using resident QKV. Versus our previous resident6, complete FWD time falls **6.37–6.57% at L768**, **8.46–8.50% at L1024**, with L384 essentially unchanged. Versus original Anthropic, **L384 is18.78–20.08% faster, L768 is1.74–3.30% faster, but L1024 remains1.47–3.45% slower** (64x40 paired graphs, both directions, node02/normal_h100). Thirty native FP64 fixtures, retry counters/changed-path graphs, complete module FP64/fullgraph and all three sanitizers pass. Initial18586 timed out on combined L1024 racecheck;18598 reran its three fixtures independently and completed qualification. Twelve controls include rejected query splits, joint N128 projection, scalar/peak softmax, PV overlap and TMA multicast; initial cluster scratch race was fixed but slower. NCU shows core speedup despite slightly more HBM traffic; register spills remain. [Algorithm](fwd_inference/QKV_ALGORITHM_DESIGN.md). Original all-masked semantic difference remains; production dispatch and48 training18246 files are unchanged. All owned jobs are terminal. SOL90 remains unclaimed.

> **Anthropic inference comparison corrected, job18528, 2026-09-26:** [Primary comparison](fwd_inference/ANTHROPIC_COMPARISON.md) measures original Anthropic fused block versus our `resident6 + front8`, complete input-preserving inference FWD, same inputs/weights, both directions, 64x40 paired graphs on node02 / normal_h100. Our time is **18.51–19.84% lower at L384**, but **3.71–5.40% higher at L768** and **10.84–13.55% higher at L1024**. Upstream default dispatch is flash_triattn at384 and native M1 at768/1024; profiles and original-source/build hashes are verified. Both upstream residual forms are measured and the faster one selected per cell. LN affine fixtures match upstream BF16 rounding. All-masked semantics differ (ours zero update, upstream uniform-mean-V attention); no unrestricted parity claim. Earlier18488 percentages below compare our internal installed eval baseline, not Anthropic. No production changes or inference SOL90 claim.

> **Inference-only FWD checkpoint18488 complete, 2026-09-26:** the user explicitly removed the backward-save requirement. [Inference results](fwd_inference/RESULTS.md) select native CUDA/TMA `resident6 + front8`: no global Q/K/V/gate/LSE or LN-statistic saves; LN+bias and residual/transpose are fused. Actual complete eval/inference_mode FWD is **3.71–6.58% faster starting, 17.14–23.55% faster ending**, acrossL384/768/1024, with64x40 paired graphs and every95% speedup CI above1. FP64, mask edge cases, changed-input graphs, fullgraph and all three sanitizers pass; fresh actual-dispatch traces18489 confirm the selected kernels. [Implementation and rejected controls](fwd_inference/README.md). The explicit inference experiment entry is ready; default dispatch and all48 training18246 files are unchanged. All owned jobs are terminal; no inference SOL90 claim.

> **Full Q/K/V/gate fusion installed at L1024, 2026-09-26:** promotion18246, `qkv_compact_retire6`. [Current results](fwd_training/QKV_ALIAS_RESULTS.md) compare actual installed full FWD/BWD/F+B with frozen18006. Both directions: FWD2.52–2.83% and F+B1.09–1.13% time reduction, independently repeated with64x40 paired graphs. Projection Z/weights share storage with attention Q/bias, enabling four projection warpgroups at1024. L384/L768 retain18006 QG. All outputs/gradients bitwise; FP64, cold compile, graph, AMP, three sanitizers includingL1024 and fresh-installed default/opt-out/fallback checks pass. Existing backward binaries unchanged. Ten new full-QKV controls are recorded; SOL90 remains unmet.

> **Full Q/K/V/gate fusion follow-up, 2026-09-26:** [Results and all twelve controls](fwd_training/QKV_STREAM_RESULTS.md). Native CUDA/TMA producer/consumer streaming and compact resident designs were implemented and measured against installed18006. Final `qkv_compact_remat` removes all register spills and reduces L768 projection+attention HBM reads66.1% (total traffic32.3%), but full FWD is still1.37–5.91% slower acrossL384/768/1024 and both directions. Native fixtures, full input/parameter gradients, FP64, changed-input graph and all three sanitizers pass. No candidate is promoted; all44 installed files still match18006. Final jobs18155/18159/18160/18161/18162 are complete. SOL90 remains unmet.

> **Q + gate projection + attention installed, 2026-09-26:** promotion18006, `qg_scoped`. [Current results](fwd_training/QG_FUSION_RESULTS.md) compare actual installed full FWD/BWD/F+B with frozen17774 atL384/768/1024, both directions. All output/gradients bitwise; FP64, graph, AMP and three sanitizers pass. Full-QKV five-warpgroup control also measured. Existing backward binaries unchanged. SOL90 remains unmet.

> **Q projection + attention installed, 2026-09-26:** [Current results](fwd_training/README.md) records promotion17774 (`q_only_head4`) against frozen17628. Seven CUDA fusion candidates were tested. Full QKV residency lost on L768/1024; Q-only fusion improves full FWD and F+B at all three lengths and both directions, with every output/gradient bitwise equal. Native CUDA/TMA, four adjacent-head CTAs, shared scratch reuse; all backward binaries remain unchanged. FP64, graph, three sanitizers, cold installed dispatch and AMP gates pass. [All fusion results](fwd_training/QKV_FUSION_RESULTS.md). SOL90 remains unmet.

> **CTA head reuse, 2026-09-25:** [Installed CUDA results](fwd_training/README.md) records `cooperative_head2`, promotion17628, against frozen17345. Adjacent-head CTA ordering halves core L768 HBM reads (910.76 ->457.77MB) with small, independently repeated full FWD/F+B gains. Arithmetic, 80-register/six-CTA resources and backward remain unchanged. FP64/all-gradients/graph/three-sanitizer/installed/AMP gates pass. [Producer/cluster experiments](fwd_training/PIPELINE_FOLLOWUP.md) added no selected gain. [QKV-attention fusion design](fwd_training/QKV_ATTENTION_FUSION.md) motivated the later17774 experiments above. SOL90 remains unmet.

> **Further training FWD improvement, 2026-09-25:** [Current CUDA results](fwd_training/README.md) records installed `cooperative_q1_s2`, promotion17345. One score fragment and two TMA stages permit six resident CTAs/SM, with 80 registers and zero spills. Additional full FWD and F+B gains are measured against frozen CUDA17310, with every output/gradient bitwise equal. FP64/graph/sanitizer/fresh-installed/BF16 AMP gates pass. Backward17211 and the dispatcher are unchanged. L768 NCU: 0.977600ms, SM73.77%, L277.58%; SOL90 remains unmet. Full evidence and all rejected candidates are recorded in the report.

> **Training FWD reopened and installed, 2026-09-25:** The user's new request reopened forward development. [CUDA training-forward results](fwd_training/README.md) records installed `cooperative_q2`, promotion17310: actual full FWD is15.5–21.8% faster and F+B6.2–7.1% faster than the preceding training Triton path, across L384/768/1024 and both directions. Native CUDA/TMA now produces projection-layout O plus FP32 base-2 LSE. FP64/all-gradient/graph/sanitizer/installed-dispatch gates and BF16 AMP17316 pass. Existing backward binaries remain checkpoint17211. L768 NCU:1.073536ms, L271.33%, SM63.03%; SOL90 remains unmet. The older standalone forward closure below is historical and does not prohibit this newly authorized training-forward work.

> **node02 continuation complete, 2026-09-25:** [Direct start-to-finish comparison](bwd/core90/NODE02_CONTINUATION.md) measures another7.05–7.90% full-backward reduction and3.89–4.91% F+B reduction versus checkpoint17069, with every gradient bitwise equal. Installed17211 uses native CUDA/TMA async R8 bias reduction with four Q/dO prefetch stages. All13 owned node02/normal_h100 jobs are complete; no queued work remains. SOL90 remains unmet.


> **Current backward installation, 2026-09-25:** [Installed report](bwd/core90/README.md). Qualified `rs8_async_q4` dK/dV + bias and `rs_softmax_overlap` dQ pass installed-path job17211, including all input/parameter gradients. Complete backward improves21.92–23.23% and F+B14.63–16.00% versus job16663. Attribution17211 accounts for every16/17 backward kernels. SOL90 remains unmet. All new experiments use `node02 / normal_h100`.

> **Previous backward update, 2026-09-23 — attribution priorities:** [bwd/priority_pass/README.md](bwd/priority_pass/README.md) records installed `ldmatrix_bias` dK/dV and native CUDA/TMA shared-input Q/K/V/gate weight gradients. Final installed job16663 gives another **4.09–5.30% whole-backward reduction** over the immediately preceding installation16580; F+B improves1.38–3.60%. L768 four-weight-gradient HBM traffic falls44.53%, and its main kernel reaches HBM SOL85.27%. Full backward launches16/17 kernels instead of22/23. dQ experiments were slower, so installed `vector_bias` dQ is retained. All gradients, FP64/cancellation, three sanitizers, fullgraph/cold compile, partial/frozen parameters, opt-outs and AMP pass. [Current timing attribution](bwd/priority_pass/ATTRIBUTION.md) is job16664. dK/dV and dQ remain below80; no full-backward SOL80/90 claim. Forward remains closed.

> **Previous backward update, 2026-09-23 — below80:** [bwd/below80/README.md](bwd/below80/README.md) records installed CUDA `vector_bias` dK/dV and dQ plus `warp_packed` projection/LN/residual. Actual installed job16580 passes all manifests, cold eager/compiled backward, AMP and10 opt-out checks. Additional whole-backward reduction versus the immediately preceding installation: L384 8.85–10.10%, L768 8.20–8.32%, L1024 7.17–7.51%; F+B also improves. Projection/LN/residual reaches **HBM SOL84.94% starting /82.84% ending**, so tuning stops there under the user's80% cutoff. dK/dV and dQ are faster but remain below80 (SM53.02% /58.09%). Gate and the final bias reducer are unchanged. No whole-backward SOL80/90 claim; forward optimization remains closed.

> **Backward fusion update, 2026-09-23:** [bwd/bias_fusion/README.md](bwd/bias_fusion/README.md) records the selected native CUDA/TMA dK/dV + grouped dBias fusion, with one producer and four consumer warpgroups per CTA. R4/FP32 halves bias scratch; final NCU job16104 measures32.7% less L768 whole-core DRAM traffic. Installed job16112 gives0.23–0.32% additional full-module backward gain at L768 and2.74–2.99% at L1024 over the already-installed projection fusion. L384 regresses and retains the previous path. Independent FP64/mask/dropout/optimizer/fullgraph/sanitizer checks and installed dispatch/BF16-autocast checks pass; evidence is recorded in that report. dQ and delta remain the prior Triton kernels. Forward SOL90 remains closed.

> **User direction, 2026-09-23:** design backward fusion before further kernel tuning; prioritize eliminating HBM intermediate writes/reads. [bwd/FUSION_PLAN.md](bwd/FUSION_PLAN.md) records the dependency/traffic analysis: local row-group dBias reduction, projection+LN+residual, gate+delta/dropout, then weight-gradient sharing. Priority1 has since been implemented and measured in the fusion update above; priorities2–4 remain proposals.

> **Backward update, 2026-09-22:** [bwd/README.md](bwd/README.md) records the installed H100 CUDA/TMA projection input-gradient fusion in the actual engine training module. Whole-module backward time falls about9–10% at L384,6–7% at L768 and5% at L1024; forward+backward also improves. Both directions, all gradients, FP64 references, dropout, optimizer updates, fullgraph compilation and three sanitizers are covered. That projection-only update left the attention core/bias buffer unchanged; the later fusion update above replaces dK/dV and halves bias scratch at768/1024. This is separate from the closed forward SOL90 search below.

> **User disposition:** core SM SOL90 search is closed as unsuccessful; do not continue the optimization search without a new user request. Retain the qualified installed core. CUDA/TMA surrounds, L384 producer/consumer dispatch, broadcast-mask staging and SAFE barrier fixes are complete within the documented H100 forward support scope. [KERNEL_STATUS.md](KERNEL_STATUS.md) records the final workstream dispositions and their limits.

> **Kernel inventory through15097:** [KERNEL_STATUS.md](KERNEL_STATUS.md) lists all default execution stages, L384/768/1024 timings, dispatch branches, CTA arrangements, installed improvements, utilization definitions and rejected experiment families. Recent search adds no installed gain. Three N16 candidates pass initial full-output bitwise checks but are slower; the remaining Q3/Q2 builds both serialize and are not launched. All owned builds/jobs are terminal. Installed9365 remains unchanged; SOL90 is unachieved.

> **Experiments through15052:** no new serving gain. Native TMA producer rewrites fit an actual24/168/160/160 register allocation, but corrected paired core comparisons are0.7–0.8%/1.9% slower. M64/N64 controls are also slower; PV-before-E controls serialize. A generator error in the first unrolled-bias build caused an overflowing register pool and a reproducible deadlock; it is fixed, and the graph harness now checks emitted SASS role allocations before launch. [PRODUCER_REGISTER_REPORT.md](PRODUCER_REGISTER_REPORT.md) records valid results, invalid experiments and four passing gate regression cases. Installed9365 and SOL61.349%/65.249% are unchanged; SOL90 remains unachieved.

> **Experiments through14964:** corrected P-to-bias probe dependencies fail compiler/resource gates; two distinct compiler scheduling alternatives are0.5% and2.6–2.7% slower in paired L768 core measurements, with full bitwise outputs in both directions. Native full-Q retains serialization under all12 compiler settings. No new serving gain; installed9365 and SOL61.349%/65.249% remain unchanged. [P_PROBE_CODEGEN_REPORT.md](P_PROBE_CODEGEN_REPORT.md) records the corrected experiment, measurements and standalone-cubin graph harness.

> **Experiments through14928:** nine four-consumer CUDA/TMA resident K/V candidates produced no serving gain. Six fail the compiler serialization gate; three pass initial full core/block bitwise checks but take1003–1608us versus788–796us. The K-resident/V-streaming and shared-P controls are also rejected. [RESIDENT_FOUR_REPORT.md](RESIDENT_FOUR_REPORT.md) records resources, barriers and results. Installed9365 is unchanged; SOL90 remains unachieved.

> **Latest installation, job 14862:** L1024 core is 4.163%/3.971% faster in paired starting/ending measurements. Installed NCU: L768 795.904us / SM61.349%; L1024 1751.712us / SM65.249%. L768 machine words are unchanged from 4be. [Q1024_REPORT.md](Q1024_REPORT.md) records the package, 82 bitwise cases, full-shape sanitizers and serving verification. **SOL90 remains unachieved.**

> **Previous core follow-up14836:** [CORE_SOL90_REPORT.md](CORE_SOL90_REPORT.md) records the installed exact L768 specialization, independent K/V TMA producers, corrected barriers, N40 PV/row-sum fusion, constant descriptors, and compact ones storage. Job14836 installed SHA4be5b7cd: three of four Q fragments now stay in12 registers. Job14833's72-round balanced comparison with independently cloned outputs gives another0.458%/0.453% core reduction, starting/ending, over976. L1024 meets nonregression. Final package passes62 bitwise cases, three full-shape sanitizers, installed default/opt-out checks and17/17 shipped tests. That installation measured784.896us, SM61.347%; SOL90 remains unachieved. The installed package is the rebuild source of truth; older prototype installers predate these changes. [q-three-results.json](core_sol90/q-three-results.json) holds the evidence.

> **Through14836:** scalar/full-Q denominator changes spilled; two clean SM80 denominator/full-Q candidates preserved initial full-output bitwise equality but took831–833us versus792us, and were rejected. Captured-output checks in pair.py were corrected to clone each graph result before the next replay overwrites its shared buffer. Job14833/14834 reran both lengths/directions with distinct snapshots. Earlier paired timings remain timing evidence; old captured references alone do not prove equality.

> **Experiments through14777:** shared-P/full-Q, mixed register/shared P, paired cooperative WGs, and M64 four-score/two-CTA candidates added no qualified speedup. Nine measured controls passed initial full-output equality but were slower; all remain isolated. [core_sol90/STATUS.md](core_sol90/STATUS.md) records compiler-stage fixes and exclusions, and [shared-probability-results.json](core_sol90/shared-probability-results.json) contains the measurements. Installed SHA97634810 and its existing qualification are unchanged.

> **Experiments through14802:** half2 HFMA2 exponentiation passed its initial numerical model but was about2.6x slower in a register microprobe; no attention integration. P-order dependency controls retained WGMMA serialization. A544-thread compact-producer design had an invalid naive register budget: ptxas capped it at96 and spilled. All were rejected; [half2-and-fence-results.json](core_sol90/half2-and-fence-results.json) records the evidence. No new serving change; SOL90 remains unachieved.

> **Main core now optimized:** [CORE_KERNEL_REPORT.md](CORE_KERNEL_REPORT.md) records the installed broadcast-mask CUDA attention specialization: about 3.8% less calculation time at L768 and 2.9% at L1024, bitwise equal. The earlier generic core remains for other masks and opt-out.

> **Core follow-up:** [CORE_REPORT.md](CORE_REPORT.md) records the current M1 bottleneck and the installed broadcast-mask preparation optimization. The main attention CUDA binary is unchanged.

> **CTA follow-up:** [CTA_REPORT.md](CTA_REPORT.md) records the warp-specialized prologue, persistent epilogue experiments, barrier ownership and measured dispatch. L384 now uses the producer/consumer prologue by default.

> **Current result:** native CUDA/TMA surrounds are now the default for the qualified H100 cases. See [CUDA_REPORT.md](CUDA_REPORT.md) for implementation, correctness, bandwidth definitions, and final serving measurements.

> First Codex pass on 2026-09-21: the settings are now wired into the serving path, with additional starting-direction traversal/tiling improvements. See [CODEX_REPORT.md](CODEX_REPORT.md) for verified results and reproduction. The remaining text is the original handoff; its first task is completed.

You are continuing an optimisation campaign on the **TriangleAttention block** of the Anthropic
`uplifting-biomolecular-modeling` release (`opt_core`), targeting MiniWorld's training shape on an H100.
Everything below is measured on this machine. Read the **Traps** section before running anything: four of the
traps silently produce numbers that look like measurements and are not.

## 1. The target

One TriangleAttention block = three kernels in sequence.

| # | kernel | file | what it does |
|---|---|---|---|
| 1 | `_triatt_prologue_kernel` (Triton) | `oc/opt_core/kernels/fpf_triatt_pro/prologue.py` | LayerNorm(z) then 5 projections → q, k, v, g, bias |
| 2 | `triattn_m1_kernel` (CUDA/CUTLASS, route `cuda_b`) | `oc/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/cuda_b/csrc/m1/triattn_m1_sm90.cuh` | the attention itself |
| 3 | `_triatt_epilogue_kernel_v2` (Triton) | `oc/opt_core/kernels/fpf_triatt_epi/epilogue.py` | gate, output projection, residual add |

At **S < 512 the core is a different kernel** — a Triton `_fwd` (route `k13`/`cuda`). Changes to the CUDA `.cuh`
do not affect L384 at all. Always check which core ran (see Traps 1).

Shape under test: `d_pair=128, n_head=4, head_dim=32` (C=128, H=4, D=32), bf16, `L ∈ {384, 768, 1024}`.

## 2. Where things stand

Against the pristine release, same process, **identical core `.so`** (hash printed by the harness),
in-place residual block:

| L | release | ours | speedup |
|---|---:|---:|---:|
| 384 | 377.4 µs | 347.9 | **1.085x** |
| 768 | 1726.2 | 1590.7 | **1.090x** |
| 1024 | 3407.4 | 3186.7 | **1.069x** |

L768 breakdown: prologue 460→364, core 874→881 (unchanged), epilogue 260→220.
`rel_rms` vs an fp32 reference is **1.7407e-03 for both** — the changes are numerically indistinguishable.

The release cannot do out-of-place residual at all (`tri_attn_block(..., out=)` → `TypeError`); ours can,
which is what the engine wants, and that path is 1550.4 µs at L768.

Three changes produced all of it, all bit-preserving:

1. **prologue `maxnreg=128` + `BN=64`** — 465→358 µs (−23%). The kernel compiled to 190 registers; at 256
   threads that is 1 CTA/SM = 12.5 % occupancy on a kernel whose roof is DRAM. Capping registers gives
   2 CTA/SM (confirmed: `launch__occupancy_limit_registers` 1→2, `sm__warps_active` 12.5→24.4 %).
2. **epilogue tile `BI=8, BJ=8, num_warps=4, num_stages=1`** (was 16×8) — 260→223 µs (−14%).
3. **epilogue residual fusion** (`RESIDUAL_OUT`, already in the tree) — writes `z+u` to a separate buffer,
   removing a z-sized pass. −7%.

**Changes 1 and 2 are still carried by environment variables, not wired in.** See §5.

## 3. Dead ends — do not repeat these

Each was measured properly (fallback-checked, 3 rounds).

| idea | result | why |
|---|---|---|
| core: `ex2.approx.f16x2` (two PTX lanes; CUDA12.9 actually emits two MUFU.EX2.F16 instructions plus PRMT, see core_sol90/half_exp_probe.sass) | **wrong and 5 % slower** | the hot pass seeds the softmax offset at `kShift = 64` so `p = 2^-64`; that is what buys the max-free pass (a later logit may exceed the seed by ~190 log2 units before fp32 `ex2` overflows). fp16 has ~40 usable log2 units — the window does not exist. max\|err\| ~1.0 vs 2e-3, 10/17 package test vectors fail. |
| core flag 8 (`kPrmtPack`: pack P with PRMT) | −1.9 %, worse numerics | Later H100 instruction measurements found zero XU work for both F2FP and PRMT; the earlier shared-XU explanation was incorrect. See `core_sol90/xu_probe_summary.json`. |
| core flag 32 (E-phase token ring) | **+22.7 %** | |
| core flag 1048576 (Q in registers) | +12.9 % | |
| core flag 4194304 (`kW`) | +8.2 % | |
| core flags 8388608 / 16777216 (warpgroup de-phasing) | ±0.3 % | |
| core: hand the bias as bf16 instead of fp32 | 0 % | the Python wrapper already stages the bias into its own buffer (`bias_staged`); L2 traffic identical to the byte (4882 vs 4884 MB) |
| prologue: drop the `broadcast_to([BM,C])` gamma/beta tiles (`FPF_TRIATTPRO_NOASM=1`) | 0 % | |
| prologue: split the `[BM,BN]` int64 store-address chain (`FPF_TRIATTPRO_ADDR=split`) | 0 % | Triton already folds it |
| prologue: past 2 CTA/SM (maxnreg 64–96, 4-warp CTAs, BN=32) | +20 % … +250 % | spills |
| epilogue: `maxnreg` < 200; and any `maxnreg` together with the 8×8 tile | +20 % … +38 % | this kernel wants registers, not occupancy |
| **q/k/v pair-natural layout** (`[I,J,H,D]` storage handed as the `[I,H,J,D]` view) | prologue **+14 %** (768), **+16 %** (1024) | the head layout is already good: a CTA tile spans BJ=16 **consecutive** j, and in `[I,H,J,D]` those 16 j × 32 d are adjacent → one **1024 B** run per (i,h), not the 64 B a per-element reading suggests. The core itself is within 0.3 % of either layout (measured), so the layout is free to change — it just does not help. |

Also: `bench_bandwidth.py` exists but **is a poor proxy**. It measures `torch.Tensor.copy_` over permuted views,
not a tiled kernel store, and it is what led to the wrong layout conclusion above. Its one trustworthy row is
the pure-stream copy: **2.95 TB/s at L768** (88 % of the 3.35 TB/s HBM3 peak). Use that as the streaming ceiling.

## 4. Traps — every one of these has already cost a day

1. **The integrity gate chain and the silent fallback.** The payload verifies, in order: `source_sha256`
   (every file under the extension's `csrc`) → `so_sha256` → the payload's `SHA256SUMS` → `loadcheck`
   (byte-compare of the kernel's output against `testvectors/<case>.expected.pt`). **Any failure falls back to
   `flash_triattn` silently**, which at L768 costs ~1710 µs instead of 883 and reads as a plausible number.
   Three A/B runs were invalidated this way before it was caught.
   - After ANY edit under `csrc/`, rebuild: `pkg/v11/tools/build_prebuilt.py --exts triattn_m1_ext`.
   - Then run `python refresh_sums.py <dir containing opt_core>` — `build_prebuilt.py` writes the `.so` and
     its record but **not** the two `SHA256SUMS` files.
   - A kernel that changes numerics cannot pass `loadcheck` by construction. Regenerate its own vectors with
     `test_pkg.py --generate` (keep the pristine set: `tv/pristine/` holds it), and judge correctness with
     `test_pkg.py` instead, which reports max\|err\| against a stored **fp64** reference and gates at
     1.5× the recorded error.
   - **Every run must be checked**: `grep "refused by kernels.triattn" <log>`, and confirm the core kernel
     name in the profile. `summarise_ab.py` prints the core kernel name per row for exactly this reason.
2. **Two Slurm jobs must never share the `oc/` tree.** One job rebuilding the prebuilt mid-flight corrupted
   7 of 12 rows of another job's sweep. Serialize with `sbatch --dependency=afterany:<jobid>` or give each
   job its own tree.
3. **The live config is `oc/opt_core/attn/pair_fused_cells.json`**, resolved by `pair_fused.pick_cell`.
   The `PINNED_CONFIG` dicts inside `prologue.py` / `epilogue.py` are **not** what the block path reads —
   editing them does nothing. Rows for this shape: `r07` (prologue, key `[128,4,32]`, cc 9.0) and `r16`
   (epilogue, same key).
4. **`TriangleAttention` zero-inits `to_out.weight`**, so the module's update is exactly 0 and any
   correctness comparison passes vacuously (this produced `rel_rms = 0.000e+00` and hid a broken check).
   Re-draw the weights — see `init()` in `sweep_pro.py` — and assert the reference update is non-zero.
   The module **owns its residual**: `forward(z)` returns `z + u`, not `u`.
5. **Slurm**: `-p h100 --qos=cssb_h100 --account=cssb --gres=gpu:h100:1`. Do not pin `-w node02` (it is
   usually full, and a pinned pending job blocks everything behind it by priority).
6. **NCU**: writing CSV to stdout fails (`LookupError: unknown encoding: utf-8-sig`). Export a `.ncu-rep`,
   then `ncu --import x.ncu-rep --page raw --csv`. Some section combinations silently return 0 for
   `dram__bytes_*` — check the value before trusting a bandwidth number.

## 5. First task: wire in what is already measured

Changes 1 and 2 above are still passed as environment variables. Move them into the cell rows so they are
what the block actually runs.

- `prologue.py` and `epilogue.py` already read `BN`, `maxnreg` (and `qkv_pair`) from `cfg`, falling back to
  the env override. So this is a data change plus verification.
- Update `pair_fused_cells.json`: `r07.cfg` += `{"BN": 64, "maxnreg": 128}`; `r16.cfg` → `BI: 8, BJ: 8`
  (leave `num_warps: 4, num_stages: 1`, and add **no** `maxnreg` — it is worse here).
- Both rows are `status: certified` with qualification evidence. Rewrite the `evidence` text to state what
  was re-measured and that the settings are bit-preserving (they change no arithmetic: `BN` splits output
  columns, not the K chain; `maxnreg` and the tile change no operation order — `rel_rms` was identical to
  5 figures across every variant).
- **Verify the wiring took**: run `sweep_pro.py` with no env overrides and confirm prologue ≈ 358 µs and
  epilogue ≈ 223 µs at L768. Additionally assert bitwise equality of the block output before vs after the
  cell change (`torch.equal`), since these settings are supposed to change nothing.

## 6. Then: the open question

Against the measured 2.95 TB/s streaming ceiling:

| kernel | now (L768) | DRAM bytes | achieved | % of stream ceiling |
|---|---:|---:|---:|---:|
| prologue | 358 µs | 764 MB | 2.13 TB/s | **72 %** |
| epilogue | 223 µs | 506 MB (NCU-measured; 604 MB nominal, ~100 MB served by L2) | 2.27 TB/s | **77 %** |

So ~82 µs and ~51 µs are unaccounted for. Both kernels are now at 2 CTA/SM. Find where the rest goes
before changing anything — profile with SpeedOfLight + MemoryWorkloadAnalysis + Occupancy + WarpStateStats
(`step28.sbatch` does this; `ncu/best-L768.ncu-rep` is a run at the current best settings).

The core is the largest single piece and the hardest:

- 881 µs at L768. Compute SoL 56 %, XU pipe 46 %, tensor pipe 32 %, issue slots 40 %.
- 128 registers × 512 threads = the whole register file → **1 CTA/SM**, 16 warps, `sm__warps_active` 21 %.
- SASS (326 M instructions): FFMA 17.6 %, MUFU 17.5 %, UMOV 10.0 %, F2FP 9.0 %, HGMMA 6.6 %.
  Stall samples: **MUFU 42 %**, **BRA 24 %**, WARPGROUP 8.9 %, HGMMA 8.7 %.
- Neither pipe is saturated, so it is issue/overlap bound, not throughput bound. Every compile-time flag the
  source exposes has been tried (table in §3). This needs restructuring, not a knob.

The one large structural lever nobody has tried: **fuse the prologue into the core** so q/k/v never reach
HBM — that removes ~906 MB of round-trip (453 MB written by the prologue + the core's reads). It means
computing the 5 projections inside a warp-specialised CUTLASS kernel. Estimate the win first from the
byte accounting above before committing to it, and note that the core's DRAM SoL is only 16.8 % (its
traffic is mostly L2: 4.88 GB at 4.21 TB/s), so the win lands on the prologue's side of the ledger.

## 7. Tooling in this directory

| script | use |
|---|---|
| `sweep_pro.py --length L --output x.json` | block level: op time (CUDA graph), per-kernel breakdown, `rel_rms` |
| `summarise_ab.py <tag> <variant>...` | median over rounds; **prints the core kernel name** so a fallback is visible |
| `bench_core_flags.py --length L --flags 0,8 --qkv-layout {nhsd,nshd,shnd} --bias-dtype {fp32,bf16}` | the core alone, imported below the gate (so gate refusals cannot hide); checks vs an fp64 reference |
| `bench_vs_release.py --oc <dir> --tag <name> --length L` | release vs ours; `--oc` decides which `opt_core` is imported and the script asserts it |
| `install_variant.py <snapshot>` / `refresh_sums.py <dir>` | install a prebuilt / repair the two `SHA256SUMS` after a local rebuild |
| `probe_regs.py`, `probe_ref.py` | compiled register count; the zero-init reference check |

Trees: `oc/opt_core` = the patched tree. `oc-release/opt_core` = a pristine snapshot of
`/home/psk6950/ext/uplifting-biomolecular-modeling/common/opt_core/opt_core` (do not modify the original).
The core source in `oc/` is currently **byte-identical to the release** (the f16x2 experiment was reverted),
so one prebuilt serves both.

Environment: wrap every GPU command as
`bash /home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/env.sh bash -c "..."`, with
`CUTLASS_PATH=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/cutlass-4.2` for any CUDA rebuild.

## 8. Standing rules for this work

- Three rounds minimum per variant, report the median, and keep a baseline row **inside the same job**.
- Every timing run is invalid until the fallback check has passed. State it in the report.
- Report negative results with the same weight as positive ones — most of §3 was found by a hypothesis that
  sounded right and measured wrong. Say which hypothesis died and why.
- Distinguish "% of peak" from "% of what this machine achieves" and say which you mean.
