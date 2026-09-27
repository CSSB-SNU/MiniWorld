"""Reproducible summary of this pass, always against its own saved baseline."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BWD=ROOT.parent
log=(ROOT/'final-16663.log').read_text()
for marker in ['WGRAD_OPCHECK_PASS','GUARDS_PASS 10','COLD_COMPILE_PASS','AMP_PASS torch.bfloat16 True','FP16_NATIVE_DECLINED']:
    assert marker in log,marker
combined=[]
for L in (384,768,1024):
    data=json.loads((ROOT/f'combined-16663-L{L}.json').read_text());assert data['complete']
    combined.append(data)

def ncu(path):
    ls=path.read_text().splitlines();start=next(i for i,l in enumerate(ls) if l.startswith('"ID"'))
    rows=list(csv.DictReader(ls[start:]));return rows[0],rows[1:]
units,rows=ncu(BWD/'wgrad/profile-16659.csv')
assert units['dram__bytes_read.sum']=='Mbyte'
traffic=[sum(float(row['dram__bytes_read.sum'])+float(row['dram__bytes_write.sum']) for row in rs) for rs in (rows[:8],rows[8:])]
wgmain=next(row for row in rows if row['Kernel Name'].startswith('grouped_wgrad'))
bu,br=ncu(BWD/'below80/profile-16658.csv');biasmain=br[0]
report=dict(final_job=16663,combined=combined,attribution_job=16664,
    installation=json.loads((ROOT/'installation.json').read_text()),
    wgrad=dict(qualification_job=16662,ncu_job=16659,traffic_Mbytes=traffic,traffic_reduction_percent=100*(1-traffic[1]/traffic[0]),
               hbm_sol=float(wgmain['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'])),
    bias=dict(qualification_job=16653,ncu_job=16658,memory_sol=float(biasmain['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed']),
              sm_sol=float(biasmain['sm__throughput.avg.pct_of_peak_sustained_elapsed']),hbm_sol=float(biasmain['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'])),
    rejected={})
for j in (16643,16644,16648,16649):
    entries=[]
    for L in (384,768,1024):
        data=json.loads((BWD/'below80'/f'pilot-{j}-L{L}.json').read_text())
        r=data['records'][0];entries.append(dict(length=L,artifact=data['artifact'],kind=data['kind'],speedup=r['speedup'],relative_l2=r['relative_l2']))
    report['rejected'][str(j)]=entries
(ROOT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Installed backward optimization — job16663','',
'2026-09-23. Native CUDA C++/TMA; H100 80GB, B1 BF16 C128/H4/D32.',
'Optimize dK/dV, dQ, then weight gradients after full-backward attribution16607.',
'Gate, projection/LN and final bias reduction remain unchanged under the user\'s 80% cutoff.',
'Forward optimization remains closed. Whole-backward SOL80/90 is not reached.','',
'## Installed full-module result','',
'This pass compares against the exact pre-pass installation16580, saved in `baseline/`.',
'All input and parameter gradients are returned. Both directions, dropout0, every seventh',
'key masked; 12 balanced AB/BA rounds of 15 CUDA graph replays. Times are median ms;',
'reductions use median paired ratios, so they need not equal the ratio of marginal medians.',
'Final installed job16663 follows independent combined-candidate job16661.','',
'| L | Direction | Bwd before | Bwd after | Reduction | F+B before | F+B after | Reduction |',
'|---:|---|---:|---:|---:|---:|---:|---:|']
for data in combined:
    for e in (0,1):
        b=data['measurements'][f'backward_e{e}'];f=data['measurements'][f'forward_backward_e{e}']
        lines.append(f"| {data['length']} | {'ending' if e else 'starting'} | {b['baseline_us']/1000:.3f} | {b['candidate_us']/1000:.3f} | {100*(1-1/b['speedup']):.2f}% | {f['baseline_us']/1000:.3f} | {f['candidate_us']/1000:.3f} | {100*(1-1/f['speedup']):.2f}% |")
lines += ['', '## What changed', '',
'- **dK/dV:** `bias_fusion/ldmatrix_bias` reads bias with two warp-level `ldmatrix.x4.trans` operations per score fragment. This replaces sixteen scalar shared loads without a new shared/HBM buffer or barrier. Q32/K64/R4, TMA producer plus four consumer warpgroups, FP32 bias partials and the final bias reducer are preserved. All isolated outputs and full-module gradients match the previous kernel bitwise. Pilot16652 reduces dK/dV-plus-reducer time by 1.83%, 3.09%, 3.18% at L384/768/1024.',
'- **Four weight gradients:** `wgrad/shared_z_tensor` fuses Q/K/V/gate dW. A CTA shares one Z tile across four consumer warpgroups, each accumulating its own weight gradient. One producer warpgroup streams Z and four gradient tiles with double-buffered TMA. Tokens are split by 2304/8960/15936 at L384/768/1024, producing 128/132/132 CTAs. FP32 partials are deterministically reduced; BF16 rounding occurs only at final output. No atomics or BF16 partial reduction.',
'- **Dispatch:** native grouped dW is used only for qualified BF16 contiguous shapes with all four wide weight gradients requested. Partial/frozen parameter sets and unsupported metadata retain the prior GEMMs. Output dW and the narrow bias dW retain cuBLAS. The custom op returns one fresh [4,128,128] allocation; unbind happens outside the opaque boundary.',
'- **dQ:** no new candidate was adopted. Its installed `vector_bias` CUDA kernel remains unchanged.', '',
'Package files: `bias_fusion.cu`, `wgrad.cu`, `wgrad_backward.py`, `build_wgrad.py`,',
'`ln_backward.py`, versioned `.so` binaries and three updated/new manifests.',
'`install.py` verifies source/binary hashes, full qualification and this pass\'s baseline before replacing files.', '',
'## Memory and utilization', '',
f"NCU16659 includes all four former cuBLAS GEMMs and their reducers versus the fused pair. Measured HBM read+write traffic falls **{traffic[0]/1000:.3f} GB -> {traffic[1]/1000:.3f} GB ({100*(1-traffic[1]/traffic[0]):.2f}% less)** at L768, including FP32 partials. The fused main kernel reaches **HBM SOL85.27%**. Isolated final-API pilot16657 gives about35%/42%/45% shorter four-weight-gradient time at L384/768/1024.", '',
'NCU16658 dK/dV: memory SOL61.37%, SM51.06%, HBM33.67%; bias final reducer HBM93.95%. Lower instruction count makes dK/dV faster without increasing SM utilization. dQ remains the previous below80 kernel (NCU16557 memory/L2 SOL68.54%, SM58.09%). There is no complete-backward SOL80/90 claim.', '',
'[ATTRIBUTION.md](ATTRIBUTION.md) records the installed full-backward kernel accounting, job16664. Kernel counts fall **22 -> 16 starting**, **23 -> 17 ending**. The six parameter gradients now require six kernels rather than twelve. At L768 the remaining dK/dV+dQ work still accounts for about70% of profiled backward time; weight gradients fall to about7%. CUPTI totals and separate event times are reported independently.', '',
'## Validation', '',
'- dK/dV qualification16653: full-module/masks/dropout/SGD/Inductor/frozen parameters, independent FP64, exact-zero bias cancellation, memcheck/racecheck/synccheck; no errors.',
'- dW qualification16662: independent FP64 (relative L2 about0.00167 including BF16 output rounding), exact cancellation, all gradients in both directions, masks/dropout/SGD, fullgraph Inductor, partially/all frozen parameters, and three sanitizers; all pass.',
'- Final installed16663: all six manifest hashes/default flags, custom-op schema/autograd-registration/fake/AOT checks, ten fusion opt-outs, cold first compiled backward, full backward/F+B, BF16 autocast and FP16 native refusal; all pass.',
'- Combined full-gradient relative L2 stays below0.00038; forward is unchanged. Timing jobs16654/16656 isolate each improvement,16661 combines prototypes,16663 checks actual installed dispatch.', '',
'## Rejected candidates', '',
'| Candidate | Pilot | L768 speedup vs installed baseline | Disposition |',
'|---|---:|---:|---|']
for j,entries in report['rejected'].items():
    e=next(x for x in entries if x['length']==768)
    lines.append(f"| {e['kind']} / {e['artifact']} | {j} | {e['speedup']:.4f}x | Slower; not installed |")
lines += ['',
'The shared-dP dQ experiment retains spills (52B stores/60B loads) and adds on-chip traffic. Shared scaled bias adds an extra CTA barrier and layout conversion. Inter-iteration pipelining is bitwise correct but slower; dK/dV also reports compiler WGMMA serialization. Initial pipeline builds were rebuilt after source changes before any GPU test.',
'Test-only failures16655/16660 were Python import-registration/name-collision issues, fixed before passing qualification16662. They are not correctness or speed evidence.', '',
'## Reproduction', '',
'Use `runs/anthropic_adoption_20260919/env.sh` for the Torch/CUDA environment.',
'GPU scripts are Slurm H100 jobs. `final.sbatch` uses the saved pre-pass baseline',
'and currently installed candidates; `attribution.sbatch` measures the installed path.',
'`summarize.py` and `summarize_attribution.py --job 16664` rebuild these reports.',
'To rebuild package artifacts, run package `build_bias.py` and `build_wgrad.py`',
'with CUTLASS4.2 available. Rebuilt binaries require fresh performance qualification.']
(ROOT/'README.md').write_text('\n'.join(lines)+'\n')
print('REPORT_READY',report['wgrad'],flush=True)
