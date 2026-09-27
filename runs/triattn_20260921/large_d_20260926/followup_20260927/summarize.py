import hashlib
import json
from pathlib import Path
import random
import statistics
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):
    d=json.loads((ROOT/name).read_text());assert d.get('complete'),name
    return d
baseline=json.loads((ROOT/'BASELINE.json').read_text())
for name,h in baseline['python_sha256'].items():assert digest(BASE/name)==h,name
for name,info in baseline['artifacts'].items():
    for n,h in info['sha256'].items():assert digest((BASE/name).parent/n)==h,(name,n)
engine=BASE.parent.parent/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention'
for name,h in baseline['installed_checkpoint18246_unchanged_sha256'].items():assert digest(engine/name)==h,name

for p in ROOT.glob('check-bias-h*-19508.json'):read(p.name)
assert len(list(ROOT.glob('check-bias-h*-19508.json')))==16
for C in (256,512):read(f'qualify-C{C}-19539.json');read(f'selected-graph-C{C}-19536.json')
read('bias-stress-19541.json')
for i in range(6):
    s=(ROOT/f'sanitize-19522_{i}.log').read_text()
    assert ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if i%3==1 else 'ERROR SUMMARY: 0 errors') in s
s=(ROOT/'fast-check-19536.log').read_text()
assert s.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')==1
assert s.count('ERROR SUMMARY: 0 errors')==2

rows=[];random.seed(927)
for C,kind,job in ((256,'prior',19517),(512,'accumulate_fast',19537)):
    for L in (384,768):
        path=f'compare-C{C}-L{L}-{kind}-b1-{job}.json';d=read(path)
        for e in (0,1):
            for regime in ('forward','backward','forward_backward'):
                record=d['measurements'][f'{regime}_e{e}'];arms=record['arms']
                names=[r['name'] for r in arms['next_wide']['profile']['kernels']]
                if regime!='forward':
                    assert any('wide_grouped_dkdv_bias16' in n for n in names)
                    assert any('wide_dq_resident_alias' in n for n in names)
                    if C==512:
                        assert any('projection_bias_accumulate_fast' in n for n in names)
                        assert any('badd' in n for n in names)
                paired=[100*(1-b/a) for a,b in zip(arms['prior_wide']['rounds_ms'],arms['next_wide']['rounds_ms'])]
                boots=sorted(statistics.median(random.choices(paired,k=len(paired))) for _ in range(4000))
                ci=[boots[100],boots[3899]]
                if regime!='forward':assert ci[0]>0,(C,L,e,regime,ci)
                rows.append(dict(width=C,length=L,ending=bool(e),regime=regime,source=path,
                                 ms={n:a['ms'] for n,a in arms.items()},speedup=record['speedup'],
                                 incremental_speedup=record['incremental_speedup'],
                                 paired_median_time_saved_pct=statistics.median(paired),paired_bootstrap95_time_saved_pct=ci))
(ROOT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')

lines=['# Wide TriangleAttention follow-up — 2026-09-27','',
       'Qualified explicit experiment; installed engine dispatch is unchanged. The previous qualified wide candidate and the original Triton engine are distinct baselines. Width D=256/512 is total pair/QKV width with4 heads (head64/128), B1, BF16 on node02 H10080GB. FWD remains the previous native attention implementation; this follow-up targets backward HBM intermediates.', '',
       '| Width | L | BWD vs original Triton | F+B vs original Triton | BWD vs previous wide | F+B vs previous wide |',
       '|---:|---:|---:|---:|---:|---:|']
for C in (256,512):
    for L in (384,768):
        cells=[]
        for key,regime in (('speedup','backward'),('speedup','forward_backward'),('incremental_speedup','backward'),('incremental_speedup','forward_backward')):
            v=[r[key] for r in rows if r['width']==C and r['length']==L and r['regime']==regime]
            cells.append(f'{min(v):.3f}–{max(v):.3f}x')
        lines.append(f'| {C} | {L} | '+' | '.join(cells)+' |')
lines+=['','Ranges cover starting and ending. All three arms run in the same job/process on the same GPU, using all six replay-order permutations,18 rounds of10 CUDA Graph replays. Outputs and all9 input/parameter gradients pass against original Triton before timing. Profiler confirms3 cudaGraphLaunch calls and actual native/core/compact-bias kernels. The new width512 projection path shows cuBLAS beta-add GEMMs plus the native vector epilogue. Each BWD/F+B paired bootstrap95% interval for time saved over the previous candidate is positive; raw rounds and interval calculation are in summary.json. These intervals cover within-job variation, not all future machines/runs.', '',
        '| Width | L | Direction | Original Triton F+B ms | Previous wide F+B ms | New wide F+B ms |',
        '|---:|---:|:---|---:|---:|---:|---:|']
for r in rows:
    if r['regime']!='forward_backward':continue
    m=r['ms'];direction='ending' if r['ending'] else 'starting'
    lines.append(f'| {r["width"]} | {r["length"]} | {direction} | {m["original_triton"]:.3f} | {m["prior_wide"]:.3f} | {m["next_wide"]:.3f} |')
lines+=['','**Selected changes.** Both widths retain FP32 local bias-gradient accumulation but store the grouped partial buffer as BF16, cutting that buffer and its logical write/read byte count in half. The final reduction is FP32. At L768, partial-buffer size drops from1.812→0.906GB for width256 and3.624→1.812GB for width512 (decimal GB). dK/dV remains bitwise equal in the direct comparison; dBias gains one BF16 rounding step. The buffer is still cubic in L and has not been eliminated.', '',
        'Width256 retains the previous native TMA/WGMMA projection-dgrad fusion. Width512 uses cuBLAS GEMMs that accumulate into the same BF16 dX output with beta=1, then a native CUDA bias-projection epilogue. This removes separately materialized projection dX tensors and their separate addition kernels. It still reads/writes dX between GEMMs; it is not an all-on-chip projection fusion. The epilogue specializes256/512 channels, replaces integer division with compile-time indexing, and accesses BF16x2.', '',
        '**Measured effects at width512/L768 starting.** The initial scalar bias epilogue took1.193ms; the bitwise-equivalent specialized vector epilogue takes0.519ms. The prior separate add kernels took about2.91ms and the selected path retains about0.58ms of residual additions. Bias reduction drops from about1.17ms to0.69ms. Grouped dK/dV remains around5ms and dQ around2.8ms; they are still major bottlenecks. Width256 bias reduction drops from about0.59ms to0.35ms.', '',
        '**Rejected controls.** Four new native projection organizations were measured at width512 against the previous custom fused projection: a3-stage pipeline was approximately tied; two consumer warpgroups sharing weights, N256 tiles, and K128 tiles were slower. Initial aggressive occupancy requests for three configurations failed PTXAS C7602; the relaxed configurations compiled and passed bitwise/FP64 checks before being timed. These controls were not selected or promoted. The beta-accumulate path with the scalar epilogue was also slower than the final vector epilogue. At width256 beta accumulation lost to the previous native fusion, so that width preserves it.', '',
        '**Accuracy and safety.** Compact-bias job19508 passes16 independent FP64 output/adjoint cases (head64/128, L64/128, four masks), plus changed-input graph replay. Whole-module PyTorch job19539 covers both directions, dropout, frozen branches and two SGD steps with changed inputs/weights/dy/masks; maximum relative error is under0.85%. Job19536 confirms full-module changed-state graphs and actual dispatch. Native epilogue changes are bitwise equal to the scalar version at both widths and small/nonmultiple/large row counts.', '',
        'Compact bias storage increases dBias FP64 relative-L2 error in the stress fixture from about0.242% to0.296%; all pre-existing full-core and full-module tolerance limits remain unchanged. dO magnitude65536 passes with finite dBias values above4million, preserving BF16 range. The cuBLAS accumulation path has FP64 projection error about0.356%, matching the original split BF16 reference (the all-FP32 fused custom path was about0.166%). It passes the full-module comparison budget. These are measured rounding differences, not bitwise claims for the entire new backward.', '',
        'All three sanitizer tools pass for compact bias at L256 with repeated TMA-stage reuse (19522_0–5), and for the final vector epilogue/accumulation graph at both widths and boundary row counts (19536). No sanitizer errors, race hazards or warnings. Initial scalar accumulation also passed19529.', '',
        '**Use and scope.** Import selected_next.attach from this directory and apply it to a compatible original-Triton TriangleAttention instance. It chooses prior projection fusion for width256 and beta accumulation plus the vector epilogue for width512, with compact bias partials for both. Baseline runtime sources/binaries are frozen and verified through BASELINE.json; all48 installed checkpoint18246 file hashes remain unchanged. New native binaries and runtime sources are pinned in CANDIDATE.json. This is an explicit experimental entry, not a global engine installation.', '',
        'No new FWD algorithm improvement is claimed. No NCU wide SOL measurement was taken and SOL90 is not established. This does not qualify B>1, other head counts, head dimensions256/512, wide L1024, QK normalization or torch.compile integration. All owned Slurm jobs are terminal; the failed builds and dependency-cancelled screen are retained as rejected evidence.', '',
        '**Evidence.** Selected paired result files:']
for name in sorted(set(r['source'] for r in rows)):lines.append(f'- [{name}]({name})')
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
artifacts={}
for folder in ('bias16_C256','bias16_C512','accumulate_fast_C512'):
    info=json.loads((ROOT/folder/'ready.json').read_text())
    for n,h in info['sha256'].items():assert digest(ROOT/folder/n)==h,(folder,n)
    artifacts[folder]=info
sources={name:digest(ROOT/name) for name in ('selected_next.py','experiments.py','accumulate.py','build.py')}
manifest=dict(status='qualified_experiment_not_installed',baseline='BASELINE.json',baseline_sha256=digest(ROOT/'BASELINE.json'),
              entry='selected_next.attach',runtime_sha256=sources,artifacts=artifacts,
              selection={'256':'prior native projection + compact bias','512':'cuBLAS beta accumulation + vector CUDA bias epilogue + compact bias'},
              validation_jobs=[19508,19522,19536,19539,19541],comparison_jobs=[19517,19537],
              measurements=sorted(set(r['source'] for r in rows)),SOL_status='not measured; SOL90 not established')
(ROOT/'CANDIDATE.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('\n'.join(lines[:11]))
