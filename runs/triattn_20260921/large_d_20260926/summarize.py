"""Rebuild the wide report and qualified-artifact manifest from completed evidence."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(name):
    d=json.loads((ROOT/name).read_text());assert d.get('complete'),name
    return d


selected=[]
for C,prefix,job in ((256,'compare-front',19431),(512,'compare',19416)):
    for L in (384,768):
        name=f'{prefix}-C{C}-L{L}-full-{job}.json'
        selected.append((C,L,name,read(name)))
for p in ROOT.glob('check-h*-19405.json'):read(p.name)
assert len(list(ROOT.glob('check-h*-19405.json')))==16
for C in (256,512):
    for stem,job in (('qualify',19417),('qualify-front',19430),('projection-check',19429),('selected-graph',19445)):
        read(f'{stem}-C{C}-{job}.json')
for i in range(6):
    s=(ROOT/f'sanitize-19418_{i}.log').read_text()
    required='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if i%3==1 else 'ERROR SUMMARY: 0 errors'
    assert required in s
s=(ROOT/'projection-check-19429.log').read_text()
assert s.count('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')==2
assert s.count('ERROR SUMMARY: 0 errors')==4

lines=['# Wide TriangleAttention results — 2026-09-27','',
       'Qualified experimental native CUDA implementation; installed engine dispatch is unchanged. C128 work is closed at training18246/inference19157. The original engine baseline is the original Triton module/core, not our installed C128 CUDA checkpoint.', '',
       'Here D means pair width and total QKV width:256/512, four heads, hence head64/128. B1, H10080GB on node02, BF16 activation/projection weights, FP32 LN affine. Timings include LayerNorm, all projections, attention, gate, residual and all nine input/parameter gradients where applicable. FWD retains training saves. They are CUDA Graph timings, not eager host-launch latency.', '',
       'Selection: width256 uses native attention plus fused projection input gradients; width512 uses native attention with the original projection path. Both retain one head per CTA. dK/dV groups4 or2 outer rows per CTA, respectively.', '',
       '| Width | L | FWD speedup | BWD speedup | F+B speedup |',
       '|---:|---:|---:|---:|---:|']
for C,L,name,d in selected:
    cells=[]
    for regime in ('forward','backward','forward_backward'):
        vs=[d['measurements'][f'{regime}_e{e}']['speedup'] for e in (0,1)]
        cells.append(f'{min(vs):.3f}–{max(vs):.3f}x')
    lines.append(f'| {C} | {L} | '+ ' | '.join(cells)+' |')
lines+=['','Ranges cover starting and ending directions. Every speedup uses original/candidate measurements from the same job and alternating replay order; widths256 and512 use jobs19431 and19416, respectively. Each cell has16 paired rounds of10 replays; profiler uses3 further replays and verifies actual native kernel names and cudaGraphLaunch.', '',
        '| Width | L | Direction | Original Triton F+B ms | Selected CUDA F+B ms | Time saved |',
        '|---:|---:|:---|---:|---:|---:|']
for C,L,name,d in selected:
    for e in (0,1):
        v=d['measurements'][f'forward_backward_e{e}']['arms'];a=v['original_triton']['ms'];b=v['wide_cuda']['ms']
        direction='ending' if e else 'starting'
        lines.append(f'| {C} | {L} | {direction} | {a:.3f} | {b:.3f} | {(1-b/a)*100:.1f}% |')
lines+=['','**What changed.** Wide native FWD uses TMA Q/K/V/bias and WGMMA online softmax. dQ keeps Q/dO in registers, reuses their retired shared storage for bias and doubles K/V stages. dK/dV uses one producer warpgroup and two consumers, fuses bias-gradient partial generation, then reduces those partials. Shared layouts are transposed from the actual load layout; the first naive wide layout failed FP64 and is retained under rejected_v1_transpose.', '',
        'At width256, fused projection dgrad accumulates four square products and the narrow bias product on chip before one BF16 store, eliminating four full input-gradient intermediates and their separate additions. Parameter gradients retain cuBLAS. Width512 projection fusion passed correctness but was slower than attention-only CUDA and is rejected for selection. The64-row projection tile beats128 rows at both lengths and widths; projection-tiles-19436.json records the bitwise/timing comparison.', '',
        '**Resource facts.** Head64/128 FWD uses58,368/99,328 bytes SMEM and96/129 registers. dQ uses50,176/99,328 bytes and154/218 registers. Grouped dK/dV uses175,104/157,696 bytes. All builds have zero spills. Projection M64 uses90 registers; its experimental setmaxnreg requests were ignored by PTXAS (C7508), so it uses the reported static allocation. Do not claim active register redistribution in that projection kernel.', '',
        'Bias-gradient FP32 scratch is half the original BF16 per-outer-row scratch at R4/head64. At R2/head128 the byte count is equal; the head128 attention gain comes from kernel scheduling, not a claimed scratch-byte reduction.', '',
        '**Validation.** Job19405 passes16 independent FP64 core forward/adjoint cases across head64/128, L64/128, no mask/mixed/one-key/all-masked. Initial and changed-input graph replays are bitwise with native eager execution. One-key reference-zero gradients use an absolute bound; all-masked outputs/gradients are exactly zero. Whole-module PyTorch checks19417/19430 cover both directions, dropout0/.1, frozen parameters, changed inputs/masks/dy/weights and two SGD steps. Job19445 checks the selected entry in full-module graphs with three input/weight/dy/mask versions and actual native dispatch.', '',
        'Core memcheck/racecheck/synccheck19418_0–5 pass at L256, exercising repeated TMA buffer reuse. Projection FP64, changed-gradient/weight graphs and all three sanitizers19429 pass at both widths. No sanitizer errors, race hazards or warnings. Full target-shape paired benchmarks check outputs and all nine gradients against original Triton before reporting timings.', '',
        '**Remaining bottlenecks.** At L768 starting, selected C256 backward spends about2.87ms in grouped dK/dV,1.52ms in dQ and1.41ms in fused projection dgrad. C512 spends about5.02ms in grouped dK/dV,2.82ms in dQ and2.91ms in separate additions; projection/weight GEMMs also remain substantial. The grouped bias reduction still costs0.59/1.16ms. Native delta preprocessing is not faster than the original Triton preprocessing. These are the next measured opportunities.', '',
        '**Scope.** No wide NCU SOL measurement was taken; SOL90 is not established. Head dimensions256/512, B>1, other head counts, QK normalization, wide L1024 and torch.compile integration are not qualified by this campaign. Installed training checkpoint18246 retains all48 original file hashes. All owned jobs are terminal.', '',
        '**Use the explicit candidate.** Run through the project env.sh, put this directory on sys.path, and call selected.attach(model) on a compatible original-Triton TriangleAttention instance. selected.py chooses the qualified width-specific path; it does not replace global engine dispatch. Load the desired trained state normally. Source/binary hashes and evidence paths are pinned in CANDIDATE.json.', '',
        '**Evidence.** BASELINE.md and baseline-C*-19395.json hold the untouched wide original-Triton profile. Selected paired timings:']
for C,L,name,d in selected:lines.append(f'- [{name}]({name})')
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

artifacts={}
for folder in ('native_head64','native_head128','projection_C256','projection_C512'):
    for p in (ROOT/folder).glob('*ready.json'):
        data=json.loads(p.read_text())
        for name,h in data['sha256'].items():assert digest(p.parent/name)==h,(p,name)
        artifacts[str(p.relative_to(ROOT))]=data
aux=json.loads((ROOT/'aux-ready.json').read_text())
for name,h in aux['sha256'].items():assert digest(ROOT/name)==h
artifacts['aux-ready.json']=aux
engine=ROOT.parent.parent/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention'
checkpoint=json.loads((ROOT.parent/'fwd_training/checkpoint18246/snapshot.json').read_text())['sha256']
for name,h in checkpoint.items():assert digest(engine/name)==h,name
sources={str(p.relative_to(ROOT)):digest(p) for p in ROOT.glob('*.py')}
manifest=dict(status='qualified_experiment_not_installed',width_to_path={'256':'native attention + projection dgrad M64','512':'native attention only'},
              assumptions=dict(batch=1,heads=4,pair_width_equals_qkv_width=True,head_dims=[64,128],performance_lengths=[384,768]),
              artifacts=artifacts,python_sha256=sources,selected_measurements=[name for _,_,name,_ in selected],
              validation_jobs=[19405,19417,19418,19429,19430,19445],comparison_jobs=[19416,19431],
              installed_checkpoint18246_unchanged_sha256=checkpoint,SOL90=False,SOL_status='not measured for wide shapes')
(ROOT/'CANDIDATE.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('\n'.join(lines[:16]))
