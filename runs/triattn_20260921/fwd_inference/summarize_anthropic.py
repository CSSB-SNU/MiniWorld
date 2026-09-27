"""Verify provenance and summarize the original Anthropic inference comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

ap = argparse.ArgumentParser()
ap.add_argument('--job', default='18528')
a = ap.parse_args()
root = Path(__file__).resolve().parent
release = root.parent/'oc-release/opt_core'
original = Path('/home/psk6950/ext/uplifting-biomolecular-modeling/common/opt_core/opt_core')
pkg = Path('kernels/triattn/triattn_native/pkg/v11/triattn_pkg')
prebuilt = pkg/'prebuilt/torch2.10.0+cu128-cpython-310-x86_64-linux-gnu'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((release/prebuilt/'triattn_m1_ext.json').read_text())
source_names = {
    'attn/pair_fused.py', 'attn/pair_fused_cells.json',
    'kernels/fpf_triatt_pro/prologue.py', 'kernels/fpf_triatt_epi/epilogue.py',
    str(pkg/'dispatch/candidate.py'), str(pkg/'dispatch/pins.py'),
}
for name, expected in manifest['source_sha256'].items():
    path = pkg/'cuda_b/csrc'/name
    assert digest(release/path) == expected, path
    source_names.add(str(path))
for name, expected in manifest['module_sha256'].items():
    path = pkg/'cuda_b'/name
    assert digest(release/path) == expected, path
    source_names.add(str(path))
assert digest(release/prebuilt/'triattn_m1_ext.so') == manifest['so_sha256']

rows, primary, correctness, dispatch, evidence = [], [], [], {}, {}
for length in (384, 768, 1024):
    path = root/('anthropic-%d-%s.json' % (length, a.job))
    data = json.loads(path.read_text())
    assert data['complete'] and data['rounds'] == 64 and data['replays'] == 40
    evidence[path.name] = digest(path)
    for name, expected in data['imported_release_sources'].items():
        name = str(Path(name).relative_to('opt_core'))
        assert digest(release/name) == expected, name
        source_names.add(name)
    dispatch[str(length)] = dict(served_cores=data['served_cores'], tier_memo=data['tier_memo'])
    for rec in data['records']:
        if rec['kind'] == 'correctness':
            correctness.append(dict(length=length, **rec))
            if rec['case'] != 'all_masked':
                assert rec['output_relative'] < .003 and rec['update_relative'] < .025
            else:
                assert rec['semantic_difference']
            assert rec['release_forms_relative'] == 0
        if rec['kind'] != 'paired_module':
            continue
        ratios = rec['paired_ratios']
        assert len(ratios) == 64
        rng = random.Random(92688)
        estimates = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(5000))
        rows.append(dict(length=length, ending=rec['ending'], release_form=rec['release_form'],
                         baseline_ms=rec['baseline_us']/1000, ours_ms=rec['candidate_us']/1000,
                         speedup=rec['speedup'], ours_time_change_percent=100*(1/rec['speedup']-1),
                         speedup_ci95=[estimates[125], estimates[4874]]))
    for ending in (False, True):
        choices = [r for r in rows if r['length'] == length and r['ending'] == ending]
        assert len(choices) == 2
        primary.append(min(choices, key=lambda r: r['baseline_ms']))
        profiles = [r for r in data['records'] if r['kind'] == 'profile' and r['ending'] == ending]
        assert len(profiles) == 3
        for rec in profiles:
            names = [e['kernel'] for e in rec['events']]
            if rec['path'] == 'ours':
                for stage in ('inference_ln_bias', 'qkv_attention_inference', 'inference_residual', 'nvjet'):
                    assert any(stage in n for n in names), (length, ending, stage)
            else:
                for stage in ('_triatt_prologue_kernel', '_triatt_epilogue_kernel_v2',
                              '_flash_triattn_fwd' if length == 384 else 'triattn_m1_kernel'):
                    assert any(stage in n for n in names), (length, ending, stage)

source_hashes = {}
for name in sorted(source_names):
    value = digest(release/name)
    assert value == digest(original/name), name
    source_hashes[name] = value
provenance = dict(release_root=str(release), original_root=str(original),
                  verified_original_source_sha256=source_hashes,
                  native_binary=str(release/prebuilt/'triattn_m1_ext.so'), native_manifest=manifest,
                  note='Original source; native binary rebuilt for the current Torch/CUDA stack.')
(root/'anthropic-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
for name in ('resident6', 'front8'):
    build = json.loads((root/name/'build-ready.json').read_text())
    for path, expected in build['sha256'].items():
        assert digest(root/name/path) == expected, (name, path)
report = dict(job=a.job, scope='complete inference FWD with input-preserving residual',
              baseline='original Anthropic opt_core: impl=fpf, core=default, ln=fused',
              candidate='resident6 + front8', installed=False, primary_rows=primary, all_rows=rows,
              primary_selection='lowest Anthropic median latency among two input-preserving residual forms',
              rounds=64, replays=40, correctness=correctness, dispatch=dispatch, evidence_sha256=evidence,
              provenance_sha256=digest(root/'anthropic-provenance.json'),
              affine='nontrivial BF16-representable values in FP32 LN parameters',
              all_masked_equivalent=False)
(root/'anthropic-results.json').write_text(json.dumps(report, indent=2)+'\n')

lines = ['# Inference FWD versus the original Anthropic implementation', '',
         'Job %s completed on node02 / normal_h100. Our resident6 + front8 wins at L384, '%a.job+
         'but loses at L768 and L1024. Earlier checkpoint18488 gains were against the installed '+
         'ordinary eval module, not the original Anthropic fused block.', '',
         'B1, C128, H4, D32, BF16, eval/inference_mode. Same inputs and nonzero random weights; '+
         'key mask disables every seventh key. Complete LN + bias + QKV + attention + gate + '+
         'output projection + residual, including ending orientation. Weight packing and compilation '+
         'are outside timing. Alternating AB/BA CUDA Graph measurements use 64 rounds x 40 replays.', '',
         'The primary table takes the faster measured Anthropic residual form per cell. Both forms '+
         'preserve the caller input. Negative time change means our kernel takes less time. Percentages '+
         'use median paired ratios; times are separate medians. Bootstrap intervals resample the 64 '+
         'paired ratios 5,000 times and describe this run, not variability across jobs.', '',
         '| L | Direction | Anthropic ms | Ours ms | Our time change | Speedup 95% CI | Anthropic form |',
         '|---:|---|---:|---:|---:|---|---|']
for row in primary:
    lo, hi = row['speedup_ci95']
    lines.append('| %d | %s | %.4f | %.4f | %+.2f%% | %.4f–%.4f | %s |' %
                 (row['length'], 'ending' if row['ending'] else 'starting', row['baseline_ms'],
                  row['ours_ms'], row['ours_time_change_percent'], lo, hi, row['release_form']))
lines += ['', '## Residual contract and all measurements', '',
          '`update_plus_add` runs the original block with residual=False and adds the input. '+
          '`copy_plus_fused_residual` copies the input into preallocated work storage and calls the '+
          'original block with residual=True. The copy is timed: upstream fused residual overwrites '+
          'its input, while our entry returns a new output. The two upstream forms return identical '+
          'outputs on all tested fixtures. This is not a timing of the upstream destructive API alone. '+
          'L1024 starting has effectively tied upstream forms; the observed lower median selects update_plus_add.', '',
          '| L | Direction | Anthropic form | Anthropic ms | Ours ms | Paired speedup |',
          '|---:|---|---|---:|---:|---:|']
for row in rows:
    lines.append('| %d | %s | %s | %.4f | %.4f | %.4f |' %
                 (row['length'], 'ending' if row['ending'] else 'starting', row['release_form'],
                  row['baseline_ms'], row['ours_ms'], row['speedup']))
lines += ['', '## Actual dispatch and provenance', '',
          'Original `oc-release/opt_core`, `impl=fpf`, `core=default`, `ln=fused`; no tuned cell overrides. '+
          'Default L384 selects flash_triattn (Triton). Default L768/L1024 selects tier:fast -> '+
          'triattn_native (CUDA M1). Served-core counters and CUDA profiles confirm these paths; '+
          'the L384 Triton selection is the intended default, not a refusal fallback. Surrounds use '+
          'the original fused prologue and epilogue. Our profile confirms front8, resident6, cuBLAS '+
          'output projection and residual. Some first-call profiler events are missing; every stage '+
          'is present in subsequent calls. Profiles establish dispatch, not timing totals.', '',
          '`anthropic-provenance.json` verifies imported opt_core sources, original surrounds, cell '+
          'table, native router and M1 source hashes against the upstream checkout. The native M1 '+
          'binary matches its source/build manifest: original source rebuilt for Torch2.10.0+cu128 '+
          'with NVCC12.9, not a claim of the originally distributed binary. The benchmark JSON field '+
          '`native_release_install` belongs to unused cuda_sm90a and is not evidence for the selected M1 core.', '',
          '## Numerical scope', '',
          'Upstream internally rounds LN affine values to BF16. Both paths use the same nontrivial '+
          'BF16-representable values held in FP32 LN parameters, avoiding different effective weights. '+
          'Initial job18527 used arbitrary FP32 affine values and stopped at a one-key tolerance '+
          'failure; it supplies no reported performance result. No kernels were changed to address it.', '',
          'Mixed, dense and one-key fixtures pass relative-L2 output <0.003 and update <0.025 in '+
          'both directions at all three lengths. These are tolerance comparisons, not bitwise '+
          'equivalence or a new independent FP64 qualification. The earlier independent FP64, '+
          'graph and sanitizer qualification remains linked in [README](README.md).', '',
          '**All-masked semantics differ:** our entry returns the input (zero update); Anthropic '+
          'uses uniform-mean-V attention and produces a nonzero update on these fixtures. This '+
          'difference is recorded explicitly, so the implementations are not interchangeable for '+
          'all masks. Timed inputs always contain valid keys.', '',
          'Evidence: [raw summary](anthropic-results.json), [provenance](anthropic-provenance.json), '+
          '`anthropic-{384,768,1024}-%s.json`, '%a.job+
          '[benchmark](anthropic_compare.py), [Slurm script](anthropic_compare.sbatch). '+
          'Reproduce this report with `python3 summarize_anthropic.py --job %s`. '%a.job+
          'No production dispatch change or inference SOL90 claim.', '']
(root/'ANTHROPIC_COMPARISON.md').write_text('\n'.join(lines))
for row in primary:
    print(row)
