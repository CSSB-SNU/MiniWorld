"""Record installed L1024 results only after actual serving and NCU complete."""
from pathlib import Path
import csv,difflib,hashlib,json
P=Path(__file__).resolve().parent;R=P.parent
D=R/'oc/opt_core/kernels/triattn_core_broadcast'
sha='9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225'
assert hashlib.sha256((D/'triattn_broadcast.so').read_bytes()).hexdigest()==sha
manifest=json.loads((D/'manifest.json').read_text())
assert all(hashlib.sha256((D/f).read_bytes()).hexdigest()==h for f,h in manifest['sha256'].items())
finish=(P/'finish-q1024-14862.log').read_text()
assert '17/17 cases pass -- OK' in finish and 'Traceback' not in finish
stage=(P/'stage-q1024-14856.log').read_text()
assert stage.count('PASS total 5')==3 and stage.count('ERROR SUMMARY: 0 errors')==2
assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in stage
profiles={}
for length in (768,1024):
 flag=1073741824 if length==768 else 536870912
 for direction in (0,1):
  d=json.loads((P/('serving-q1024-L%d-e%d.json'%(length,direction))).read_text())
  assert d['length']==length and d['ending']==bool(direction)
  assert all(v['bitwise_equal'] for v in d['results'].values())
  kernels=d['results']['broadcast']['kernels']
  assert any('ta_core_broadcast::triattn_m1_kernel' in k and 'Traits<%d>'%flag in k for k in kernels)
  assert not any('flash_triattn' in k for k in kernels)
 rows=list(csv.DictReader((P/('installed-q1024-L%d-profile.csv'%length)).open()))
 matches=[row for row in rows[1:] if 'Traits<%d>'%flag in row['Kernel Name']]
 assert len(matches)==1
 row=matches[0]
 def value(key):return float(row[key].replace(',',''))
 time_factor={'ns':.001,'us':1.,'ms':1000.,'s':1e6}[rows[0]['gpu__time_duration.sum']]
 def mbytes(key):
  factor={'byte':1e-6,'Kbyte':.001,'Mbyte':1.,'Gbyte':1000.}[rows[0][key]]
  return value(key)*factor
 assert rows[0]['lts__t_sectors.sum']=='sector'
 profiles[str(length)]={'kernel_us':value('gpu__time_duration.sum')*time_factor,
  'sm_sol_pct':value('sm__throughput.avg.pct_of_peak_sustained_elapsed'),
  'tensor_pct':value('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),
  'occupancy_pct':value('sm__warps_active.avg.pct_of_peak_sustained_active'),
  'dram_MB':mbytes('dram__bytes_read.sum')+mbytes('dram__bytes_write.sum'),
  'l2_GB':value('lts__t_sectors.sum')*32/1e9,
  'shared_lsu_pct':value('l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed')}
r=json.loads((P/'q1024-results.json').read_text())
r.update(installed=True,installed_sha256=sha,installed_profiles=profiles,
 final_package_job=14856,finish_job=14862,sanitizers={'cases_each':5,'racecheck_hazards':0,'synccheck_errors':0,'memcheck_errors':0},
 installed_serving={'directions':[0,1],'lengths':[768,1024],'masks_each':5,'full_core_block_bitwise':True,'no_flash_fallback':True,'shipped_tests_passed':17,'shipped_tests_total':17,'expected_vectors_regenerated':False})
(P/'q1024-results.json').write_text(json.dumps(r,indent=2)+'\n')
old=json.loads((P/'q-three-results.json').read_text())
old.update(installed=False,previously_installed_by_job=14836,superseded_by_sha256=sha)
(P/'q-three-results.json').write_text(json.dumps(old,indent=2)+'\n')
for file,old,new in [('verify_fast_serving.py',"os.environ.get('VERIFY_FLAG1024','0')","os.environ.get('VERIFY_FLAG1024','536870912')"),('profile_installed.sbatch','PROFILE_NAME:-installed-qthree','PROFILE_NAME:-installed-q1024-L768'),('verify_fast.sbatch','VERIFY_PREFIX:-serving-qthree','VERIFY_PREFIX:-serving-q1024')]:
 path=P/file;s=path.read_text();assert old in s or new in s;path.write_text(s.replace(old,new))
patch=[]
for name in ('build_source.py','csrc/m1/m1_binding.cu','csrc/m1/triattn_m1_sm90.cuh'):
 target=str((D/name).relative_to(R.parents[1]))
 patch.extend(difflib.unified_diff((P/'before_fast_install'/name).read_text().splitlines(True),(D/name).read_text().splitlines(True),fromfile='a/'+target,tofile='b/'+target))
(R/'codex_core_sol90.patch').write_text(''.join(patch))
a,b=profiles['768'],profiles['1024']
text='''# L1024 CUDA/TMA core specialization

2026-09-22, H10080GB, C128/H4/D32 BF16. **SOL90 remains unachieved.**

Job14862 installs a separate hot536870912 for squareN=S1024,H4 and the usual
scale. It uses the qualified twelve-register Q cache, independent K/V TMA
producers, constant descriptors, N40 PV/denominator fusion and exact grouped
FFMA/EX2. Row clamps handle the final partial group of three pair rows.
All floating-point operations and probability rounding retain their order.
Other scales retain hot0. L768 retains hot1073741824; SAFE remains1024.

Job14851 used16 balanced paired graph rounds, cloning each captured output
before another graph could overwrite it. Full core/block outputs are bitwise
equal. Percentages use paired ratios, rather than ratios of timing medians.

| Direction | Previous core us | Current core us | Paired core reduction | Paired block reduction |
|---|---:|---:|---:|---:|
| starting | 1966.960 | 1879.503 | 4.163%% | 2.444%% |
| ending | 1989.822 | 1905.414 | 3.971%% | 2.494%% |

The baseline is the preceding4be package, saved in
`core_sol90/before_q1024_install/`. L768's actual machine words are unchanged;
the preceding twelve-register Q change's0.45%% gain remains in that path.

Final package14856 passes20 fullN1024/B2,20 fullN768/B2,40 generic and2
custom-scale bitwise cases. Its all-four-flags machine words exactly match
the measured prototype; hot0, SAFE1024 and L768hot1073741824 exactly match4be.
Both hot kernels compile with128 initial registers, zero spills and no WGMMA
serialization warning; C7519 inserted fences remain.

FullN1024/B1 strided racecheck, synccheck and memcheck each pass five patterns
(dense, prefix, empty batch, late seed, forced SAFE) with zero hazards/errors.
Installed job14862 confirms both directions, both lengths, five masks,
bitwise default/opt-out outputs, expected native flags and no flash fallback.
All17 shipped tests pass. Expected vectors and the separate generic M1 binary
were not regenerated.

Installed NCU job14862 uses separate kernel profiling runs, not paired graph
timing. Do not infer incremental speedups from different NCU jobs.

| Length | Kernel us | SM SOL | Tensor | Occupancy |
|---|---:|---:|---:|---:|
| 768 | %.3f | %.3f%% | %.3f%% | %.3f%% |
| 1024 | %.3f | %.3f%% | %.3f%% | %.3f%% |

Installed SHA256: `%s`.
Measurements and validation: [q1024-results.json](core_sol90/q1024-results.json),
`core_sol90/serving-q1024-*.json` and `installed-q1024-L*-profile.*`.
Rebuild commands and cumulative history: [CORE_SOL90_REPORT.md](CORE_SOL90_REPORT.md).
'''%(a['kernel_us'],a['sm_sol_pct'],a['tensor_pct'],a['occupancy_pct'],b['kernel_us'],b['sm_sol_pct'],b['tensor_pct'],b['occupancy_pct'],sha)
(R/'Q1024_REPORT.md').write_text(text)
report=R/'CORE_SOL90_REPORT.md';s=report.read_text()
s=s.replace('**Current installed NCU job14836:','**Previous installed NCU job14836:')
s=s.replace('Installed SHA256:\n`4be5b7cdd291150436b9b048318b6cac7977b0db82b9a4f748520d398cfb4c32`.','Installed SHA256:\n`'+sha+'`.')
s=s.replace('immediately preceding package is saved in `core_sol90/before_qthree_install/`','immediately preceding package is saved in `core_sol90/before_q1024_install/`')
notice='\n\n> **Latest installation14862:** L1024 core is4.163%%/3.971%% faster in paired starting/ending measurements. Installed NCU isL768 %.3fus/SM%.3f%% andL1024 %.3fus/SM%.3f%%. L768 machine words are unchanged from4be. [Q1024_REPORT.md](Q1024_REPORT.md) records the final package,82 bitwise cases, full-shape sanitizers and serving verification. **SOL90 remains unachieved.**\n'%(a['kernel_us'],a['sm_sol_pct'],b['kernel_us'],b['sm_sol_pct'])
if 'Latest installation' not in s:s=s.replace('\n\n2026-09-22,',notice+'\n2026-09-22,',1)
s=s.replace('For square N=S768, H4 and the usual scale,','For square N=S768 or1024, H4 and the usual scale,')
s=s.replace('The native binding selects hot flag1073741824 only for the qualifying shape\nand scale. L1024 and other scales keep hot flag0. SAFE uses flag1024.','The native binding selects hot1073741824 for the qualifying L768 shape/scale,\nand hot536870912 for L1024. Other shapes/scales keep hot0. SAFE uses1024.')
report.write_text(s)
handoff=R/'HANDOFF.md';s=handoff.read_text().replace('2026-09-22 latest core follow-up:','Previous core follow-up14836:')
if 'Latest installation' not in s:s=s.replace('\n\n> ',notice+'\n> ',1)
handoff.write_text(s)
status=P/'STATUS.md';s=status.read_text();start=s.index('Installed SHA');end=s.index('SOL90 remains unachieved. See',start)
s=s[:start]+'''Installed SHA%s.
Latest job14862 extends the exact pipeline to L1024hot536870912: paired core
reductions4.163%%/3.971%% and block2.444%%/2.494%%, starting/ending (14851).
L768hot1073741824 machine words unchanged from4be; its preceding0.45%% gain
remains. Final14856:82 bitwise cases and three fullN1024 strided sanitizers,
five cases each, zero hazards/errors. Installed14862: both directions/lengths,
five masks, bitwise default/optout, expected native flags,17/17 shipped tests.
NCU14862: L768%.3fus/SM%.6f%%, L1024%.3fus/SM%.6f%%.
q1024-results.json and ../Q1024_REPORT.md record the current result.
'''%(sha,a['kernel_us'],a['sm_sol_pct'],b['kernel_us'],b['sm_sol_pct'])+s[end:]
s+='\n14856/14862 completed successfully; package9365 installed and actual serving/profiles verified. Documentation, default verifier flags/prefixes and cumulative patch updated. SOL90 remains ACTIVE and unachieved.\n'
status.write_text(s)
print(json.dumps({'installed_sha256':sha,'profiles':profiles,'sol90_achieved':False},indent=2))
