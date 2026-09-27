"""Same-process cell A/B; nonzero weights, exact comparison, real kernel census."""
from pathlib import Path
import sys, json, copy, statistics, os, hashlib
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'oc'))
# Reuse the established setup/reference/graph harness without its final run.
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
assert not any(k.startswith('FPF_TRIATT') for k in os.environ), 'unset experiment overrides'
assert float((ref-x.float()).norm()) > 0
rows={r['id']:r for r in pf.cells()['rows']}
old={r['id']:r for r in json.loads((R/'cells_before_codex.json').read_text())['rows']}
best={k:copy.deepcopy(rows[k]['cfg']) for k in ('r07','r16')}
best['r07'].pop('j_fast',None); best['r16'].pop('starting_tile',None)
variants=[('previous_default',{k:old[k]['cfg'] for k in best}),('wired',best)]
# More contiguous j per CTA: reduce scattering while retaining the same BM and arithmetic.
if os.environ.get('TUNE_TILES')=='1':
 for bi,bj in [(4,32),(2,64),(1,128),(16,8)]:
  c=copy.deepcopy(best); c['r07'].update(BI=bi,BJ=bj); variants.append((f'pro_{bi}x{bj}',c))
 for bi,bj in [(4,16),(2,32),(1,64),(16,4)]:
  c=copy.deepcopy(best); c['r16'].update(BI=bi,BJ=bj); variants.append((f'epi_{bi}x{bj}',c))
if os.environ.get('TUNE_SPLIT') == '1':
 import split_prologue_interleaved as split_prologue
 PRO = pf._carried('fpf_triatt_pro','prologue')
 original_kernel = PRO._triatt_prologue_kernel
 class SplitLaunch:
  def __getitem__(self, grid):
   def launch(*args, **kwargs):
    return split_prologue._triatt_prologue_kernel[(grid[0]*grid[1]*4*kwargs['H']*kwargs['D']//kwargs['BN'],)](*args, **kwargs)
   return launch
 variants=variants[:2]
 for bi,bj,nw,bn in [(4,16,4,64),(1,64,4,64),(4,32,4,64),(1,128,4,64),(4,16,4,128),(1,64,4,128),(1,128,8,128)]:
  c=copy.deepcopy(best); c['r07'].update(BI=bi,BJ=bj,num_warps=nw,num_stages=1,BN=bn,maxnreg=0)
  variants.append((f'split_{bi}x{bj}_w{nw}_n{bn}',c))
results=[]
with torch.no_grad():
 expected=None
 for tag,cfgs in variants:
  for k,c in cfgs.items(): rows[k]['cfg']=copy.deepcopy(c)
  if os.environ.get('TUNE_SPLIT') == '1': PRO._triatt_prologue_kernel = SplitLaunch() if tag.startswith('split_') else original_kernel
  y=call().clone(); torch.cuda.synchronize()
  if expected is None: expected=y
  eq=torch.equal(y,expected)
  err=float((y.float()-ref).square().mean().sqrt()/ref.square().mean().sqrt())
  if not eq:
   print('REJECT',tag,'non-bitwise',err,flush=True); continue
  per=cupti(call,10)
  names=list(per)
  assert not any('flash_triattn' in k for k in names), names
  assert any(('triattn_m1_kernel' in k if a.length>=512 else k=='_fwd') for k in names),names
  times=[graph_us(call) for _ in range(3)]
  r=dict(tag=tag,length=a.length,bitwise_equal=eq,rel_rms=err,op_us=statistics.median(times),rounds_us=times,prologue_us=sum(v for k,v in per.items() if 'prologue' in k),epilogue_us=sum(v for k,v in per.items() if 'epilogue' in k),kernels=per,cfg=cfgs)
  results.append(r); print('RESULT',json.dumps(r),flush=True)
  Path(a.output).write_text(json.dumps(results,indent=2))
 for k,c in best.items(): rows[k]['cfg']=c
print('PASS: bitwise equality, nonzero reference, real core checked',flush=True)
