"""Alternate baseline and candidate timing; check both directions and mask edge cases."""
from pathlib import Path
import sys, os, copy, json, statistics, hashlib
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'oc'))
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
ending=os.environ.get('ENDING')=='1'
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
assert not any(k.startswith('FPF_TRIATT') for k in os.environ)
assert float((ref-x.float()).norm())>0
rows={r['id']:r for r in pf.cells()['rows']}
old={r['id']:r for r in json.loads((R/'cells_before_codex.json').read_text())['rows']}
current={k:copy.deepcopy(rows[k]['cfg']) for k in ('r07','r16')}
handoff=copy.deepcopy(current)
handoff['r07'].pop('j_fast',None);handoff['r16'].pop('starting_tile',None)
variants={'handoff':handoff,'previous_default':{k:old[k]['cfg'] for k in current},'final':current}

def select(name):
 for k,c in variants[name].items(): rows[k]['cfg']=copy.deepcopy(c)

results={name:dict(rounds_us=[]) for name in variants}
with torch.no_grad():
 expected=None
 for name in variants:
  select(name); y=call().clone()
  if expected is None: expected=y
  assert torch.equal(y,expected),name
  assert torch.isfinite(y).all()
  results[name]['rel_rms']=float((y.float()-ref).square().mean().sqrt()/ref.square().mean().sqrt())
  per=cupti(call,20)
  assert not any('flash_triattn' in k for k in per),per
  assert any(('triattn_m1_kernel' in k if a.length>=512 else k=='_fwd') for k in per),per
  results[name]['kernels']=per
 for round_ in range(3):
  order=list(variants) if round_%2==0 else list(reversed(variants))
  for name in order:
   select(name);t=graph_us(call);results[name]['rounds_us'].append(t)
   print('ROUND',ending,a.length,round_,name,t,flush=True)
 # Check changed inputs and dense/empty/irregular masks, no zero-init loophole.
 checks=[]
 for seed,mode in [(881,'dense'),(882,'empty'),(883,'irregular')]:
  torch.manual_seed(seed); x.normal_()
  if mode=='dense': mask.fill_(True)
  elif mode=='empty': mask.fill_(False)
  else: mask.copy_(torch.rand_like(mask,dtype=torch.float32)>.3)
  select('previous_default'); y0=call().clone()
  select('final'); y1=call().clone()
  assert torch.equal(y0,y1),(ending,seed,mode)
  assert torch.isfinite(y1).all(),(ending,seed,mode)
  checks.append(dict(seed=seed,mask=mode,bitwise_equal=True))
 for name in results: results[name]['op_us']=statistics.median(results[name]['rounds_us'])
 select('final')
so=list((R/'oc/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/prebuilt').glob('torch2.10*/triattn_m1_ext.so'))
res=dict(length=a.length,ending=ending,results=results,checks=checks,bitwise_equal=True,fallback=False,cfg=current,core_sha256=hashlib.sha256(so[0].read_bytes()).hexdigest())
Path(a.output).write_text(json.dumps(res,indent=2));print('RESULT',json.dumps(res),flush=True)
