from pathlib import Path
import sys,os,json,statistics
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'oc'))
os.environ['FPF_TRIATT_MASK_STAGE']='off'
ending=os.environ.get('ENDING')=='1'
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
from build import triattn_m1 as candidate
candidate._build()
def make_core(flags):
 def run(q,k,v,b,mask,scale):
  bb=b.reshape(1,1,4,a.length,a.length)
  y=candidate.triangle_attention_m1(q,k,v,bb,mask=mask,scale=scale,flags=flags)
  return y.reshape_as(q)
 return run
variants={'serving':'tier:triattn_native','rebuilt':make_core(0),'poly_half':make_core(33554432),'poly_all':make_core(67108864),'poly_quarter':make_core(134217728)}
def rms(t,r):return float((t.double()-r.double()).square().mean().sqrt())
results={k:dict(core_rounds=[],block_rounds=[]) for k in variants}
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 def core_call():return pf.core_attention(q,k,v,b,m5,core=KW['core'])
 KW['core']=variants['serving'];expected=call().clone();expected_c=core_call().clone()
 qq=q.reshape(1,a.length,4,a.length,32)[:,:1].double();kk=k.reshape_as(q).reshape(1,a.length,4,a.length,32)[:,:1].double();vv=v.reshape(1,a.length,4,a.length,32)[:,:1].double()
 logits=qq@kk.transpose(-1,-2)/32**.5+b.reshape(1,1,4,a.length,a.length).double()
 logits.masked_fill_(~m5[:,:1],-torch.inf);reference=logits.softmax(-1)@vv
 base_err=rms(expected_c.reshape(1,a.length,4,a.length,32)[:,:1],reference)
 for name,fn in variants.items():
  print('CHECK',name,flush=True)
  KW['core']=fn;y=call().clone();c=core_call().clone().reshape_as(expected_c);torch.cuda.synchronize()
  err=rms(c.reshape(1,a.length,4,a.length,32)[:,:1],reference)
  assert torch.isfinite(c).all() and err<=base_err*1.02+1e-8,(name,err,base_err)
  if name=='rebuilt':assert torch.equal(c,expected_c),(name,'rebuild changed baseline')
  kernels=cupti(call,5)
  expected_kernel='triattn_m1_kernel' if name=='serving' else 'triattn_core_candidate'
  assert any(expected_kernel in k for k in kernels) and not any('flash_triattn' in k for k in kernels),kernels
  results[name].update(core_rms_fp64=err,baseline_core_rms_fp64=base_err,core_delta_rms=rms(c,expected_c),block_delta_rms=rms(y,expected),core_equal=torch.equal(c,expected_c),kernels=kernels)
 for rd in range(3):
  for name in (list(variants) if rd%2==0 else list(reversed(variants))):
   KW['core']=variants[name]
   results[name]['core_rounds'].append(graph_us(core_call));results[name]['block_rounds'].append(graph_us(call))
   print('ROUND',rd,name,results[name]['core_rounds'][-1],results[name]['block_rounds'][-1],flush=True)
 for r in results.values():r.update(core_us=statistics.median(r['core_rounds']),block_us=statistics.median(r['block_rounds']))
 out=dict(length=a.length,ending=ending,results=results)
 Path(a.output).write_text(json.dumps(out,indent=2));print('RESULT',json.dumps(out),flush=True)
