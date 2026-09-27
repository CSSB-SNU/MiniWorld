"""Validate and time the projected block including preparation and epilogue."""
from pathlib import Path
import importlib.util,sys,os,json,statistics,hashlib
HERE=Path(__file__).resolve().parent;R=HERE.parents[1]
sys.path.insert(0,str(R/'oc'))
ending=os.environ.get('ENDING')=='1';tag=os.environ.get('PROJECTED_VARIANT','p64')
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}').replace('e.name[:40]','e.name')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
assert a.length==768
name='triattn_projected_'+tag
spec=importlib.util.spec_from_file_location(name,HERE/(name+'.so'))
ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
cache=W._fpf_cache['triatt_pro']
wkv=torch.cat((cache['wqkvg'][128:256].reshape(4,32,128),cache['wqkvg'][256:384].reshape(4,32,128)),dim=1).contiguous()
norm=torch.empty_like(x[0]);gate=torch.empty_like(norm)
bias=torch.empty(4,a.length,a.length,device='cuda',dtype=torch.float32)
def prep():
 ext.prepare(x[0],cache['wqkvg'],cache['wb'],cache['lnw'],cache['lnb'],norm,norm,norm,gate,bias,cache['eps'],ending)
def fused_core():return ext.forward(norm,cache['wqkvg'],wkv,bias,m5,32**-.5)
def fused_block():
 prep();o=fused_core()
 return pf.epilogue(o.reshape(1,a.length,4,a.length,32),gate.unsqueeze(0),W,z=x,ending=ending,residual=True,impl='fpf',out=buf)
def rms(t,r):return float((t.double()-r.double()).square().mean().sqrt())
checks=[]
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 def serving_core():return pf.core_attention(q,k,v,b,m5,core='tier:triattn_native')
 for pattern in os.environ.get('CHECK_PATTERNS','holes,all,late,one,empty').split(','):
  print('RUN',pattern,'baseline',flush=True)
  mask.fill_(True)
  if pattern=='holes':mask[:,::7]=False
  elif pattern=='late':mask[:,:128]=False
  elif pattern=='one':mask.fill_(False);mask[:,-1]=True
  elif pattern=='empty':mask.fill_(False)
  elif pattern!='all':raise ValueError(pattern)
  expected_core=serving_core().clone();expected_block=call().clone()
  print('RUN',pattern,'projected core',flush=True)
  prep();got_core=fused_core().reshape_as(expected_core).clone();torch.cuda.synchronize()
  print('RUN',pattern,'projected block',flush=True)
  got_block=fused_block().clone();torch.cuda.synchronize()
  qq=q.reshape(1,768,4,768,32)[:,:1].double();kk=k.reshape_as(q).reshape(1,768,4,768,32)[:,:1].double();vv=v.reshape_as(q).reshape(1,768,4,768,32)[:,:1].double()
  if pattern=='empty':reference=vv.mean(-2,keepdim=True).expand_as(qq)
  else:
   logits=qq@kk.transpose(-1,-2)/32**.5+b.reshape(1,1,4,768,768).double()
   logits.masked_fill_(~m5[:,:1],-torch.inf);reference=logits.softmax(-1)@vv
  base_error=rms(expected_core.reshape(1,768,4,768,32)[:,:1],reference)
  error=rms(got_core.reshape(1,768,4,768,32)[:,:1],reference)
  row=dict(pattern=pattern,core_bitwise=torch.equal(got_core,expected_core),block_bitwise=torch.equal(got_block,expected_block),core_delta=rms(got_core,expected_core),block_delta=rms(got_block,expected_block),baseline_rms=base_error,rms=error)
  print('CHECK',json.dumps(row),flush=True);checks.append(row)
  Path(a.output).write_text(json.dumps(dict(checks=checks),indent=2))
  assert torch.isfinite(got_core).all() and error<=base_error*1.05+1e-8,row
 mask.fill_(True);mask[:,::7]=False;prep()
 variants={'serving':(serving_core,call),'projected':(fused_core,fused_block)}
 results={key:dict(core_rounds=[],block_rounds=[]) for key in variants}
 for key,(cf,bf) in variants.items():
  kernels=cupti(bf,5);results[key]['kernels']=kernels
  required='ta_core_broadcast::triattn_m1_kernel' if key=='serving' else 'triattn_projected_fused_'+tag+'::attention'
  assert any(required in n for n in kernels) and not any('flash_triattn' in n for n in kernels),kernels
 for rd in range(5):
  for key in (list(variants) if rd%2==0 else list(reversed(variants))):
   cf,bf=variants[key]
   results[key]['core_rounds'].append(graph_us(cf));results[key]['block_rounds'].append(graph_us(bf))
   print('ROUND',rd,key,results[key]['core_rounds'][-1],results[key]['block_rounds'][-1],flush=True)
 for d in results.values():d.update(core_us=statistics.median(d['core_rounds']),block_us=statistics.median(d['block_rounds']))
 result=dict(ending=ending,checks=checks,results=results,sha256=hashlib.sha256((HERE/(name+'.so')).read_bytes()).hexdigest(),scope='isolated prototype, not installed')
 Path(a.output).write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
