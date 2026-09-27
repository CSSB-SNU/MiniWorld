"""CUDA HFMA2 numerical/cost model only, not an attention kernel."""
from pathlib import Path
import importlib.util,json,os
HERE=Path(__file__).resolve().parent
source=(HERE/'bench.py').read_text().split('\nwith torch.no_grad():')[0]
exec(compile(source,str(HERE/'bench.py'),'exec'))
name='triattn_half2_exp_probe_v1'
spec=importlib.util.spec_from_file_location(name,HERE/('build_'+name)/(name+'.so'))
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
rows=[0,17,257,767];results=[]
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 shape=(1,a.length,4,a.length,32)
 for strength in (1.,2.,4.):
  qs=(q.float()*strength).bfloat16();ks=(k.float()*strength).bfloat16()
  actual=pf.core_attention(qs,ks,v,b,m5,core='tier:triattn_native').reshape(shape)[:,rows].double()
  qq=qs.reshape(shape)[:,rows].double();kk=ks.reshape(shape)[:,rows].double();vv=v.reshape(shape)[:,rows].double()
  logits=qq@kk.transpose(-1,-2)/32**.5+b.reshape(1,1,4,a.length,a.length).double()
  mask=m5.expand(1,a.length,1,1,a.length)[:,rows]
  logits=logits.masked_fill(~mask,-torch.inf)
  reference=logits.softmax(-1)@vv;base_error=rms(actual,reference)
  # A power-of-two normalization does not change BF16 rounding in normal
  # range. Keep the model safe at high strengths; actual core uses rescaling.
  log2=logits*1.4426950408889634
  shifted=(log2-torch.floor(log2.max(-1,keepdim=True).values)-64.).float().contiguous()
  for degree in (0,3,4):
   probability=probe.map(shifted,degree).bfloat16().double()
   answer=((probability@vv)/probability.sum(-1,keepdim=True)).bfloat16().double()
   error=rms(answer,reference)
   item=dict(strength=strength,degree=degree,rms=error,baseline_rms=base_error,ratio=error/base_error,finite=bool(torch.isfinite(answer).all()))
   results.append(item);print('MODEL',json.dumps(item),flush=True)
 costs={}
 for degree in (0,3,4):
  costs[str(degree)]=graph_us(lambda:probe.cost(q,degree,256))
  print('MICRO',degree,costs[str(degree)],flush=True)
Path(a.output).write_text(json.dumps(dict(model_only=True,ending=ending,rows=rows,results=results,register_probe_us=costs),indent=2)+'\n')
