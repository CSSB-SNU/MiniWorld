"""Numerical screening only: pre-scale Q in FP16 before QK, fixed exp shift.

This reports error, not CUDA kernel performance. It does not alter serving.
"""
from pathlib import Path
import math
source=(Path(__file__).resolve().parent/'bench.py').read_text().split('\nwith torch.no_grad():')[0]
exec(compile(source,str(Path(__file__).resolve().parent/'bench.py'),'exec'))
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 actual=pf.core_attention(q,k,v,b,m5,core='tier:triattn_native').reshape(1,a.length,4,a.length,32)[:,:1].double()
 qq=q.reshape(1,a.length,4,a.length,32)[:,:1].double()
 kk=k.reshape_as(q).reshape(1,a.length,4,a.length,32)[:,:1].double()
 vv=v.reshape_as(q).reshape(1,a.length,4,a.length,32)[:,:1].double()
 bias=b.reshape(1,1,4,a.length,a.length).double()
 logits=qq@kk.transpose(-1,-2)/32**.5+bias
 logits.masked_fill_(~m5[:,:1],-torch.inf)
 ref=logits.softmax(-1)@vv
 results={'baseline_rms':rms(actual,ref)}
 for case in ['fixed_shift','scaled_q_half']:
  if case=='fixed_shift':z=logits*math.log2(math.e)
  else:z=(qq*(32**-.5*math.log2(math.e))).half().double()@kk.half().double().transpose(-1,-2)+bias*math.log2(math.e)
  z.masked_fill_(~m5[:,:1],-torch.inf)
  p=torch.exp2(z-64).to(torch.bfloat16).double()
  y=((p@vv)/p.sum(-1,keepdim=True)).to(torch.bfloat16).double()
  results[case]=dict(rms=rms(y,ref),ratio=rms(y,ref)/results['baseline_rms'],delta=rms(y,actual),finite=bool(torch.isfinite(y).all()))
 Path(a.output).write_text(json.dumps(results,indent=2)+'\n')
 print('NUMERICAL MODEL ONLY',json.dumps(results),flush=True)
