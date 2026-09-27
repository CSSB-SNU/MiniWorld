"""Numerical screen only for FP16 transport of staged bias/scale.

Original BF16 Q/K/V and BF16 probability rounding remain. These FP64-model
logits do not simulate WGMMA rounding or CUDA synchronization and do not
qualify a kernel. Compare exact-bias controls as well as installed outputs.
"""
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=(HERE/'bench.py').read_text().split('\nwith torch.no_grad():')[0]
exec(compile(source,str(HERE/'bench.py'),'exec'))
rows=[0,17,257,767]
results=[]
with torch.no_grad():
    q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
    shape=(1,a.length,4,a.length,32)
    qq=q.reshape(shape)[:,rows].double()
    kk=k.reshape(shape)[:,rows].double()
    vv=v.reshape(shape)[:,rows].double()
    qk=qq@kk.transpose(-1,-2)/32**.5
    mask=m5.expand(1,a.length,1,1,a.length)[:,rows]
    for strength in [.25,1.,4.,16.]:
        bias=(b.float()*strength).reshape(1,1,4,a.length,a.length)
        actual=pf.core_attention(q,k,v,bias.reshape_as(b),m5,core='tier:triattn_native').reshape(shape)[:,rows].double()
        reference=(qk+bias.double()).masked_fill(~mask,-torch.inf).softmax(-1)@vv
        base_error=rms(actual,reference)
        for mode in ['exact_control','half_scaled','half_raw']:
            if mode=='half_scaled':
                staged=(bias*float(32**.5)).half().float().double()/32**.5
            elif mode=='half_raw':
                staged=bias.half().float().double()
            else:
                staged=bias.double()
            logits=(qk+staged).masked_fill(~mask,-torch.inf)
            seed=logits[...,:32].max(-1,keepdim=True).values
            probability=(torch.exp(logits-seed)*2.**-64).bfloat16().double()
            answer=((probability@vv)/probability.sum(-1,keepdim=True)).bfloat16().double()
            error=rms(answer,reference)
            item=dict(strength=strength,mode=mode,rms=error,baseline_rms=base_error,ratio=error/base_error,
                      finite=bool(torch.isfinite(answer).all()),bias_abs_max=float(bias.abs().max()))
            results.append(item)
            print('MODEL',json.dumps(item),flush=True)
Path(a.output).write_text(json.dumps(dict(model_only=True,ending=ending,rows=rows,results=results),indent=2)+'\n')
