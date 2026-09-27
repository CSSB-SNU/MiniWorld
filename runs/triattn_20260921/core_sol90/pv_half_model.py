"""Bounded FP16-PV error model; NOT a CUDA accuracy/performance qualification."""
from pathlib import Path
import sys,os,json
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'oc'))
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
records=[]
def rms(x,y):return float((x.double()-y.double()).square().mean().sqrt())
with torch.no_grad():
 for ending in (False,True):
  q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
  q=q.reshape(1,a.length,4,a.length,32);k=k.reshape_as(q);v=v.reshape_as(q)
  for pattern in ('periodic','constant_v','positive_v','all','one','none'):
   v_original=v
   if pattern=='constant_v':v=torch.ones_like(v)
   if pattern=='positive_v':v=(v.float()*.05+1).bfloat16()
   mask=torch.ones(1,a.length,device='cuda',dtype=torch.bool)
   if pattern=='periodic':mask[:,::7]=False
   if pattern=='one':mask[:]=False;mask[:,371]=True
   if pattern=='none':mask[:]=False
   m5=mask[:,None,None,None,:].expand(-1,a.length,-1,-1,-1)
   native=pf.core_attention(q,k,v,b,m5,core='tier:triattn_native').reshape_as(q)
   ix=[0,7,511,767]
   qq=q[:,ix].double();kk=k[:,ix].double();vv=v[:,ix].double()
   logits=qq@kk.transpose(-1,-2)/32**.5+b.reshape(1,1,4,a.length,a.length).double()
   logits.masked_fill_(~m5[:,:1],-torch.inf)
   ref=torch.nan_to_num(logits.softmax(-1))@vv
   if pattern=='none':ref=vv.mean(-2,keepdim=True).expand_as(qq)
   base=rms(native[:,ix],ref)
   fl=logits.float()
   # Model only first valid 32-key tile seeded rows. Empty initial tiles request SAFE.
   seed=fl[...,:32].amax(-1,keepdim=True)
   valid=seed.isfinite();seed=torch.where(valid,seed,torch.zeros_like(seed))
   p=(fl-seed).exp().half();vf=v[:,ix].half()
   for splits in (1,2):
    acc=[torch.zeros_like(qq,dtype=torch.float32) for _ in range(splits)]
    den=torch.zeros_like(seed)
    for begin in range(0,a.length,16):
     pp=p[...,begin:begin+16].float()
     part=pp@vf[...,begin:begin+16,:].float()
     j=(begin//16)%splits
     acc[j]=(acc[j]+part).half().float()
     den+=pp.sum(-1,keepdim=True)
     if (begin+16)%256==0 and begin+16<a.length:
      # Scale accumulators and future P by the same exact power of two.
      scale=torch.exp2(torch.floor(torch.log2(den.clamp_min(2**-14)))-4)
      scale=torch.where(valid,scale,torch.ones_like(scale))
      acc=[(a_/scale).half().float() for a_ in acc]
      den=den/scale;p[...,begin+16:]=(p[...,begin+16:].float()/scale).half()
    out=torch.nan_to_num(sum(acc)/den).bfloat16()
    # A failed seed explicitly means SAFE, whose output is native here.
    out=torch.where(valid,out,native[:,ix])
    error=rms(out,ref)
    row={'ending':ending,'pattern':pattern,'splits':splits,'baseline_rms':base,'model_rms':error,
      'ratio':error/max(base,1e-30),'seed_safe_rows':int((~valid).sum()),
      'nonfinite':int((~out.isfinite()).sum()),'v_conversion_exact':torch.equal(vf.float(),v[:,ix].float()),
      'qualified':False,'hot_exercised':pattern not in ('one','none')}
    records.append(row);print(json.dumps(row),flush=True)
    # Reconstruct P for the next split setting.
    p=(fl-seed).exp().half()
   v=v_original
Path(a.output).write_text(json.dumps(records,indent=2))
