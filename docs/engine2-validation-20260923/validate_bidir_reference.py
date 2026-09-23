from pathlib import Path
import torch,json,gc
from miniworld_engine import settings
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication as B
from miniworld_engine.integrations import trimul_h100
P=Path(__file__).resolve().parent
settings.configure(engine_backend='auto')
torch.backends.cuda.matmul.allow_tf32=False
records=[]
def rel(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
for L,D,zero_gamma in [(384,64,False),(384,128,False),(768,128,False),(384,128,True)]:
 torch.manual_seed(228)
 m=B(D,implementation='miniworld',p_drop=.25).cuda().bfloat16().train()
 ref=B(D,implementation='pytorch',p_drop=.25).cuda().float().train()
 with torch.no_grad():
  for name,p in m.named_parameters():
   if p.ndim==2:p.normal_(std=D**-.5)
   elif 'weight' in name:p.copy_(1+.1*torch.randn_like(p))
   else:p.normal_(std=.05)
 if zero_gamma:
  with torch.no_grad():
   m.ln_pair.weight[::3]=0;m.ln_out.weight[::3]=0
 ref.load_state_dict(m.state_dict())
 x=torch.randn(1,L,L,D,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 mask=torch.rand(1,L,device='cuda')>.2
 ds=(torch.rand(1,1,L,D,device='cuda')>.25).bfloat16()/0.75
 m._make_drop_row_scale=lambda pair,p:ds
 ref._make_drop_row_scale=lambda pair,p:ds.float()
 dy=torch.randn_like(x)
 assert trimul_h100.serves(m,x)
 y=m(x,mask);y.backward(dy)
 actual={'output':y.detach().float().cpu(),'dx':x.grad.float().cpu(),**{n:p.grad.float().cpu() for n,p in m.named_parameters()}}
 xx=x.detach().float().requires_grad_();rr=ref(xx,mask);rr.backward(dy.float())
 expected={'output':rr.detach().cpu(),'dx':xx.grad.cpu(),**{n:p.grad.cpu() for n,p in ref.named_parameters()}}
 errors={n:rel(actual[n],expected[n]) for n in actual}
 row=dict(length=L,width=D,zero_gamma=zero_gamma,dropout=.25,reference='PyTorch FP32, same BF16-quantized weights/input/dy and identical dropout mask',errors=errors,finite=all(torch.isfinite(v).all().item() for v in actual.values()))
 records.append(row);print(row,flush=True)
 assert row['finite'] and max(errors.values())<.03,row
 del m,ref,x,xx,y,rr,actual,expected;gc.collect();torch.cuda.empty_cache()
(P/'accuracy-bidir-fp32.json').write_text(json.dumps(records,indent=2))
