from pathlib import Path
import sys
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'oc'))
from opt_core.kernels import triattn_surround_tma as bridge
from types import SimpleNamespace
ext=SimpleNamespace(stage=bridge._load().stage_broadcast_mask)
torch.manual_seed(926)
for B,N,S in [(1,768,768),(2,33,1024),(2,7,385),(1,2,1)]:
 storage=torch.rand(B,S*2,device='cuda')>.3
 storage[0].zero_()
 mask=storage[:,::2][:,None,None,None,:].expand(B,N,1,1,S)
 counts=torch.zeros(2,device='cuda',dtype=torch.int32)
 words,any_,kind,end,start,rs,re=ext.stage(mask,counts)
 torch.cuda.synchronize()
 base=mask[:,0,0,0].cpu();W=4*((S+127)//128+1)
 expected=torch.zeros(B,W,dtype=torch.int64)
 for wi in range(W):
  for lane in range(32):
   if wi*32+lane<S:expected[:,wi]|=base[:,wi*32+lane].long()<<lane
 expected=expected.to(torch.int32)
 assert torch.equal(any_.cpu(),expected)
 assert torch.equal(words.cpu(),expected[:,None,:].expand(B,N,W))
 for b in range(B):
  keys=base[b].nonzero().flatten();lo=int(keys[0])//32 if len(keys) else 0;hi=int(keys[-1])//32+1 if len(keys) else 0
  assert int(start[b])==lo and int(end[b])==hi
  assert (rs[b]==lo).all() and (re[b]==hi).all() and (kind[b]==(0 if hi else 2)).all()
 assert int(counts[0])==0 and int(counts[1])==int((~base.any(-1)).sum())*N
 print('PASS',B,N,S,flush=True)

# Dispatch boundaries and opt-out: nonbroadcast and unsupported lengths retain
# the original preparer, with no device counters touched by this optional hook.
import os
counts=torch.zeros(2,device='cuda',dtype=torch.int32)
nonbroadcast=torch.ones((1,7,1,1,768),device='cuda',dtype=torch.bool)
assert bridge.try_stage_mask(nonbroadcast,counts) is None
unsupported=torch.ones((1,1,1,1,385),device='cuda',dtype=torch.bool).expand(1,7,1,1,385)
assert bridge.try_stage_mask(unsupported,counts) is None
broadcast=torch.ones((1,1,1,1,768),device='cuda',dtype=torch.bool).expand(1,7,1,1,768)
os.environ['FPF_TRIATT_MASK_STAGE']='off'
assert bridge.try_stage_mask(broadcast,counts) is None
os.environ.pop('FPF_TRIATT_MASK_STAGE')
assert bridge.try_stage_mask(broadcast,counts) is not None
assert int(counts.sum())==0
print('PASS dispatch guards and default',flush=True)
