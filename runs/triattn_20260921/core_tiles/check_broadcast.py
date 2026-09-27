from pathlib import Path
import importlib.util,os,sys,json
import torch
R=Path(__file__).resolve().parents[1]
PKG=R/'oc/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg'
so=next((PKG/'prebuilt').glob('torch2.10.0+cu128-cpython-310-*/triattn_m1_ext.so'))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=module('original_m1_check',PKG/'cuda_b/triattn_m1.py');base._EXT=module('triattn_m1_ext',so)
from build_broadcast import triattn_m1 as candidate
candidate._build()
os.environ['FPF_TRIATT_MASK_STAGE']='off'
torch.manual_seed(926)
results=[]
flags=[int(f) for f in os.environ.get('CHECK_FLAGS','0,256').split(',')]
for L in [768,1024]:
 for layout in ['contiguous','strided']:
  B,N,H=2,7,4
  def operand():
   q=torch.randn(B,N,H,L,32,device='cuda',dtype=torch.bfloat16)
   if layout=='strided':q=q.transpose(2,3).contiguous().transpose(2,3)
   return q
  q,k,v=[operand() for _ in range(3)]
  bias=torch.randn(B,1,H,L,L,device='cuda',dtype=torch.float32)*.2
  for pattern in ['none','dense','ragged','prefix','empty_batch','empty','one_key','late_seed','large_logits','forced_safe']:
   storage=torch.ones(B,L*2,device='cuda',dtype=torch.bool);keys=storage[:,::2]
   if pattern=='ragged':keys.copy_(torch.rand(B,L,device='cuda')>.3)
   if pattern=='prefix':keys[:,:73]=False;keys[:,L-127:]=False
   if pattern=='empty_batch':keys[0]=False
   if pattern=='empty':keys[:]=False
   if pattern=='one_key':keys[:]=False;keys[:,L//2]=True
   mask=None if pattern=='none' else keys[:,None,None,None,:].expand(B,N,1,1,L)
   bb=bias.clone()
   if pattern=='late_seed':bb[...,:32]=-torch.inf
   qq=q*10 if pattern=='large_logits' else q
   if pattern=='forced_safe':os.environ['TRIATTN_M1_FORCE_SAFE']='1'
   else:os.environ.pop('TRIATTN_M1_FORCE_SAFE',None)
   os.environ['FPF_TRIATT_CORE_BROADCAST']='off'
   expected=base.triangle_attention_m1(qq,k,v,bb,mask,32**-.5)
   torch.cuda.synchronize()
   for fl in flags:
    actual=candidate.triangle_attention_m1(qq,k,v,bb,mask,32**-.5,flags=fl)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all(),(L,layout,pattern,fl,'nonfinite')
    assert torch.equal(actual,expected),(L,layout,pattern,fl,float((actual.float()-expected.float()).abs().max()))
    results.append([L,layout,pattern,fl])
   print('PASS',L,layout,pattern,flush=True)
print('PASS total',len(results),flush=True)
(R/'core_tiles/check_broadcast.json').write_text(json.dumps(results,indent=2))
