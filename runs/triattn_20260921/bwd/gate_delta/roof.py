"""Independent gate/delta checks and balanced graph timings."""
from pathlib import Path
import argparse,json,statistics
import torch,triton
from native import extension,ARTIFACT_ROOT
from miniworld_engine.kernels.triangle_attention.triton import main as core
from miniworld_engine.autotune.shape_key import token_key,pack
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--mask',choices=['none','mixed','one_key','all_masked'],default='mixed');ap.add_argument('--no-bench',action='store_true');ap.add_argument('--native-only',action='store_true');a=ap.parse_args()
L=a.length;torch.manual_seed(95331);ext=extension()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
from miniworld_engine.kernels.bias_only_attention.triton.gate_out import _dgrad_epilogue
M=L*L
dy,gate,out=[torch.randn(M,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
w=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.08

def baseline():
 dr,dg,aa=_dgrad_epilogue(dy,w,gate,out,shape_key=token_key(L))
 o=out.view(1,L,L,4,32).permute(0,3,1,2,4)
 d=dr.view(1,L,L,4,32).permute(0,3,1,2,4)
 delta=torch.empty((1,4,L,L),device='cuda',dtype=torch.float32)
 grid=lambda META:[triton.cdiv(L,META['BLOCK_M1']),4*L,1]
 core._attn_bwd_preprocess[grid](o,d,delta,*o.stride(),*d.stride(),4*L,1,L,32,shape_key=pack(token_key(L),HEAD_DIM=32),HEAD_DIM_PAD=32)
 return dr,dg,aa,delta.reshape(4,M)
def candidate():return tuple(ext.backward(dy,w,gate,out))
st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
 for _ in range(3):y=candidate()
torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g,stream=st):y=candidate()
g.replay();torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart();g.replay();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
