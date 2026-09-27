"""Independent dQ checks and balanced graph timings."""
from pathlib import Path
import argparse,json,statistics
import torch,triton
from native import extension,ARTIFACT_ROOT
from miniworld_engine.kernels.triangle_attention.triton import main as core
from miniworld_engine.autotune.shape_key import token_key,pack
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--mask',choices=['none','mixed','one_key','all_masked'],default='mixed');ap.add_argument('--no-bench',action='store_true');ap.add_argument('--native-only',action='store_true');a=ap.parse_args()
L=a.length;torch.manual_seed(95331);ext=extension()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
q,k,v,dy=[torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,32).permute(0,3,1,2,4) for _ in range(4)]
b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
if a.mask=='mixed':b[...,::3]=torch.finfo(b.dtype).min
if a.mask in ('one_key','all_masked'):
 b.fill_(torch.finfo(b.dtype).min)
 if a.mask=='one_key':b[...,7]=0
out,m=core._tri_attn_fwd(q,k,v,b,token_key(L))
delta=torch.empty((1,4,L,L),device='cuda',dtype=torch.float32)
grid=lambda META:[triton.cdiv(L,META['BLOCK_M1']),4*L,1]
core._attn_bwd_preprocess[grid](out,dy,delta,*out.stride(),*dy.stride(),4*L,1,L,32,shape_key=pack(token_key(L),HEAD_DIM=32),HEAD_DIM_PAD=32)
def baseline():
 dq=torch.empty((1,L,L,128),device='cuda',dtype=torch.bfloat16).view(1,L,L,4,32).permute(0,3,1,2,4)
 grid=lambda META:[triton.cdiv(L,META['BLOCK_M1']),1,4*L]
 core._attn_bwd_dq[grid](q,k,v,b,32**-.5,dy,dq,m,delta,*q.stride(),*dq.stride(),*dy.stride(),*b.stride(),L,4*L,32,HEAD_DIM_PAD=32,shape_key=pack(token_key(L),HEAD_DIM=32))
 return dq
def candidate():return ext.backward(q,k,v,b,m,delta,dy)
st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
 for _ in range(3):y=candidate()
torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g,stream=st):y=candidate()
g.replay();torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart();g.replay();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
