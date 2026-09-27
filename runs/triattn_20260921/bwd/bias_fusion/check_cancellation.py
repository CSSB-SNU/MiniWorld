"""Independent zero-sum row adjoint: partial rounding must retain the error budget."""
import json, os
from pathlib import Path
import torch
from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.triangle_attention.triton import main as core
import hybrid
from native import ARTIFACT_ROOT

torch.manual_seed(92317)
L=64
def layout(x): return x.view(1,L,L,4,32).permute(0,3,1,2,4)
q=layout(torch.zeros(1,L,L,128,device='cuda',dtype=torch.bfloat16))
k=torch.zeros_like(q)
v=layout(torch.randn(1,1,L,128,device='cuda',dtype=torch.bfloat16).expand(1,L,L,128).clone())
b=torch.zeros(1,4,L,L,device='cuda',dtype=torch.bfloat16)
dy0=torch.zeros(1,L,L,128,device='cuda',dtype=torch.bfloat16)
# Group sums are 3, -7, 4, then zero. Each row coefficient is a power of two,
# so per-row BF16 dS has the same rounded mantissa and sums exactly to zero.
coeff=torch.tensor([1,1,1,0,-1,-2,-4,0,2,2,0,0]+[0]*(L-12),device='cuda')
dy0[0,:,:,0]=coeff[:,None]
dy=layout(dy0)
out,m=core._tri_attn_fwd(q,k,v,b,token_key(L))
base=core._tri_attn_bwd(q,k,v,b,m,out,dy,token_key(L))[-1].reshape(1,4,L,L,L).sum(2)
cand=hybrid.backward(q,k,v,b,m,out,dy,int(os.environ.get('FUSION_ROWS','4')))[-1]
qr,kr,vr,br=[x.double().requires_grad_() for x in (q,k,v,b)]
prob=torch.softmax(qr@kr.transpose(-2,-1)*(32**-.5)+br.unsqueeze(2),dim=-1)
ref=torch.autograd.grad(prob@vr,br,dy.double())[0]
baseline_rms=float((base.double()-ref).square().mean().sqrt())
candidate_rms=float((cand.double()-ref).square().mean().sqrt())
limit=baseline_rms*1.10+2e-7  # Exactly the existing zero-reference criterion.
report=dict(artifact=str(ARTIFACT_ROOT),reference_norm=float(ref.norm()),
            baseline_rms=baseline_rms,candidate_rms=candidate_rms,limit=limit,
            candidate_nonzero=int(torch.count_nonzero(cand)),passed=candidate_rms<=limit)
path=Path(__file__).parent/('cancellation-'+os.environ.get('SLURM_JOB_ID','local')+'.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
assert report['reference_norm']<1e-10,report
assert report['passed'],report
