import torch
import candidate
torch.manual_seed(1)
for D in (64,128):
    L=64
    q,k,v=[torch.randn(1,L,L,4*D,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,D).permute(0,3,1,2,4) for _ in range(3)]
    b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.3
    for uniform in (False,True):
        if uniform:q.zero_();k.zero_();b.zero_()
        o,m=candidate.forward(q,k,v,b)
        scores=torch.einsum('bhijd,bhikd->bhijk',q.double(),k.double())*(D**-.5)+b.double()[:,:,None,:,:]
        mr=torch.logsumexp(scores,dim=-1)/torch.log(torch.tensor(2.))
        ref=torch.einsum('bhijk,bhikd->bhijd',scores.softmax(-1),v.double())
        print('DIAG',D,uniform,'lse shape',m.shape,'lse max',float((m-mr).abs().max()),'out rel',float((o-ref).norm()/ref.norm()),flush=True)
        print('OUT',o[0,0,0,0,:16].tolist(),'REF',ref[0,0,0,0,:16].tolist(),flush=True)
