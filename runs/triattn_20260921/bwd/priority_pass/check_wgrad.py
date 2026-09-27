"""Installed custom-op schema, fake output, compile, dispatch and result checks."""
import torch
from miniworld_engine.kernels.triangle_attention.cuda import wgrad_backward as wg
z=torch.randn(384*384,128,device='cuda',dtype=torch.bfloat16)
dy=[torch.randn_like(z) for _ in range(4)]
assert wg.can_use(dy,z)
for zz,dd in [(z.float(),dy),(z.half(),dy),(z[:4096], [d[:4096] for d in dy]),
              (z,dy[:3]),(z,[dy[0].float(),*dy[1:]])]:
    assert not wg.can_use(dd,zz)
result=torch.library.opcheck(wg._backward,(dy,z,2304))
assert all(v=='SUCCESS' for v in result.values()),result
out=wg.backward(dy,z);ref=[d.T@z for d in dy]
errors=[float((a.float()-b.float()).norm()/b.float().norm()) for a,b in zip(out,ref)]
assert max(errors)<.005,errors
print('WGRAD_OPCHECK_PASS',result,errors,flush=True)
