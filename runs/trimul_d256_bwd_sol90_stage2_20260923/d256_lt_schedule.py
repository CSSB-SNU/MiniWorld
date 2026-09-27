"""Explicit D256 Lt choices; preserve saved forward and fixed prefix dX."""
from types import SimpleNamespace
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from wide_lt_schedule import LtSchedule as Previous

class LtSchedule(Previous):
    def __init__(self,plan):
        products=plan.b1.prepare.original
        b=SimpleNamespace(wp=plan.leaves[6],wg=plan.leaves[5],proj=products.proj,gate=products.gate)
        proxy=SimpleNamespace(p=plan.p,b1=b,dx=plan.dx,f=plan.f,split_dwp=None)
        super().__init__(proxy)
        self.kernels.pop('dx').close();self.args.pop('dx');self.plan=plan

    def forward(self):
        plan=self.plan;p=plan.p;f=plan.f;out=f.output
        f.mask.copy_(plan.live_mask.reshape_as(f.mask))
        F.pack_into(f.w,*f.weights);f.front();self.run('fw0');self.run('fw1')
        out.norm.launch((out.grid,1,1),(out.threads,1,1),[out.np],out.smem)
        self.run('proj');self.run('gate');out.epi.launch((1056,1,1),(256,1,1),[out.ep],0)
        plan._has_forward=True
        return out.y

    def backward(self):
        plan=self.plan
        plan.b1.prepare();self.run('dn');self.run('dwp');self.run('dwg');plan.b1.ln()
        for name in ('bc0','bc1','bc2','bc3'):self.run(name)
        plan.b7.source_only();plan.dx()
        return plan.p.outputs
