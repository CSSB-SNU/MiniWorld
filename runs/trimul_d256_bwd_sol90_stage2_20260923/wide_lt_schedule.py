"""Explicit cuBLASLt choices for the existing wide training GEMMs."""
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from lt_contract import LtBmm


class LtSchedule:
    def __init__(self,plan):
        self.plan=plan;p=plan.p;b=plan.b1;d=p.D;h=2*d;ab=p.front.ab
        self.workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
        self.args=dict(proj=(p.tensors[6],b.wp.t(),b.proj),gate=(p.xn.reshape(p.M,d),b.wg.t(),b.gate),
            dn=(p.tensors[7],b.wp,p.tensors[9]),dwg=(p.dg.t(),p.xn.reshape(p.M,d),p.dwg),
            dx=(plan.dx.input.t(),plan.dx.weights,p.tensors[10]),
            fw0=(ab[:d],ab[h:h+d].transpose(-1,-2),plan.f.tri[:d]),
            fw1=(ab[d:h].transpose(-1,-2),ab[h+d:],plan.f.tri[d:]),
            bc0=(p.dt[:d],ab[h:h+d],p.dl[:d]),bc1=(p.dt[:d].transpose(-1,-2),ab[:d],p.dr[:d]),
            bc2=(ab[h+d:],p.dt[d:].transpose(-1,-2),p.dl[d:]),bc3=(ab[d:h],p.dt[d:],p.dr[d:]))
        if plan.split_dwp is None:self.args['dwp']=(p.tensors[7].t(),p.tensors[6],p.dwp)
        self.kernels={};self.selected={}
        for name,operands in self.args.items():
            self.kernels[name]=LtBmm(*[t.unsqueeze(0) if t.ndim==2 else t for t in operands],self.workspace)

    def original(self,name):
        a,b,c=self.args[name]
        if a.ndim==2:torch.mm(a,b,out=c)
        else:torch.bmm(a,b,out=c)

    def run(self,name):
        if name in self.selected:self.kernels[name]()
        else:self.original(name)

    def forward(self):
        plan=self.plan;p=plan.p;f=plan.f;out=plan.product_output
        f.mask.copy_(plan.mask);p.mask.copy_(plan.mask.reshape_as(p.mask))
        F.pack_into(f.w,*f.weights);f.front();self.run('fw0');self.run('fw1')
        out.norm.launch((out.grid,1,1),(out.threads,1,1),[out.np],out.smem)
        self.run('proj');self.run('gate');out.epi.launch((1056,1,1),(256,1,1),[out.ep],0)
        return out.y

    def backward(self):
        plan=self.plan;p=plan.p;b=plan.b1
        if p.D==512:
            b.norm.launch((b.grid,1,1),(b.threads,1,1),[b.np],b.smem);self.run('proj')
        b.epi.launch((1056,1,1),(256,1,1),[b.ep],0);self.run('dn')
        if plan.split_dwp is not None:plan.split_dwp()
        else:self.run('dwp')
        self.run('dwg');plan.ln()
        for name in ('bc0','bc1','bc2','bc3'):self.run(name)
        plan.b7.source_only();plan.dx.copy_prefix();plan.dx.pack_weights();self.run('dx');plan.dx.reduce_only()
        return p.outputs

    def __call__(self):return self.forward(),self.backward()
