"""Explicit dependency events and separate workspaces for independent dW."""
import torch
from lt_contract import LtBmm

class OverlapWeights:
    def __init__(self,plan,output=True,input=False):
        self.plan=plan;self.output=output;self.input=input
        self.side=torch.cuda.Stream(device=plan.p.x.device)
        self.ready=torch.cuda.Event();self.done=torch.cuda.Event();self.ops={}
        if plan.p.D!=256:
            self.workspace=torch.empty(64*1024*1024,device=plan.p.x.device,dtype=torch.uint8)
            for name in ('dwg','dwp'):
                if name in plan.schedule.selected:
                    original=plan.schedule.kernels[name]
                    op=LtBmm(original.a,original.b,original.out,self.workspace)
                    op.index=original.index
                    assert list(op.heuristics[op.index].algo.data)==list(original.heuristics[original.index].algo.data)
                    self.ops[name]=op
        assert not input or plan.p.D==512

    def weights(self):
        plan=self.plan;p=plan.p
        if p.D==256:
            torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp)
            torch.mm(p.dg.t(),p.xn.reshape(p.M,p.D),out=p.dwg)
        else:
            if plan.split_dwp is not None:plan.split_dwp()
            elif 'dwp' in self.ops:self.ops['dwp']()
            else:plan.schedule.original('dwp')
            if 'dwg' in self.ops:self.ops['dwg']()
            else:plan.schedule.original('dwg')

    def __call__(self):
        plan=self.plan;p=plan.p;b=plan.b1;main=torch.cuda.current_stream()
        if p.D==256:b.prepare()
        else:
            if p.D==512:
                b.norm.launch((b.grid,1,1),(b.threads,1,1),[b.np],b.smem);plan.schedule.run('proj')
            b.epi.launch((1056,1,1),(256,1,1),[b.ep],0)
        if self.output:
            self.ready.record(main)
            with torch.cuda.stream(self.side):
                self.side.wait_event(self.ready);self.weights();self.done.record(self.side)
        if p.D==256:torch.mm(p.tensors[7],b.wp,out=p.tensors[9])
        else:plan.schedule.run('dn')
        if not self.output:self.weights()
        if p.D==256:
            b.ln();d=p.D;h=2*d;ab=p.front.ab
            torch.bmm(p.dt[:d],ab[h:h+d],out=p.dl[:d])
            torch.bmm(p.dt[:d].transpose(-1,-2),ab[:d],out=p.dr[:d])
            torch.bmm(ab[h+d:],p.dt[d:].transpose(-1,-2),out=p.dl[d:])
            torch.bmm(ab[d:h],p.dt[d:],out=p.dr[d:])
        else:
            plan.ln()
            for name in ('bc0','bc1','bc2','bc3'):plan.schedule.run(name)
        if self.input:
            plan.saved_source.derivatives();self.ready.record(main)
            with torch.cuda.stream(self.side):
                self.side.wait_event(self.ready);plan.saved_source.matmul();self.done.record(self.side)
        else:plan.b7.source_only()
        plan.dx.copy_prefix();plan.dx.pack_weights()
        if p.D==256:plan.dx.gemm_only()
        else:plan.schedule.run('dx')
        if self.input:main.wait_event(self.done)
        plan.dx.reduce_only()
        if self.output and not self.input:main.wait_event(self.done)
        return p.outputs
