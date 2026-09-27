"""Independent output dW at wide short contraction/input-DW/DX boundaries."""
import torch
from lt_contract import LtBmm

class WideLateOutputDW:
    def __init__(self,plan,phase='source',main_first=False):
        assert plan.p.D in (384,512) and plan.p.n==384 and plan.split_dwp is None
        self.plan=plan;self.phase=phase;self.main_first=main_first
        self.oldrun=plan.schedule.run;self.oldbackward=plan.backward
        self.oldsource=plan.contract_gp;self.oldinput=plan.input_dw;self.oldln=plan.ln
        self.targets=('dwp',) if plan.joint_choice is not None else ('dwp','dwg')
        self.workspace=torch.empty_like(plan.schedule.workspace);self.ops=[]
        for name in self.targets:
            original=plan.schedule.kernels[name] if name=='dwp' else plan.prefix_gate_dw
            if name=='dwp':assert name in plan.schedule.selected
            op=LtBmm(original.a,original.b,original.out,self.workspace);op.index=original.index
            assert list(op.heuristics[op.index].algo.data)==list(original.heuristics[original.index].algo.data)
            self.ops.append(op)
        self.side=torch.cuda.Stream();self.ready=torch.cuda.Event();self.done=torch.cuda.Event();self.launched=False

    def fork(self,main_op):
        assert not self.launched
        self.launched=True;main=torch.cuda.current_stream();self.ready.record(main)
        if self.main_first:main_op()
        with torch.cuda.stream(self.side):
            self.side.wait_event(self.ready)
            for op in self.ops:op()
            self.done.record(self.side)
        if not self.main_first:main_op()

    def run(self,name):
        if name in self.targets:return
        if self.phase=='dx' and name=='dx':self.fork(lambda:self.oldrun(name))
        else:self.oldrun(name)
    def source(self):
        if self.phase=='source':self.fork(self.oldsource)
        else:self.oldsource()
    def input(self):
        if self.phase=='input':self.fork(self.oldinput)
        else:self.oldinput()
    def ln(self):
        if self.phase=='ln':self.fork(self.oldln)
        else:self.oldln()
    def backward(self):
        self.launched=False;result=self.oldbackward();assert self.launched
        torch.cuda.current_stream().wait_event(self.done)
        return result
    def select(self,enabled):
        p=self.plan;p.schedule.run=self.run if enabled else self.oldrun
        p.contract_gp=self.source if enabled else self.oldsource
        p.input_dw=self.input if enabled else self.oldinput
        p.ln=self.ln if enabled else self.oldln
        p.backward=self.backward if enabled else self.oldbackward
