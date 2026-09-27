"""Delay independent output weight gradients until source or dX is ready."""
import torch
from lt_contract import LtBmm

class LateOutputDW:
    def __init__(self,plan,phase='source',main_first=False,targets=('dwp','dwg')):
        assert plan.p.D==256
        self.plan=plan;self.phase=phase;self.main_first=main_first;self.targets=targets
        self.oldrun=plan.schedule.run;self.oldbackward=plan.backward
        self.oldsource=plan.b7.source_only;self.olddx=plan.dx.gemm_only
        self.workspace=torch.empty_like(plan.schedule.workspace)
        original=plan.schedule.kernels['dwp']
        self.dwp=LtBmm(original.a,original.b,original.out,self.workspace)
        self.dwp.index=original.index
        assert list(self.dwp.heuristics[self.dwp.index].algo.data)==list(original.heuristics[original.index].algo.data)
        self.side=torch.cuda.Stream();self.ready=torch.cuda.Event();self.done=torch.cuda.Event()
        self.launched=False

    def fork(self,main_op):
        assert not self.launched
        self.launched=True;main=torch.cuda.current_stream();self.ready.record(main)
        if self.main_first:main_op()
        with torch.cuda.stream(self.side):
            self.side.wait_event(self.ready)
            if 'dwp' in self.targets:self.dwp()
            if 'dwg' in self.targets:self.oldrun('dwg')
            self.done.record(self.side)
        if not self.main_first:main_op()

    def run(self,name):
        if name in self.targets:return
        if self.phase=='contract' and name=='bc0':self.fork(lambda:self.oldrun(name))
        else:self.oldrun(name)

    def source(self):
        if self.phase=='source':self.fork(self.oldsource)
        else:self.oldsource()

    def dx(self):
        if self.phase=='dx':self.fork(self.olddx)
        else:self.olddx()

    def backward(self):
        self.launched=False
        result=self.oldbackward();assert self.launched
        torch.cuda.current_stream().wait_event(self.done)
        return result

    def select(self,enabled):
        p=self.plan;p.schedule.run=self.run if enabled else self.oldrun
        p.b7.source_only=self.source if enabled else self.oldsource
        p.dx.gemm_only=self.dx if enabled else self.olddx
        p.backward=self.backward if enabled else self.oldbackward
