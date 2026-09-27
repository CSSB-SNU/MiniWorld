"""Launch independent output-gate dW when dNorm starts, joining before return."""
import torch

class GateDwOverlap:
    def __init__(self,plan):
        assert plan.p.n==384 and plan.p.D in (256,512)
        assert not getattr(plan,'joint_choice',None)
        assert plan.prefix_gate_dw.workspace.data_ptr()!=plan.schedule.workspace.data_ptr()
        self.plan=plan;self.original_backward=plan.backward;self.original_run=plan.schedule.run
        self.side=torch.cuda.Stream();self.ready=torch.cuda.Event();self.done=torch.cuda.Event()
        self.active=False;self.launched=False

    def run(self,name):
        if self.active and name=='dn':
            assert not self.launched
            self.launched=True
            main=torch.cuda.current_stream();self.ready.record(main)
            with torch.cuda.stream(self.side):
                self.side.wait_event(self.ready);self.original_run('dwg');self.done.record(self.side)
        if self.active and name=='dwg':return
        return self.original_run(name)

    def backward(self):
        self.active=True;self.launched=False
        try:result=self.original_backward()
        finally:self.active=False
        assert self.launched
        torch.cuda.current_stream().wait_event(self.done)
        return result

    def select(self,enable):
        self.plan.schedule.run=self.run if enable else self.original_run
        self.plan.backward=self.backward if enable else self.original_backward
