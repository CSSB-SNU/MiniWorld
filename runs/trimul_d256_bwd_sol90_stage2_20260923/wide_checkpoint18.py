"""Overlap independent input dW and dX for D512 L768 with explicit graph events."""
import torch
from wide_checkpoint17 import Training as Previous,configuration as previous_configuration

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        self.input_overlap=(self.p.D,self.p.n)==(512,768)
        if self.input_overlap:
            assert self.input_dw.workspace.data_ptr()!=self.schedule.workspace.data_ptr()
            self.input_side=torch.cuda.Stream(device=self.p.x.device)
            self.input_ready=torch.cuda.Event();self.input_done=torch.cuda.Event()

    def backward(self):
        if not self.input_overlap:return super().backward()
        p=self.p;b=self.b1;s=self.schedule;main=torch.cuda.current_stream()
        self.delta_backward.launch();self.delta_projection()
        b.epi.launch((1056,1,1),(256,1,1),[b.ep],0);s.run('dn')
        if self.split_dwp is not None:self.split_dwp()
        else:s.run('dwp')
        s.run('dwg');self.ln()
        for name in ('bc0','bc1','bc2','bc3'):s.run(name)
        self.contract_gp();self.dx.copy_prefix();self.dx.pack_weights()
        self.input_ready.record(main)
        with torch.cuda.stream(self.input_side):
            self.input_side.wait_event(self.input_ready)
            self.input_dw();self.input_done.record(self.input_side)
        s.run('dx')
        main.wait_event(self.input_done)
        # The finish consumes both dX and all gate/input weight partials.
        self.dx.reduce_only()
        return p.outputs

def configuration(plan):
    result=previous_configuration(plan)
    result['input_weight_overlap']=dict(enabled=plan.input_overlap,order='dw_first',explicit_fork_join_events=True,separate_workspaces=True)
    return result
