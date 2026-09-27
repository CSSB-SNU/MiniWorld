"""Shape-specific shared-input dW and short D384 prefetched output LN."""
from pathlib import Path
import json
import torch
from wide_checkpoint9 import Training as Previous,configuration as previous_configuration
from wide_joint_input_reduce import JointInputReduce
from wide_prefetch_ln import PrefetchLN
from lt_contract import LtBmm
class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        p=self.p;d=p.D
        self.joint_choice=None
        choices=json.loads(Path(__file__).with_name('wide_checkpoint10_dw.json').read_text())
        if f'{d}_{p.n}' in choices:
            self.joint_choice=choices[f'{d}_{p.n}'];splits=self.input_splits;step=p.M//splits
            a=self.dx.input.as_strided((splits,9*d,step),(step,p.M,1))
            b=p.xn.as_strided((splits,step,d),(step*d,d,1))
            partial=p.floats[7].reshape(-1)[2*d*d:].as_strided((splits,9*d,d),(11*d*d,d,1))
            self.joint_workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
            self.input_dw=LtBmm(a,b,partial,self.joint_workspace);self.input_dw.index=self.joint_choice['index']
            assert list(self.input_dw.heuristics[self.input_dw.index].algo.data)==self.joint_choice['algo']
            self.input_tma=JointInputReduce(p,16,128,4,splits=splits);self.dx.reduce_only=self.input_tma
            oldrun=self.schedule.run
            def run(name):
                if name!='dwg':oldrun(name)
            self.schedule.run=run;self.artifacts.append(self.input_tma.cubin)
        if d==384 and p.n==384:
            self.ln=PrefetchLN(p,16,True,False,2);self.artifacts.append(self.ln.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    result['joint_input_gate_dw']=plan.joint_choice
    if plan.p.D==384 and plan.p.n==384:result['output_ln_pipeline']=dict(rows=16,slots=2,minblocks=2)
    return result
