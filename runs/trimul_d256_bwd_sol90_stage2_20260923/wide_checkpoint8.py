"""Channel-major preactivation saves and fused contraction/GP, pending qualification."""
import torch
from wide_checkpoint7 import Training as Previous,configuration as previous_configuration
from wide_saved_front_transposed import SavedTransposedFront
from wide_two_group_contract_gp import TwoGroupContractGP
from wide_cached_input import CachedInput
from lt_contract import LtBmm

D384_DW_ALGO=[897648164930,4294967331,0,844424930131972,518849229225985,60129542158,292057776128,0]

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        p=self.p;d=p.D
        self.pre=p.x.new_empty((8*d,p.M))
        self.f.front=SavedTransposedFront(self.baseline_front,self.pre)
        self.channel_group=2 if d==384 else 1
        self.contract_gp=TwoGroupContractGP(self,channel_group=self.channel_group)
        if d==512:self.input_dw=self.saved_source.matmul
        else:
            splits=8 if p.n==384 else 16;step=p.M//splits
            self.input_splits=splits
            self.input_partial=p.floats[7].reshape(-1)[3*d*d:].as_strided((splits,8*d,d),(11*d*d,d,1))
            a=p.gp_all.as_strided((splits,8*d,step),(step,p.M,1))
            b=p.xn.as_strided((splits,step,d),(step*d,d,1))
            self.input_workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
            self.input_dw=LtBmm(a,b,self.input_partial,self.input_workspace);self.input_dw.index=0
            assert list(self.input_dw.heuristics[0].algo.data)==D384_DW_ALGO
            self.input_tma=CachedInput(p,16,128,4,splits=splits)
            self.dx.reduce_only=self.input_tma;self.artifacts.append(self.input_tma.cubin)
        self.b7.source_only=self._source
        self._schedule_run=self.schedule.run;self.schedule.run=self._run
        self.artifacts.extend((self.f.front.cubin,self.contract_gp.cubin))

    def _source(self):self.contract_gp();self.input_dw()

    def _run(self,name):
        if name not in ('bc0','bc1','bc2','bc3'):self._schedule_run(name)


def configuration(plan):
    result=previous_configuration(plan)
    op=plan.input_dw
    result.update(source='fused_contraction_gp_then_split_input_dw',front='channel_major_preactivation_tma',
                  source_tile=[128,128],source_groups=2,source_channel_group=plan.channel_group,
                  input_splits=plan.input_splits,saved_bytes=plan.pre.numel()*2,
                  source_lt=dict(index=op.index,algo=list(op.heuristics[op.index].algo.data)))
    return result
