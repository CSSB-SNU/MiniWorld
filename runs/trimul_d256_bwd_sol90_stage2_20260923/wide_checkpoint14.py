"""Interleaved contraction consumers and low-register lossless D512 normalization."""
import os
from wide_checkpoint13 import Training as Previous,configuration as previous_configuration
from wide_interleaved_contract_gp import InterleavedContractGP
from wide_lowreg_delta_norm import LowRegisterDeltaNorm,PatchDeltaNorm
from wide_compact_projection import CompactProjection

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        self.contract_gp=InterleavedContractGP(self,'pair_tiles')
        self.artifacts.append(self.contract_gp.cubin)
        if self.p.D==512:
            cap=0 if os.environ.get('FORCE_NORM_OVERFLOW')=='1' else None
            rowcap=64 if os.environ.get('FORCE_ROW_OVERFLOW')=='1' else None
            self.delta_output=LowRegisterDeltaNorm(self.p,self.product_output,capacity=cap,cached=True,unswitch=True,minblocks=3,rows=32,threads=128)
            self.product_output=self.delta_output
            self.delta_backward=PatchDeltaNorm(self.delta_output)
            self.delta_projection=CompactProjection(self,self.delta_output,capacity=rowcap)
            self.artifacts.extend((self.delta_output.cubin,self.delta_output.fallback_cubin,self.delta_projection.cubin,self.delta_projection.fallback_cubin))

def configuration(plan):
    result=previous_configuration(plan)
    result['contraction_order']='pair_tiles'
    if plan.p.D==512:
        result['delta_normalization'].update(rows=32,threads=128,minblocks=3,input_reload=True)
    return result
