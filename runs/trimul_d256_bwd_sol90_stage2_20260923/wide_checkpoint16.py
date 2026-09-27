"""Current wide kernels with epilogue loads in retired contraction slots."""
from wide_checkpoint15 import Training as Previous,configuration as previous_configuration
from wide_staged_epilogue_gp import StagedEpilogueGP

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        self.contract_gp=StagedEpilogueGP(self)
        self.artifacts.append(self.contract_gp.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    result['contraction_epilogue']='load_gate_and_projection_into_retired_mma_slots'
    return result
