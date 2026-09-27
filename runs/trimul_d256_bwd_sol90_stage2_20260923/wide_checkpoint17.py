"""Long-shape contraction specialization with fully unrolled K iterations."""
from wide_checkpoint16 import Training as Previous,configuration as previous_configuration
from wide_fixed_length_gp import FixedLengthGP

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.n==768:
            self.contract_gp=FixedLengthGP(self,self.p.n//64)
            self.artifacts.append(self.contract_gp.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    result['fixed_contraction_length']=plan.p.n if plan.p.n==768 else None
    result['contraction_k_unroll']=plan.p.n//64 if plan.p.n==768 else None
    return result
