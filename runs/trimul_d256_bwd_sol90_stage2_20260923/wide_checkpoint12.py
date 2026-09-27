"""Checkpoint11 with a cached affine, 32-row forward norm at D384."""
from wide_checkpoint11 import Training as Previous,configuration as previous_configuration
from wide_cached_forward_norm import CachedForwardNorm

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.D==384:
            self.product_output=CachedForwardNorm(self.p,self.product_output,32)
            self.artifacts.append(self.product_output.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    if plan.p.D==384:result['forward_norm']=dict(rows=32,threads=128,cached_affine=True)
    return result
