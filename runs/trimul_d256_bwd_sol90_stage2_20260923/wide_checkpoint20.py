"""Short wide LN retains packed BF16 operands across exact scalar passes."""
from wide_checkpoint19 import Training as Previous,configuration as previous_configuration
from wide_cached_ln import CachedLN
from wide_prefetch_ln import PrefetchLN
from wide_packed_retained_ln import PackedRetainedLN

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.n==384:
            base=(PrefetchLN if self.p.D==384 else CachedLN)(self.p,16,True,False,2)
            self.ln=PackedRetainedLN(self.p,base,16,2)
            self.artifacts.append(self.ln.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    if plan.p.n==384:result['output_ln_operands']='exact_bf16_pairs_retained_between_scalar_passes'
    return result
