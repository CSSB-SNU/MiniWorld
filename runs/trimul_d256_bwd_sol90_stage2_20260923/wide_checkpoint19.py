"""Short wide output LN with whole-tensor TMA and unchanged shared layout."""
from wide_checkpoint18 import Training as Previous,configuration as previous_configuration
from wide_cached_ln import CachedLN
from wide_prefetch_ln import PrefetchLN
from wide_permuted_stats_ln import PermutedStatsLN

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.n==384:
            base=(PrefetchLN if self.p.D==384 else CachedLN)(self.p,16,True,False,2)
            self.ln=PermutedStatsLN(self.p,base,16,2)
            self.artifacts.append(self.ln.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    if plan.p.n==384:result['output_ln_transfer']='one_3d_map_per_tensor_original_shared_layout'
    return result
