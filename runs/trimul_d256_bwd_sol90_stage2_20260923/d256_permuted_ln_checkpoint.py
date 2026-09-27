"""D256 short checkpoint with one TMA command per output-LN tensor."""
from d256_stats_checkpoint import Training as Previous,configuration as previous_configuration,configure
from wide_cached_ln import CachedLN
from wide_permuted_stats_ln import PermutedStatsLN

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.n==384:
            base=CachedLN(self.p,32,True,False,3)
            self.b1.ln=PermutedStatsLN(self.p,base,32,3)
            self.artifacts.append(self.b1.ln.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    if plan.p.n==384:result['output_ln_transfer']='one_3d_map_per_tensor_original_shared_layout'
    return result
