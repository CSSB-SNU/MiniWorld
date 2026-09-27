"""Current D256 kernels with row32 TMA-prefetched output-LN statistics."""
from d256_gate_checkpoint import Training as Previous,configuration as previous_configuration,configure
from wide_cached_ln import CachedLN
from wide_tuned_stats_ln import TunedStatsLN

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        base=CachedLN(self.p,32,True,False,3)
        self.b1.ln=TunedStatsLN(self.p,base,32,3)
        self.artifacts.append(self.b1.ln.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    result['output_ln_stats']=dict(tma=True,rows=32,threads=128,minblocks=3)
    return result
