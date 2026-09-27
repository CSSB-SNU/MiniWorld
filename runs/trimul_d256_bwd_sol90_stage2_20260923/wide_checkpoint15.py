"""Candidate: existing wide kernels with TMA-prefetched output-LN statistics."""
from wide_checkpoint14 import Training as Previous,configuration as previous_configuration
from wide_stats_ln import StatsLN

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        self.ln=StatsLN(self.p,self.ln)
        self.artifacts.append(self.ln.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    result['output_ln_stats']=dict(tma=True,rows=plan.ln.rows,kind=plan.ln.kind,smem=plan.ln.smem)
    return result
