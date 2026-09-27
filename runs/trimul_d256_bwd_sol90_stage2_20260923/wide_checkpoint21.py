"""D384 short packed LN with independent per-warp transpose synchronization."""
from wide_checkpoint20 import Training as Previous,configuration as previous_configuration
from wide_warp_packed_ln import WarpPackedLN

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if (self.p.D,self.p.n)==(384,384):
            self.ln=WarpPackedLN(self.p,self.ln)
            self.artifacts.append(self.ln.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    if (plan.p.D,plan.p.n)==(384,384):result['output_ln_transpose_sync']='warp_local_tiles_then_cta_join'
    return result
