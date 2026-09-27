"""Short wide paths initialize affines early; D512 uses ordered two-warp LN."""
from wide_checkpoint21 import Training as Previous, configuration as previous_configuration
import early_affine_checkpoint as early


class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.n==384:early.attach(self,ordered=self.p.D==512)


def configuration(plan):
    result=previous_configuration(plan)
    result['early_affines']=early.configuration(plan)
    return result
