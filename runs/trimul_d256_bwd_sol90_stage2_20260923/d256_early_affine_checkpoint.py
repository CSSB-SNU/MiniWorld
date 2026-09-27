"""D256 short affine initialization in the preceding output-gate kernel."""
from d256_contract_checkpoint import Training as Previous, configuration as previous_configuration, configure
import early_affine_checkpoint as early


class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.n==384:early.attach(self)


def configuration(plan):
    result=previous_configuration(plan)
    result['early_affines']=early.configuration(plan)
    return result
