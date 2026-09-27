"""D256 short full N384 contraction tile with ordered K accumulation."""
from d256_early_affine_checkpoint import Training as Previous, configuration as previous_configuration, configure
from d256_full_spatial_contract import FullSpatialContract


class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.n==384:
            self.native_contract=FullSpatialContract(self,2,3,256)
            self.artifacts.append(self.native_contract.cubin)


def configuration(plan):
    result=previous_configuration(plan)
    if plan.p.n==384:
        result['native_contraction']=dict(tile=[128,384,64],slots=3,grid='paired_modes',whole_tma=True,exact_k_order=True,columns=[256,128])
    return result
