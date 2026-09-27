"""D512 short contraction keeps two CTAs with a loader warp and whole TMA maps."""
from wide_checkpoint23 import Training as Previous,configuration as previous_configuration
from wide_loader_warp_gp import LoaderWarpGP


class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if (self.p.D,self.p.n)==(512,384):
            self.contract_gp=LoaderWarpGP(self,False,True)
            self.artifacts.append(self.contract_gp.cubin)


def configuration(plan):
    result=previous_configuration(plan)
    if (plan.p.D,plan.p.n)==(512,384):
        result['contraction_loader']=dict(threads=288,loader_warps=1,compute_groups=2,input_maps='whole4d',epilogue_maps='whole4d',unroll=False,resident_ctas=plan.contract_gp.occupancy)
    return result
