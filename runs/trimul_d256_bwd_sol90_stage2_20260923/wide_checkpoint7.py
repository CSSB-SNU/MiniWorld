"""Wide explicit candidate: separate D384 TMA loader, pipelined D512 GP."""
from wide_checkpoint6 import Training as Previous,configuration as previous_configuration
from wide_four_group_source import FourGroupSource
from wide_saved_gp_pipe import SavedGpPipe

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.D==384:self.new_source=FourGroupSource(self.p,self.b7,async_dw=False)
        else:self.new_source=SavedGpPipe(self.saved_source,pipe=True,rank_major=True,grid_factor=4)
        self.b7.source_only=self.new_source
        self.artifacts.append(self.new_source.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    if plan.p.D==384:result['source']='four_warpgroup_separate_loader_sync_dw'
    else:result['source_gp']=dict(packed=True,input_slots=2,rank_major=True,grid_factor=4)
    return result
