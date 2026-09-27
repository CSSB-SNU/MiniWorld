"""D256 dense output and bulk mask staging, explicit candidate only."""
from d256_dense_checkpoint import Training as Previous,configuration as previous_configuration,configure
from d256_mask_source import MaskSource

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        self.b7=MaskSource(self.b7,'bulk')
        self.artifacts.append(self.b7.cubin)
        self.metadata['mask_source_cubin']=str(self.b7.cubin)

def configuration(plan):
    result=previous_configuration(plan);result['mask']='bulk';return result
