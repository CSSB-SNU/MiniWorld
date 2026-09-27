"""D256 short output weights overlap source; long path retains prior kernels."""
from d256_permuted_ln_checkpoint import Training as Previous,configuration as previous_configuration,configure
from d256_late_output_dw import LateOutputDW

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.n==384:
            self.late_output_dw=LateOutputDW(self,'source',False,('dwp','dwg'))
            self.late_output_dw.select(True)
            self.metadata['late_output_weights']=dict(phase='source',targets=['dwp','dwg'],weights_first=True,separate_workspace=True,explicit_join=True)

def configuration(plan):
    result=previous_configuration(plan)
    result['late_output_weights']=plan.metadata.get('late_output_weights',dict(enabled=False))
    return result
