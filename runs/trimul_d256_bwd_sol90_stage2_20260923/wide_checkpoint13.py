"""D512 L768 output LN with a separate affine worker and protected output tile."""
from wide_checkpoint12 import Training as Previous,configuration as previous_configuration
from wide_affine_output_ln import AffineOutputLN

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.D==512 and self.p.n==768:
            self.ln=AffineOutputLN(self.p)
            self.artifacts.append(self.ln.cubin)

def configuration(plan):
    result=previous_configuration(plan)
    if plan.p.D==512 and plan.p.n==768:
        result['output_ln']=dict(rows=16,threads=256,affine_worker=True,separate_output_tile=True,registers=[144,112])
    return result
