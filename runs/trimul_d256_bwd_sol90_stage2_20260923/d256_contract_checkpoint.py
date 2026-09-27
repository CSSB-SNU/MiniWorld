"""D256 short fixed native contractions plus qualified late output weights."""
from d256_late_dw_checkpoint import Training as Previous,configuration as previous_configuration,configure
from d256_whole_fixed_contract import WholeFixedContract

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.n==384:
            self.native_contract=WholeFixedContract(self,2,3)
            self.artifacts.append(self.native_contract.cubin)
            previous_run=self.schedule.run
            def run(name):
                if name=='bc0':self.native_contract()
                elif name not in ('bc1','bc2','bc3'):previous_run(name)
            self.schedule.run=run

def configuration(plan):
    result=previous_configuration(plan)
    if plan.p.n==384:result['native_contraction']=dict(tile=[128,128,64],slots=3,grid='paired_modes',whole_tma=True,exact_k_order=True)
    return result
