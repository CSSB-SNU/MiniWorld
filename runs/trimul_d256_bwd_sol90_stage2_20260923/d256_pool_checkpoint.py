"""D256 short N256 dW with a verified 32/208 dynamic register allocation."""
from pathlib import Path
from d256_spatial_checkpoint import Training as Previous,configuration as previous_configuration,configure
from d256_register_budget_source import RegisterBudgetSource


class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        if self.p.n==384:
            self.pool_source=RegisterBudgetSource(self,208)
            self.late_output_dw.oldsource=self.pool_source
            self.artifacts.extend((Path(self.pool_source.pool_metadata['original']),self.pool_source.cubin))


def configuration(plan):
    result=previous_configuration(plan)
    if plan.p.n==384:
        op=plan.pool_source
        result['source_register_pool']=dict(columns=256,producer=32,consumer=208,initial=120,resident_ctas=op.occupancy,local_bytes=op.local_bytes,machine_code_unchanged=op.pool_metadata['machine_code_unchanged'],original_sha256=op.pool_metadata['original_sha256'],patched_sha256=op.pool_metadata['patched_sha256'])
    return result
