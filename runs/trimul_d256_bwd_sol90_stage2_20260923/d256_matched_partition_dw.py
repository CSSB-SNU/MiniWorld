"""Fit source's full CTA wave to its actual SM partition."""
from d256_partitioned_output_dw import PartitionedOutputDW
from tma_b7 import TmaB7
from d256_mask_source import MaskSource
from wide_cached_input import CachedInput

class MatchedPartitionDW(PartitionedOutputDW):
    def __init__(self,plan,small_sms=16,splits=7,split_flags=0):
        super().__init__(plan,small_sms,split_flags)
        self.newsource=MaskSource(TmaB7(plan.p,plan.leaves,packed=True,mask=plan.f.mask,splits=splits),'bulk')
        self.oldfinish=plan.dx.reduce_only
        self.newfinish=CachedInput(plan.p,16,128,4,splits=splits)
        drv=self.newsource.source.unit.drv;fn=drv.d.CUfunction(int(self.newsource.source.handle))
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.newsource.smem)))
        assert 32*splits<=self.partition.sm_counts[1]*self.occupancy

    def source(self):self.fork(self.newsource.source_only)

    def select(self,enabled):
        super().select(enabled)
        self.plan.dx.reduce_only=self.newfinish if enabled else self.oldfinish
