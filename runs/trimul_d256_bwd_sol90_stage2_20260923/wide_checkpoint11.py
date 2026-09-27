"""D512 lossless normalization correction and compact projection recomputation."""
import os
from wide_checkpoint10 import Training as Previous,configuration as previous_configuration
from wide_delta_norm import DeltaNorm,PatchDeltaNorm
from wide_compact_projection import CompactProjection

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        self.delta_output=None
        if self.p.D!=512:return
        cap=0 if os.environ.get('FORCE_NORM_OVERFLOW')=='1' else None
        rowcap=64 if os.environ.get('FORCE_ROW_OVERFLOW')=='1' else None
        self.delta_output=DeltaNorm(self.p,self.product_output,capacity=cap,cached=True,unswitch=True,minblocks=2)
        self.product_output=self.delta_output
        self.delta_backward=PatchDeltaNorm(self.delta_output)
        self.delta_projection=CompactProjection(self,self.delta_output,capacity=rowcap)
        self.artifacts.extend((self.delta_output.cubin,self.delta_output.fallback_cubin,self.delta_projection.cubin,self.delta_projection.fallback_cubin))

    def backward(self):
        if self.delta_output is None:return super().backward()
        p=self.p;b=self.b1;sch=self.schedule
        self.delta_backward.launch();self.delta_projection()
        b.epi.launch((1056,1,1),(256,1,1),[b.ep],0);sch.run('dn')
        if self.split_dwp is not None:self.split_dwp()
        else:sch.run('dwp')
        sch.run('dwg');self.ln()
        for name in ('bc0','bc1','bc2','bc3'):sch.run(name)
        self.b7.source_only();self.dx.copy_prefix();self.dx.pack_weights();sch.run('dx');self.dx.reduce_only()
        return p.outputs

def configuration(plan):
    result=previous_configuration(plan)
    if plan.delta_output is not None:
        p=plan.delta_projection
        result['delta_normalization']=dict(cached_affine=True,unswitch=True,minblocks=2,patch_capacity=plan.delta_output.capacity,row_capacity=p.capacity,projection_algo=list(p.mm.heuristics[p.mm.index].algo.data))
    return result
