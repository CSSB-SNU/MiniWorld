"""Explicit wide training candidate with dense output and TMA LN operands."""
from wide_checkpoint3 import Training as Previous
from wide_dense_output import DenseOutput
from wide_tile_ln import TileLN
from wide_tma_input import TmaInput


class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        super().__init__(leaves,mask,ds,dy)
        p=self.p
        self.product_output=DenseOutput(self.f,p,(self.b1.proj,self.b1.gate),
                                       128 if p.D==384 else 256,packed_unroll=1 if p.D==384 else 16)
        self.ln=TileLN(p,32 if p.D==384 else 16,True,False,2)
        self.input_tma=TmaInput(p,16,128,4)
        self.dx.reduce_only=self.input_tma
        self.artifacts.extend((self.product_output.cubin,self.ln.cubin,self.input_tma.cubin))


def configuration(plan):
    d=plan.p.D
    return dict(source='prefetch_offset_n64',input_reduce=['tma',16,128,4],
                output_ln=['tma',32 if d==384 else 16,128,2],
                output=['dense',128 if d==384 else 256,1 if d==384 else 16],
                dwp_algo=list(plan.split_dwp.matmul.heuristics[plan.split_dwp.matmul.index].algo.data) if plan.split_dwp is not None else None)
