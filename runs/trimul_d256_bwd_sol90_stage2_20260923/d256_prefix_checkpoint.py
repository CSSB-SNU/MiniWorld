"""D256 bulk-mask source with ordered Lt dX and cached TMA input LN."""
import os
import torch
from d256_dense_checkpoint import Training as Previous,configuration as previous_configuration,configure
from d256_mask_source import MaskSource
from prefix_dx import PrefixDx
from wide_cached_input import CachedInput
from tma_b7 import TmaB7
from lt_contract import LtBmm

LT_ALGO=[85899345986,4294967331,0,844424930131969,518849229225985,60129542158,292058693646,0]

class Training(Previous):
    def __init__(self,leaves,mask,ds,dy):
        os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
        super().__init__(leaves,mask,ds,dy)
        p=self.p
        self.dx=PrefixDx(p,leaves,splits=8 if p.n==384 else 16)
        # PrefixDx rebinds GP; construct source maps only after that binding.
        self.b7=MaskSource(TmaB7(p,leaves,packed=True,mask=self.f.mask),'bulk')
        self.input_tma=CachedInput(p,16,128,4,splits=self.b7.splits)
        self.dx.reduce_only=self.input_tma
        self.workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
        self.dx_lt=LtBmm(self.dx.input.t().unsqueeze(0),self.dx.weights.unsqueeze(0),p.tensors[10].unsqueeze(0),self.workspace)
        self.dx_lt.index=3
        assert list(self.dx_lt.heuristics[3].algo.data)==LT_ALGO
        self.dx.gemm_only=self.dx_lt
        self.b7.wide_finish=self.dx
        self.artifacts.extend((self.b7.cubin,self.dx.copy_cubin,self.input_tma.cubin))
        self.metadata['prefix_dx']=dict(source_cubin=str(self.b7.cubin),copy_cubin=str(self.dx.copy_cubin),
                                       input_cubin=str(self.input_tma.cubin),lt_index=3,lt_algo=LT_ALGO)

def configuration(plan):
    result=previous_configuration(plan)
    result.update(mask='bulk',dx='prefix_lt',input_ln='cached_tma16_128_4',
                  dx_lt=dict(index=3,algo=LT_ALGO),input_stats_used=False)
    return result
