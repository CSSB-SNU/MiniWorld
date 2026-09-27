"""Saved preactivations with shape-specific FP32 input-weight splits."""
from wide_saved_front import SavedSource
from lt_contract import LtBmm

class SavedSplitSource(SavedSource):
    def __init__(self,p,pre,mask,splits):
        super().__init__(p,pre,mask)
        assert p.M%splits==0
        d=p.D;step=p.M//splits;self.splits=splits
        self.partial=p.floats[7].reshape(-1)[3*d*d:].as_strided((splits,8*d,d),(11*d*d,d,1))
        a=p.gp_all.as_strided((splits,8*d,step),(step,p.M,1))
        b=p.xn.as_strided((splits,step,d),(step*d,d,1))
        self.matmul=LtBmm(a,b,self.partial,self.workspace)
