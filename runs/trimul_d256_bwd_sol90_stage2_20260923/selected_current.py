"""Current D256 candidate; explicit forward/backward, no engine dispatch changes."""
from pathlib import Path
import sys,torch,os
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'trimul_d256_bwd_sol90_20260923'))
from selected import Training as Previous
sys.path.insert(0,str(R))
from hybrid_b1 import HybridB1
from tma_b7 import TmaB7
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class Training(Previous):
 def __init__(self,leaves,mask,dropscale,dy):
  super().__init__(leaves,mask,dropscale,dy)
  self.live_mask=mask
  with torch.no_grad(),T.native_context(leaves[0].device):
   self.b1=HybridB1(self.p,leaves,use_cute=True)
   self.b7=TmaB7(self.p,leaves,packed=True,mask=self.f.mask)
   if os.environ.get('DX_LN')=='1':
    from dx_ln import DxLN
    self.b7.wide_finish=DxLN(self.p,self.b7.splits)
   if os.environ.get('SAVE_NORM')=='1':
    from saved_norm import enable
    enable(self)
 def forward(self):
  self._guard()
  with T.native_context(self.leaves[0].device):
   self.f.mask.copy_(self.live_mask.reshape_as(self.f.mask))
   return super().forward()
