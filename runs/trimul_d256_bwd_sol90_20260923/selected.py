"""Explicit D256 training plan. No automatic dispatch/autograd registration."""
from pathlib import Path
import sys,torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent.parent/'.engine-release-2.0.0/src'))
sys.path.insert(0,str(R.parent/'trimul_backward_wide_20260923'))
from plan import B1
from b7 import B7
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W

class Training:
    def __init__(self,leaves,mask,dropscale,dy):
        if leaves[0].shape[-1]!=256 or leaves[0].shape[1] not in (384,768):
            raise ValueError('Selected D256 plan supports B1 BF16 L384/768 only')
        import os
        if os.environ.get('B7_LOOP')=='1':
            raise ValueError('B7_LOOP is an unselected experimental schedule')
        self.leaves=leaves
        with torch.no_grad(),T.native_context(leaves[0].device):
            self.f=F.Forward(leaves,mask,dropscale)
            self.p=W.Training(*leaves,mask,dropscale,dy,
                saved=(self.f.front.ab,self.f.tri,self.f.front.xn),packed=self.f.w)
            self.b1=B1(self.p);self.b7=B7(self.p,8 if self.p.n==384 else 16)
        self.saved=self.f.saved
        self._has_forward=False

    def _guard(self):
        if torch.is_grad_enabled():
            raise RuntimeError('Explicit plan requires torch.no_grad(); returns gradients directly')

    def forward(self):
        self._guard()
        with T.native_context(self.leaves[0].device):
            y=self.f();self._has_forward=True;return y

    def backward(self):
        self._guard()
        if not self._has_forward:raise RuntimeError('Run this plan forward before backward')
        p=self.p
        with T.native_context(p.x.device):
            self.b1();ab=p.front.ab;d=256;h=512
            torch.bmm(p.dt[:d],ab[h:h+d],out=p.dl[:d])
            torch.bmm(p.dt[:d].transpose(-1,-2),ab[:d],out=p.dr[:d])
            torch.bmm(ab[h+d:],p.dt[d:].transpose(-1,-2),out=p.dl[d:])
            torch.bmm(ab[d:h],p.dt[d:],out=p.dr[d:])
            return self.b7()

    def __call__(self):return self.forward(),self.backward()
