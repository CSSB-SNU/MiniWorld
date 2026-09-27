"""Explicit experimental wide training path; no engine dispatch changes."""
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
from wide_hybrid_b1 import WideB1
from wide_b7 import WideB7
from prefix_dx import PrefixDx
from wide_saved_norm import SavedOutput
import torch


class WideTraining:
    def __init__(self, leaves, mask, ds, dy):
        self.mask = mask
        self.f = F.Forward(leaves, mask, ds)
        self.f()
        self.p = W.Training(*leaves, mask, ds, dy,
                            saved=(self.f.front.ab, self.f.tri, self.f.front.xn),
                            packed=self.f.w)
        p = self.p
        self.original_output = self.f.output
        # Rebind the GP maps before constructing either source consumer.
        self.dx = PrefixDx(p, leaves)
        self.b1 = WideB1(p, leaves)
        self.b7 = WideB7(p, self.f.mask)
        self.saved_output = SavedOutput(self.f, p) if p.D == 384 else None
        self.artifacts = [self.dx.cubin, self.dx.copy_cubin,
                          self.b1.norm_cubin, self.b1.ln_cubin,
                          self.b7.source_cubin]
        if self.b1.exact_dwp is not None:
            self.artifacts.append(self.b1.exact_dwp.cubin)
        if self.saved_output is not None:
            self.artifacts.append(self.saved_output.cubin)

    def forward(self, saved=True):
        self.f.mask.copy_(self.mask)
        self.p.mask.copy_(self.mask.reshape_as(self.p.mask))
        use_saved = saved and self.saved_output is not None
        self.f.output = self.saved_output if use_saved else self.original_output
        self.b1.use_saved_norm = use_saved
        return self.f()

    def contract(self):
        p = self.p
        d, h, ab = p.D, 2 * p.D, p.front.ab
        torch.bmm(p.dt[:d], ab[h:h+d], out=p.dl[:d])
        torch.bmm(p.dt[:d].transpose(-1, -2), ab[:d], out=p.dr[:d])
        torch.bmm(ab[h+d:], p.dt[d:].transpose(-1, -2), out=p.dl[d:])
        torch.bmm(ab[d:h], p.dt[d:], out=p.dr[d:])

    def backward(self):
        self.b1()
        self.contract()
        self.b7.source_only()
        self.dx()
        return self.p.outputs

    def __call__(self):
        return self.forward(), self.backward()
