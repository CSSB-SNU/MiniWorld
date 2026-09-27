"""Pilot dense preactivation and input-weight GEMMs for wide backward."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class BlasWideSource:
    def __init__(self, p, mask):
        self.p = p
        d = p.D
        self.pre = p.x.new_empty((8*d, p.M))
        self.w = p.x.new_empty((8*d, d))
        self.dw = p.x.new_empty((8*d, d))
        body = Path(__file__).with_name('blas_b7.cu').read_text()
        body = body.replace('256', 'WIDTH').replace('1024', '(4*WIDTH)').replace('512', '(2*WIDTH)')
        # The launch block size is independent of the channel dimension.
        body = body.replace('blockIdx.x*WIDTH+threadIdx.x', 'blockIdx.x*256+threadIdx.x')
        body = body.replace('gridDim.x*WIDTH', 'gridDim.x*256')
        body = body.replace('mw_dWIDTH_blas_gp', 'mw_wide_blas_gp')
        flags = ['-std=c++17', '-O3', '-arch=sm_90a', '--cubin', '-lineinfo',
                 '-Xptxas=-v', '-I'+str(T._upstream()/'csrc'), f'-DWIDTH={d}']
        self.cubin = T.compile_text(body, flags)
        self.epi = T.load_unit(str(self.cubin), 'mw_wide_blas_gp').kernel('mw_wide_blas_gp')
        self.params = T._launch_module().Struct([self.pre, p.dl, p.dr, mask, p.gp_all, p.M])

    def products(self):
        p = self.p
        torch.cat(p.weights, out=self.w)
        torch.mm(self.w, p.xn.reshape(p.M, p.D).t(), out=self.pre)

    def derivatives(self):
        self.products()
        self.epi.launch((1056, 1, 1), (256, 1, 1), [self.params], 0)

    def weights(self):
        p = self.p
        torch.mm(p.gp_all.reshape(8*p.D, p.M), p.xn.reshape(p.M, p.D), out=self.dw)
        for i in range(4):
            p.dw[i].copy_(self.dw[i*2*p.D:(i+1)*2*p.D])
