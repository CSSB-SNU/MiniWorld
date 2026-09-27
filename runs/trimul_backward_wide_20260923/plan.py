"""Development B1: reuse our wide forward's on-chip LN and streamed products."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

R = Path(__file__).resolve().parent


class B1:
    def __init__(self, training, groups=None):
        self.p = p = training
        D, M, N = p.D, p.M, p.n
        self.groups = G = groups or {256: 4, 384: 3, 512: 4}[D]
        self.smem = 2 * D * 128 + 2 * 8192 * (G + 1) + 640
        flags = ['-std=c++17', '-O3', '-arch=sm_90a', '--cubin', '-lineinfo', '-Xptxas=-v',
                 '-I' + str(F.headers()), '-DMW_MINB=1', '-DMW_K1_STREAM=0',
                 f'-DWIDTH={D}', f'-DGROUPS={G}', '-DKCHUNK=1', '-DTMN_SIGMOID_TANH=1']
        out = T.compile(R / 'prepare.cu', flags)
        self.prepare = T.load_unit(str(out), 'mw_b1_prepare').kernel('mw_wide_b1_prepare')
        self.prepare.set_max_dynamic_smem(self.smem)
        flags = ['-std=c++17', '-O3', '-arch=sm_90a', '--cubin', '-lineinfo', '-Xptxas=-v',
                 '-I' + str(T._upstream() / 'csrc'), f'-DWIDTH={D}', '-DWEIGHT_SPLITS=32']
        out = T.compile(R / 'finish.cu', flags)
        self.finish = T.load_unit(str(out), 'mw_b1_finish').kernel('width_b1')
        self.finish.set_max_dynamic_smem(W.tuning(D)[2])
        drv = self.finish.unit.drv
        occ = int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',
            drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.finish.handle)), 256, W.tuning(D)[2])))
        import torch
        self.sms = torch.cuda.get_device_properties(p.x.device).multi_processor_count
        self.grid = self.sms * occ
        maps = [F.tm(p.tri, [64,64], [M,2*D], [2*M]), p.maps[0], p.maps[1], p.maps[2]]
        self.params = T._launch_module().Struct([
            *maps, p.x, p.ds, p.y, p.dy, p.tensors[7], p.dg, p.tensors[6],
            p.floats[5], p.floats[6], p.floats[2], p.floats[3], M, N])

    def __call__(self):
        self.prepare.launch((self.sms,1,1), (128*self.groups,1,1), [self.params], self.smem)
        W.launch(self.finish, self.p.params, self.grid, D=self.p.D)


def tail(p):
    import torch
    ab, d = p.front.ab, p.D
    h = 2*d
    torch.bmm(p.dt[:d], ab[h:h+d], out=p.dl[:d])
    torch.bmm(p.dt[:d].transpose(-1,-2), ab[:d], out=p.dr[:d])
    torch.bmm(ab[h+d:], p.dt[d:].transpose(-1,-2), out=p.dl[d:])
    torch.bmm(ab[d:h], p.dt[d:], out=p.dr[d:])
    W.launch(p.ks['b7'], p.params7, p.grid7, D=d, gp=p.gp_native)
    return p.outputs
