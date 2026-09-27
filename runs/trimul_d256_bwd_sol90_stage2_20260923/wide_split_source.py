"""Pilot native derivatives plus exact-interval FP32 batched input dW."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from lt_contract import LtBmm


class SplitInputSource:
    def __init__(self, p, original):
        self.p = p
        self.original = original
        root = Path(__file__).resolve().parent
        pre = root.parent/'trimul_d256_bwd_sol90_20260923'
        body = (root/'wide_source.cu').read_text()
        body = body.replace('// MMA_HELPERS', (pre/'mma.cuh').read_text())
        body = body.replace('// PACKED_GLU', (root/'packed_glu.cuh').read_text().replace('s+32768', 's+CH'))
        body = body.replace('float dw[NC][32]={};', '')
        begin = body.index('  static_for<NC>([&](auto cc)')
        end = body.index('  if(tid==0)tma_store_wait_all();', begin)
        body = body[:begin] + body[end:]
        begin = body.index(' static_for<NC>([&](auto nn)')
        body = body[:begin] + '}\n'
        body = body.replace('__launch_bounds__(256,1)', '__launch_bounds__(128,1)')
        body = body.replace('mw_wide_b7_source', 'mw_wide_gp_only')
        flags = ['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
                 '-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}','-DWEIGHT_SPLITS=32']
        self.cubin = T.compile_text(body,flags)
        self.kernel = T.load_unit(str(self.cubin),'mw_wide_gp_only').kernel('mw_wide_gp_only')
        self.kernel.set_max_dynamic_smem(original.source_smem)
        d, step = p.D, p.M//32
        self.partial = p.floats[7].reshape(-1)[3*d*d:].as_strided((32,8*d,d),(11*d*d,d,1))
        a = p.gp_all.as_strided((32,8*d,step),(step,p.M,1))
        b = p.xn.as_strided((32,step,d),(step*d,d,1))
        self.workspace = torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
        self.matmul = LtBmm(a,b,self.partial,self.workspace)

    def derivatives(self):
        self.kernel.launch((self.p.D//8*32,1,1),(128,1,1),[self.original.params],self.original.source_smem)

    def __call__(self):
        self.derivatives()
        self.matmul()
