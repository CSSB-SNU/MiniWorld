"""Stream K32 dNorm tiles across all D output columns, then exact-order LN."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class StreamDnLN:
    def __init__(self,plan,emit_dn=False):
        p=plan.p;d=p.D;h=2*d;self.p=p;self.threads=d;root=Path(__file__).resolve().parent
        body=(root/'wide_stream_dn_ln.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text()).replace('// TRANSPOSE_HELPERS',(root/'tile_transpose.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_wide_stream_dn_ln')
        self.k=unit.kernel('mw_wide_stream_dn_ln');self.smem=8192+384*d+128;self.k.set_max_dynamic_smem(self.smem)
        self.reduce_first=unit.kernel('mw_wide_stream_ln_reduce_first');self.reduce_last=unit.kernel('mw_wide_stream_ln_reduce_last')
        self.rows=p.M//64;self.chunks=(self.rows+255)//256
        self.partial=torch.empty((self.rows,2,h),device=p.x.device,dtype=torch.float32)
        self.tmp=torch.empty((self.chunks,2,h),device=p.x.device,dtype=torch.float32)
        L=T._launch_module()
        tri=lambda t:L.tensor_map(t,[16,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='32B',l2='128B')
        dp=L.tensor_map(p.tensors[7],[32,64],dims=[d,p.M],strides_bytes=[d*2],swizzle='64B',l2='128B')
        wp=L.tensor_map(plan.b1.wp,[64,32],dims=[h,d],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.params=L.Struct([tri(p.tri),tri(p.dt),dp,wp,p.floats[5],p.floats[6],p.floats[2],self.partial,p.tensors[9],p.M])
        self.rp=L.Struct([self.partial,self.tmp,p.floats[10],p.floats[11],self.rows,self.chunks])
    def __call__(self):
        self.k.launch((self.rows,1,1),(self.threads,1,1),[self.params],self.smem)
        self.reduce_first.launch((2*self.p.D//128,self.chunks,1),(256,1,1),[self.rp],0)
        self.reduce_last.launch((2*self.p.D//128,1,1),(128,1,1),[self.rp],0)
