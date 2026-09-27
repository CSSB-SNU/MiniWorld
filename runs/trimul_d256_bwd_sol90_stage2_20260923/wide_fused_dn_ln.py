"""Resident dp, two K32 weight slots, shared dNorm, and sliced output LN."""
from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class FusedDnLN:
    def __init__(self,plan,emit_dn=False):
        p=plan.p;d=p.D;h=2*d;self.p=p
        root=Path(__file__).resolve().parent
        body=(root/'wide_fused_dn_ln.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        body=body.replace('// TRANSPOSE_HELPERS',(root/'tile_transpose.cuh').read_text()).replace('// AFFINE_HELPER',(root/'ln_aggregate.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_fused_dn_ln').kernel('mw_wide_fused_dn_ln')
        self.smem=64*d*2+32768+64*h*2+128;assert self.smem<=232448
        self.k.set_max_dynamic_smem(self.smem)
        launch=T._launch_module()
        tri=lambda t:launch.tensor_map(t,[32,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='64B',l2='128B')
        dp=launch.tensor_map(p.tensors[7],[64,64],dims=[d,p.M],strides_bytes=[d*2],swizzle='128B',l2='128B')
        wp=launch.tensor_map(plan.b1.wp,[64,32],dims=[h,d],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.params=launch.Struct([tri(p.tri),tri(p.dt),dp,wp,p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.tensors[9],p.M])
        drv=self.k.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),256,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ;assert occ>0
    def __call__(self):
        launch=T._launch_module();drv=self.k.unit.drv;args=launch._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,256,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
