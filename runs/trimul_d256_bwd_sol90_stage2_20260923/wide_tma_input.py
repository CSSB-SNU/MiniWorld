"""TMA input-LN/residual and exact 32-way dW reduction pilot."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class TmaInput:
    def __init__(self,p,rows=16,threads=128,minblocks=4):
        root=Path(__file__).resolve().parent;d=p.D;self.p=p;self.threads=threads
        self.smem=max(3*rows*d*2+128,2*(threads//32)*d*4)
        body=(root/'wide_tma_input.cu').read_text().replace('// AFFINE_HELPER',(root/'ln_aggregate.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),
               f'-DWIDTH={d}',f'-DINPUT_ROWS={rows}',f'-DINPUT_THREADS={threads}',f'-DINPUT_MINBLOCKS={minblocks}']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_wide_tma_input').kernel('mw_wide_tma_input')
        self.kernel.set_max_dynamic_smem(self.smem);drv=self.kernel.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.kernel.handle)),threads,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module();tm=lambda t:L.tensor_map(t,[64,rows],dims=[d,p.M],strides_bytes=[d*2],swizzle='128B',l2='128B')
        self.params=L.Struct([tm(p.x),tm(p.tensors[10]),tm(p.dy),tm(p.dx),p.floats[0],p.floats[8],p.floats[9],p.floats[7],*p.tensors[17:21],p.M])

    def __call__(self):
        L=T._launch_module();drv=self.kernel.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.kernel.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
