from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
R=Path(__file__).resolve().parent

class Finish:
    def __init__(self,p,stage,splits,groups=2):
        self.p,self.stage=p,stage
        self.threads=128*groups;self.smem=2*8192*(1+groups*2)+128
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=256','-DWIDTH_N=128',f'-DWIDTH_GROUPS={groups}',f'-DWEIGHT_SPLITS={splits}','-DFUSED_GP=0','-DEXTERNAL_GP=1','-DB7_MINB=2']
        out=T.compile(R/'wide_finish.cu',flags)
        self.k=T.load_unit(str(out),'mw_d256_finish128').kernel('width_'+stage)
        self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),self.threads,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
    def __call__(self):
        L=T._launch_module();drv=self.k.unit.drv
        args=L._Packed([self.p.params if self.stage=='b1' else self.p.params7])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
