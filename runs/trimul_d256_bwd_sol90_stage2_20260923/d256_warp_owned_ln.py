"""Keep the current standalone LN, replacing only32-row transpose ownership."""
from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class WarpOwnedLN:
    def __init__(self,p,original):
        self.__dict__.update(original.__dict__);assert p.D==256 and self.rows==32
        body=original.source_text;root=Path(__file__).resolve().parent
        begin=body.index('template<bool INVERSE>');end=body.index('TMN_DEVI void put_tile',begin)
        body=body[:begin]+(root/'warp_owned_transpose32.cuh').read_text()+'\n'+body[end:]
        body=body.replace('mw_permuted_stats_ln','mw_d256_warp_owned_ln');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=256','-DLN_ROWS=32','-DLN_DN_TMA=1','-DLN_FENCE=0','-DLN_MINBLOCKS=3']
        self.cubin=T.compile_text(body,flags);self.kernel=T.load_unit(str(self.cubin),'mw_d256_warp_owned_ln').kernel('mw_d256_warp_owned_ln');self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv;fn=drv.d.CUfunction(int(self.kernel.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,128,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
    def __call__(self):
        L=T._launch_module();drv=self.kernel.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.kernel.handle)),self.grid,1,1,128,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
