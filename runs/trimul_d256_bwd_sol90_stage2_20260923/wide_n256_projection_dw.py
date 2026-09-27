"""Full N256 dW instructions and split counts matched to the output tile grid."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class N256ProjectionDW:
    def __init__(self,plan,splits=16):
        p=plan.p;d=p.D;h=2*d;assert d in (256,384,512) and p.M%64==0
        root=Path(__file__).resolve().parent
        body=(root/'wide_native_projection_dw.cu').read_text().replace('// MMA_HELPER',(root/'mma256.cuh').read_text())
        body=body.replace('STAGE=32768','STAGE=49152').replace('__launch_bounds__(256,2)','__launch_bounds__(256,1)')
        body=body.replace('bar+slot,0,ki,cn*2);','bar+slot,0,ki,cn*4);')
        body=body.replace('(H/128)','(H/256)')
        old='steps=p.M/64/DW_SPLITS,begin=split*steps*64;'
        assert body.count(old)==1
        body=body.replace(old,'begin=(p.M/64*split/DW_SPLITS)*64,steps=(p.M/64*(split+1)/DW_SPLITS)-begin/64;')
        body=body.replace('float v[64]={};','float v[128]={};')
        old='mma128_off<k*2048,k*2048,1,1>(v,smem_desc(smem_u32(sm+slot*STAGE+wg*8192),16,1024,1),smem_desc(smem_u32(sm+slot*STAGE+16384),8192,1024,1),it>0||k>0);'
        assert body.count(old)==1
        body=body.replace(old,'mma256<1,1>(v,smem_desc(smem_u32(sm+slot*STAGE+wg*8192+k*2048),16,1024,1),smem_desc(smem_u32(sm+slot*STAGE+16384+k*2048),8192,1024,1),it>0||k>0);')
        body=body.replace('out=sm+wg*32768','out=sm+wg*65536').replace('static_for<32>([&](auto jj)','static_for<64>([&](auto jj)').replace('"r"(cn*4):','"r"(cn*8):')
        body=body.replace('mw_wide_native_projection_dw','mw_wide_n256_projection_dw');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DDW_SPLITS={splits}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_wide_n256_projection_dw')
        self.k=unit.kernel('mw_wide_n256_projection_dw');self.reduce=unit.kernel('mw_wide_n256_projection_dw_reduce')
        self.smem=147456+128;self.grid=(d//128)*(h//256)*splits;self.reduce_grid=(d*h+511)//512
        self.k.set_max_dynamic_smem(self.smem);L=T._launch_module()
        self.partial=torch.empty((splits,d,h),device=p.x.device,dtype=torch.float32)
        a=L.tensor_map(p.tensors[7],[64,64,2],dims=[64,p.M,d//64],strides_bytes=[d*2,128],swizzle='128B',l2='256B')
        b=L.tensor_map(p.tensors[6],[64,64,4],dims=[64,p.M,h//64],strides_bytes=[h*2,128],swizzle='128B',l2='256B')
        y=L.tensor_map(self.partial,[32,64,8],dims=[32,d*splits,h//32],strides_bytes=[h*4,128],swizzle='128B',l2='128B')
        self.params=L.Struct([a,b,y,self.partial,p.dwp,p.M]);drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
    def __call__(self):
        self.k.launch((self.grid,1,1),(256,1,1),[self.params],self.smem)
        self.reduce.launch((self.reduce_grid,1,1),(256,1,1),[self.params],0)
