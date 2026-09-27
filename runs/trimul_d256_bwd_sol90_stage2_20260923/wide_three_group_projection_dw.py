"""Three N256 compute groups fill the single-CTA register allocation."""
import torch
from wide_n256_projection_dw import N256ProjectionDW
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class ThreeGroupProjectionDW(N256ProjectionDW):
    def __init__(self,plan,splits=22):
        super().__init__(plan,splits)
        p=plan.p;d=p.D;h=2*d;assert d in (384,512)
        padded=((d+191)//192)*192;body=self.source_text
        body=body.replace('H=2*D,STAGE=49152','H=2*D,PD=((D+191)/192)*192,STAGE=57344')
        body=body.replace('__launch_bounds__(256,1)','__launch_bounds__(384,1)')
        body=body.replace('ki,rm*2);','ki,rm*3);').replace('STAGE+16384','STAGE+24576')
        body=body.replace('(D/128)*(H/256)','(PD/192)*(H/256)')
        body=body.replace('split*D+rm*128+wg*64','split*PD+rm*192+wg*64')
        body=body.replace('size_t(s)*D*H+i','size_t(s)*PD*H+i')
        body=body.replace('mw_wide_n256_projection_dw','mw_wide_three_group_projection_dw');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DDW_SPLITS={splits}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_wide_three_group_projection_dw')
        self.k=unit.kernel('mw_wide_three_group_projection_dw');self.reduce=unit.kernel('mw_wide_three_group_projection_dw_reduce')
        self.smem=196608+128;self.grid=(padded//192)*(h//256)*splits
        self.k.set_max_dynamic_smem(self.smem);L=T._launch_module()
        self.partial=torch.empty((splits,padded,h),device=p.x.device,dtype=torch.float32)
        a=L.tensor_map(p.tensors[7],[64,64,3],dims=[64,p.M,d//64],strides_bytes=[d*2,128],swizzle='128B',l2='256B')
        b=L.tensor_map(p.tensors[6],[64,64,4],dims=[64,p.M,h//64],strides_bytes=[h*2,128],swizzle='128B',l2='256B')
        y=L.tensor_map(self.partial,[32,64,8],dims=[32,padded*splits,h//32],strides_bytes=[h*4,128],swizzle='128B',l2='128B')
        self.params=L.Struct([a,b,y,self.partial,p.dwp,p.M]);drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))
    def __call__(self):
        self.k.launch((self.grid,1,1),(384,1,1),[self.params],self.smem)
        self.reduce.launch((self.reduce_grid,1,1),(256,1,1),[self.params],0)
