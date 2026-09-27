"""Leave shared memory for small dX CTAs alongside persistent GP producers."""
from wide_spatial_gp_dx_overlap import SpatialGPDX
from wide_one_group_transposed_dx import WideOneGroupTransposedDX
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class NativeBandDX(WideOneGroupTransposedDX):
    def __init__(self,plan):
        super().__init__(plan,2,1,64)
        assert plan.p.n==384;self.band_grid=self.grid//3
        body=self.source_text.replace('CUtensorMap input,weight,output;','CUtensorMap input,weight,output;int band;')
        assert 'int band;' in body
        body=body.replace('blockIdx.x',f'(blockIdx.x+p.band*{self.band_grid})')
        body=body.replace('mw_wide_one_group_transposed_dx','mw_wide_native_band_dx')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDX_FULL_SLOTS=2','-DDX_FULL_K=64','-DDX_FULL_DEPTH=1']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_native_band_dx').kernel('mw_wide_native_band_dx');self.k.set_max_dynamic_smem(self.smem)
        self.band_params=[T._launch_module().Struct([*self.params.fields,i]) for i in range(3)]
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')

    def band(self,i):self.k.launch((self.band_grid,1,1),(160,1,1),[self.band_params[i]],self.smem)


class PersistentGPDX(SpatialGPDX):
    def __init__(self,plan,grid=132):
        super().__init__(plan,True,False);self.grid=grid
        body=self.source_text
        start=body.index(' int tiles=p.N/128;int half=',body.index('extern "C" __global__'))
        end=body.rindex('}')
        tasks=4*plan.p.D*3
        task_body=body[start:end].replace('blockIdx.x','task')
        body=body[:start]+f' for(int task=blockIdx.x;task<{tasks};task+=gridDim.x){{\n'+task_body+'''
  // Every GP store and TMA input transaction is complete before barriers reset.
  __syncthreads();
  if(threadIdx.x==0)for(int i=0;i<7;++i)asm volatile("mbarrier.inval.shared.b64 [%0];"::"r"(smem_u32(bar+i)):"memory");
  __syncthreads();
 }
'''+body[end:]
        body=body.replace('mw_wide_spatial_gp_dx_overlap','mw_wide_persistent_gp_dx_overlap');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={plan.p.D}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_persistent_gp_dx_overlap').kernel('mw_wide_persistent_gp_dx_overlap');self.k.set_max_dynamic_smem(self.smem)
        self.native_dx=NativeBandDX(plan)
        self.dx=[lambda i=i:self.native_dx.band(i) for i in range(3)]
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')+self.native_dx.local_bytes

    def producer(self,i):self.k.launch((self.grid,1,1),(256,1,1),[self.params[i]],self.smem)
