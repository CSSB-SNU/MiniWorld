"""Issue future K32 inputs after only the overwritten slot's MMA has retired."""
from d256_k32_n192_contract import K32N192Contract
from d256_k32_full_spatial_contract import K32FullSpatialContract
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class PipelinedK32Contract:
    def __init__(self,plan,full=False,slots=4,depth=1):
        base=K32FullSpatialContract(plan,slots) if full else K32N192Contract(plan,2,slots)
        self.__dict__.update(base.__dict__)
        if depth==0:return
        assert 1<=depth<=2 and slots>=depth+2
        body=self.source_text
        fence='fence_regs(v0);fence_regs(v1);' if full else 'fence_regs(v);'
        declaration='float v0[128]={},v1[64]={};' if full else 'float v[96]={};'
        assert body.count(declaration)==1
        body=body.replace(fence+'wgmma_fence();','').replace('wgmma_wait<0>();'+fence,'')
        body=body.replace(declaration,declaration+fence+'wgmma_fence();')
        body=body.replace('step<SLOTS-1;',f'step<SLOTS-{depth};')
        marker='  if(threadIdx.x==0 && step+SLOTS-1<12)load<MODE>(p,sm,bar,ch,mi,ni,step+SLOTS-1);'
        assert body.count(marker)==1;body=body.replace(marker,'')
        marker='});wgmma_commit();'
        assert body.count(marker)==1
        body=body.replace(marker,marker+f'''wgmma_wait<{depth}>();__syncthreads();
  if(threadIdx.x==0&&step+SLOTS-{depth}<12)load<MODE>(p,sm,bar,ch,mi,ni,step+SLOTS-{depth});''')
        marker=' }\n __syncthreads();\n static_for<'
        assert body.count(marker)==1
        body=body.replace(marker,' }\n wgmma_wait<0>();'+fence+'__syncthreads();\n static_for<')
        original='mw_d256_k32_full_spatial_contract' if full else 'mw_d256_k32_n192_contract'
        body=body.replace(original,'mw_d256_pipelined_k32_contract');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DROW_GROUPS={1 if full else 2}','-DGRID_ORDER=1','-DMIN_BLOCKS=2',f'-DCONTRACT_SLOTS={slots}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_pipelined_k32_contract').kernel('mw_d256_pipelined_k32_contract');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))

    def __call__(self):self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
