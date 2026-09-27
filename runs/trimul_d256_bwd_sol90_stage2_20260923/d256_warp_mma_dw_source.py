"""Warp-local m16n8k16 dW accumulation with the selected projection and GLU."""
from initial_register_pool import initial_pool
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class WarpMMADWSource:
    def __init__(self,plan,consumer=208):
        prior=plan.pool_source;self.__dict__.update(prior.__dict__);body=prior.source_text
        helper='''
TMN_DEVI void warp_dw(float (&dw)[128],uint8_t* dg,uint8_t* xn){
 int lane=threadIdx.x%32,warp=(threadIdx.x/32)%4;
 static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
  uint32_t a[4];int mat=lane/8;
  ldsm_x4(a,smem_u32(dg)+swz128(warp*16+lane%8+8*(mat&1),k*32+16*(mat/2)));
  static_for<32>([&](auto nn){constexpr int n=decltype(nn)::value;
   uint32_t b0,b1,addr=smem_u32(xn)+(n/8)*8192+swz128(k*16+lane%16,(n%8)*16);
   asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];":"=r"(b0),"=r"(b1):"r"(addr):"memory");
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(dw[n*4]),"+f"(dw[n*4+1]),"+f"(dw[n*4+2]),"+f"(dw[n*4+3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  });
 });
}
'''
        body=body.replace('TMN_DEVI void compute_source',helper+'\nTMN_DEVI void compute_source')
        start=body.index('  fence_regs(dw);wgmma_fence();',body.index('TMN_DEVI void compute_source'))
        end=body.index('wgmma_wait<0>();fence_regs(dw);',start)+len('wgmma_wait<0>();fence_regs(dw);')
        body=body[:start]+'  warp_dw(dw,sm+DERIV,xn);'+body[end:]
        body=body.replace('setmaxnreg_inc<208>()',f'setmaxnreg_inc<{consumer}>()')
        # Prevent the initial role-selection thread ID from staying live inside
        # the large synchronous dW accumulator's register allocation.
        body=body.replace('threadIdx.x','source_tid()')
        helper='TMN_DEVI unsigned source_tid(){unsigned v;asm volatile("mov.u32 %0, %%tid.x;":"=r"(v)::"memory");return v;}\n'
        body=body.replace('constexpr int D=256',helper+'constexpr int D=256')
        body=body.replace('mw_d256_register_budget_source','mw_d256_warp_mma_dw_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool(cubin,'mw_d256_warp_mma_dw_source',(32+consumer)//2,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_warp_mma_dw_source').kernel('mw_d256_warp_mma_dw_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
