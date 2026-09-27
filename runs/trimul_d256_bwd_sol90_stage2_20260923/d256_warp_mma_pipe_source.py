"""Two warp-local dW groups overlap the existing projection/GLU producer."""
from d256_compact_four_source import CompactFourSource
from three_role_register_pool import initial_pool_three
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class WarpMMAPipeSource:
    def __init__(self,plan,consumer=88,opt=3,static_slots=True,barrier_batch=0):
        prior=CompactFourSource(plan,64,96,static_slots,False,False,True,False,static_slots)
        self.__dict__.update(prior.__dict__);body=prior.source_text
        helper='''
TMN_DEVI void warp_dw(float (&dw)[64],uint8_t* dg,uint8_t* xn){
 int lane=threadIdx.x%32,warp=(threadIdx.x/32)%4;
 static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
  uint32_t a[4];int mat=lane/8;
  ldsm_x4(a,smem_u32(dg)+swz128(warp*16+lane%8+8*(mat&1),k*32+16*(mat/2)));
  static_for<16>([&](auto nn){constexpr int n=decltype(nn)::value;
   uint32_t b0,b1,addr=smem_u32(xn)+(n/8)*8192+swz128(k*16+lane%16,(n%8)*16);
   asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];":"=r"(b0),"=r"(b1):"r"(addr):"memory");
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(dw[n*4]),"+f"(dw[n*4+1]),"+f"(dw[n*4+2]),"+f"(dw[n*4+3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
   #if WARP_LOAD_BATCH > 0
   if constexpr ((n+1)%WARP_LOAD_BATCH==0) __syncwarp();
   #endif
  });
 });
}
'''
        body=body.replace('template<int WG> TMN_DEVI void consume(',helper+'template<int WG> TMN_DEVI void consume(')
        start=body.index('  fence_regs(dw0);wgmma_fence();',body.index('template<int WG> TMN_DEVI void consume'))
        end=body.index('wgmma_wait<0>();fence_regs(dw0);',start)+len('wgmma_wait<0>();fence_regs(dw0);')
        body=body[:start]+'  warp_dw(dw0,dg,xn+WG*16384);'+body[end:]
        body=body.replace('threadIdx.x','source_tid()')
        getter='TMN_DEVI unsigned source_tid(){unsigned v;asm volatile("mov.u32 %0, %%tid.x;":"=r"(v)::"memory");return v;}\n'
        body=body.replace('constexpr int D=256',getter+'constexpr int D=256')
        body=body.replace('mw_d256_compact_four_source','mw_d256_warp_mma_pipe_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',f'-Xptxas=--opt-level={opt}','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}','-DPRODUCER_REGS=64',f'-DCONSUMER_REGS={consumer}']
        flags.append(f'-DWARP_LOAD_BATCH={barrier_batch}')
        cubin=T.compile_text(body,flags);initial=((64+2*consumer+23)//24)*8
        self.cubin,self.pool_metadata=initial_pool_three(cubin,'mw_d256_warp_mma_pipe_source',initial,64,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_warp_mma_pipe_source').kernel('mw_d256_warp_mma_pipe_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(384,1,1),[self.params],self.smem)
