"""Warp-local projection feeds two WGMMA dW groups with a smaller producer pool."""
from d256_compact_four_source import CompactFourSource
from three_role_register_pool import initial_pool_three
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class WarpProjectionPipeSource:
    def __init__(self,plan,producer=48,runtime_k=False,batch=0,loader_first=False,two_warps=False):
        prior=CompactFourSource(plan,64,96,False,False,False,True,False,False)
        self.__dict__.update(prior.__dict__);body=prior.source_text
        helper='''
TMN_DEVI void warp_projection(float (&pre)[32],uint8_t* xn,uint8_t* weight){
 int lane=threadIdx.x%32,warp=(threadIdx.x/32)%4,mat=lane/8;
 #if RUNTIME_K
 #pragma unroll 1
 for(int k=0;k<16;++k){
 #else
 static_for<16>([&](auto kk){constexpr int k=decltype(kk)::value;
 #endif
  uint32_t a[4];
  ldsm_x4(a,smem_u32(xn)+(k/4)*8192+swz128(warp*16+lane%8+8*(mat&1),(k%4)*32+16*(mat/2)));
  static_for<8>([&](auto nn){constexpr int n=decltype(nn)::value;
   uint32_t b0,b1,addr=smem_u32(weight)+(k/4)*8192+swz128(n*8+lane%8,(k%4)*32+16*(mat&1));
   asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];":"=r"(b0),"=r"(b1):"r"(addr):"memory");
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(pre[n*4]),"+f"(pre[n*4+1]),"+f"(pre[n*4+2]),"+f"(pre[n*4+3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
   #if OPERAND_BATCH > 0
   if constexpr ((n+1)%OPERAND_BATCH==0) __syncwarp();
   #endif
  });
 #if RUNTIME_K
 }
 #else
 });
 #endif
}
'''
        body=body.replace('TMN_DEVI void produce(',helper+'TMN_DEVI void produce(')
        start=body.index('  float pre[32]={};');end=body.index('  if(tile+1<end){',start)
        body=body[:start]+'  float pre[32]={};warp_projection(pre,xn,sm+WEIGHT);\n'+body[end:]
        body=body.replace('  wgmma_wait<0>();fence_regs(pre);','')
        if loader_first or two_warps:
            start=body.index('  if(tile+1<end){',body.index('TMN_DEVI void produce('))
            end=body.index('  uint32_t packed_pre',start)
            prefetch=body[start:end];body=body[:start]+body[end:]
            body=body.replace('  float pre[32]={};warp_projection',prefetch+'  float pre[32]={};warp_projection')
        if two_warps:
            body=body.replace('uint8_t* weight){','uint8_t* weight,int row_half){')
            body=body.replace('warp=(threadIdx.x/32)%4,mat=lane/8;','warp=(threadIdx.x/32)%2+row_half*2,mat=lane/8;')
            start=body.index('TMN_DEVI void produce(');end=body.index('template<int WG> TMN_DEVI void consume(',start)
            fragment=body[start:end]
            fragment=fragment.replace('tid=threadIdx.x%128,lane=tid%32,w=tid/32,mat=lane/8','tid=threadIdx.x%64,lane=tid%32,mat=lane/8')
            fragment=fragment.replace('named_bar_sync(1,128)','named_bar_sync(1,64)')
            fragment=fragment.replace('  float pre[32]={};warp_projection(pre,xn,sm+WEIGHT);','  static_for<2>([&](auto hh){constexpr int half=decltype(hh)::value;int w=tid/32+half*2;\n  float pre[32]={};warp_projection(pre,xn,sm+WEIGHT,half);')
            fragment=fragment.replace('  named_bar_sync(1,64);fence_proxy_async();','  });\n  named_bar_sync(1,64);fence_proxy_async();')
            body=body[:start]+fragment+body[end:]
            start=body.index(' if(threadIdx.x<128){',body.index('extern "C" __global__'))
            body=body[:start]+''' if(threadIdx.x>=256){produce(p,sm,bar);}
 else{if(threadIdx.x<128)consume<0>(p,sm,bar);else consume<1>(p,sm,bar);}
}
'''
        body=body.replace('threadIdx.x','source_tid()')
        getter='TMN_DEVI unsigned source_tid(){unsigned v;asm volatile("mov.u32 %0, %%tid.x;":"=r"(v)::"memory");return v;}\n'
        body=body.replace('constexpr int D=256',getter+'constexpr int D=256')
        body=body.replace('mw_d256_compact_four_source','mw_d256_warp_projection_pipe_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}',f'-DPRODUCER_REGS={producer}','-DCONSUMER_REGS=96',f'-DRUNTIME_K={int(runtime_k)}',f'-DOPERAND_BATCH={batch}']
        cubin=T.compile_text(body,flags);initial=((producer+192+23)//24)*8
        if two_warps:
            self.cubin=cubin;self.pool_metadata={'uniform_registers':96,'threads':320}
        else:
            self.cubin,self.pool_metadata=initial_pool_three(cubin,'mw_d256_warp_projection_pipe_source',initial,producer,96)
        self.k=T.load_unit(str(self.cubin),'mw_d256_warp_projection_pipe_source').kernel('mw_d256_warp_projection_pipe_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.threads=320 if two_warps else 384
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(self.threads,1,1),[self.params],self.smem)
