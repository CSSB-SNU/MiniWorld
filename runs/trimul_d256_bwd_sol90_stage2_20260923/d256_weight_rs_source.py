"""Transpose projection so cached global weight fragments feed register-A WGMMA."""
from d256_whole_aliased_source import WholeAliasedSource
from two_role_register_pool import initial_pool_roles
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class WeightRSSource:
    def __init__(self,plan,producer=80,chunk=64,runtime_chunks=False):
        prior=WholeAliasedSource(plan,producer);self.__dict__.update(prior.__dict__)
        consumer=256-producer;body=prior.source_text
        assert chunk in (64,128)
        body=body.replace('WEIGHT=81920,BAR=114688','PRE=81920,BAR=90112')
        body=body.replace('float* part;int M;','float* part;const bf* raww;int M;')
        start=body.index(' if(tid==0){mbar_arrive_expect_tx(bar+2,CH);');end=body.index(' for(int tile=begin',start)
        body=body[:start]+''' if(tid==0)load_input(p,sm,bar,begin*64,rank,0);
''' +body[end:]
        start=body.index('  float pre[32]={};');end=body.index('  int ra=w*16+lane/4;',start)
        body=body[:start]+f'''  float pre[32]={{}};fence_regs(pre);wgmma_fence();
  static_for<{256//chunk}>([&](auto hh){{constexpr int h=decltype(hh)::value;
   uint32_t frag[{chunk//16}][4];
   static_for<{chunk//16}>([&](auto qq){{constexpr int q=decltype(qq)::value;
    static_for<4>([&](auto jj){{constexpr int j=decltype(jj)::value;
     int r=w*16+lane/4+8*(j&1),c=h*{chunk}+q*16+2*(lane%4)+8*(j/2);
     frag[q][j]=__ldg(reinterpret_cast<const uint32_t*>(p.raww+(rank*64+r)*D+c));
    }});fence_regs(frag[q]);
   }});
   uint64_t desc=smem_desc(smem_u32(xn+h*{chunk//64}*8192),16,1024,1);fence_regs(pre);wgmma_fence();
   static_for<{chunk//16}>([&](auto qq){{constexpr int q=decltype(qq)::value;
    wgmma_m64n64k16_rs_off<(q/4)*8192+(q%4)*32>(pre,frag[q],uint32_t(desc),uint32_t(desc>>32),h>0||q>0);
   }});wgmma_commit();wgmma_wait<0>();
   static_for<{chunk//16}>([&](auto qq){{fence_regs(frag[decltype(qq)::value]);}});fence_regs(pre);
  }});
  if(tile+1<end){{
   if(it>=1)mbar_wait(bar+5+(1-slot),((it-1)/2)&1);
   if(tid==0)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);
  }}
  static_for<4>([&](auto qq){{constexpr int q=decltype(qq)::value;
   uint32_t values[4];static_for<4>([&](auto jj){{constexpr int j=decltype(jj)::value;values[j]=pack_bf16(pre[q*8+j*2],pre[q*8+j*2+1]);}});
   uint32_t off=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);
   stsm_x4_t(smem_u32(sm+PRE)+off,values[0],values[1],values[2],values[3]);
  }});named_bar_sync(1,128);
''' +body[end:]
        old='uint32_t gr=pack_bf16(pre[q*8+j*2],pre[q*8+j*2+1]),pr=pack_bf16(pre[(q+2)*8+j*2],pre[(q+2)*8+j*2+1]);'
        assert body.count(old)==1
        body=body.replace(old,'''int r=w*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j/2);
    uint32_t gr=*reinterpret_cast<uint32_t*>(sm+PRE+swz128(r,c*2)),pr=*reinterpret_cast<uint32_t*>(sm+PRE+swz128(r,(c+32)*2));''')
        if runtime_chunks:
            body=body.replace(f'static_for<{256//chunk}>([&](auto hh){{constexpr int h=decltype(hh)::value;',f'\n  #pragma unroll 1\n  for(int h=0;h<{256//chunk};++h){{')
            marker=f'   static_for<{chunk//16}>([&](auto qq){{fence_regs(frag[decltype(qq)::value]);}});fence_regs(pre);\n  }});'
            assert body.count(marker)==1
            body=body.replace(marker,marker[:-3]+'}')
        body=body.replace('mw_d256_whole_aliased_source','mw_d256_weight_rs_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool_roles(cubin,'mw_d256_weight_rs_source',producer,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_weight_rs_source').kernel('mw_d256_weight_rs_source');self.smem=90496;self.k.set_max_dynamic_smem(self.smem)
        fields=prior.params.fields.copy();fields.insert(-1,plan.p.w1);self.params=T._launch_module().Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
