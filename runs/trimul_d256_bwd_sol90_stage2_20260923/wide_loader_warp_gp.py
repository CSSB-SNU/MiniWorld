"""One loader warp feeds two compute warpgroups, retaining two resident CTAs."""
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class LoaderWarpGP:
    def __init__(self,plan,unroll=False,whole=False):
        if whole:
            from wide_whole_tma_gp import WholeTmaGP
            prior=WholeTmaGP(plan,True)
        else:prior=plan.contract_gp
        self.__dict__.update(prior.__dict__)
        assert self.p.n==384
        body=prior.source_text
        marker='template<int MODE> TMN_DEVI void consume('
        producer='''template<int MODE> TMN_DEVI void produce_small(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 if(threadIdx.x==256){
  for(int it=0;it<p.N/64;++it){
   int slot=it%SLOTS;
   if(it>=SLOTS)mbar_wait(bar+SLOTS+slot,((it/SLOTS)-1)&1);
   load_input<MODE>(p,sm,bar,ch,mi,ni,it*64,slot);
  }
  // Each final input slot is reusable only after both MMA owners retire it.
  mbar_wait(bar+SLOTS,(p.N/64/SLOTS-1)&1);
  load_epi<MODE,0>(p,sm,bar,ch,mi,ni);
  mbar_wait(bar+SLOTS+1,(p.N/64/SLOTS-1)&1);
  load_epi<MODE,1>(p,sm,bar,ch,mi,ni);
  mbar_wait(bar+SLOTS+2,(p.N/64/SLOTS-1)&1);
  load_epi<MODE,2>(p,sm,bar,ch,mi,ni);
 }
}
'''
        assert body.count(marker)==1;body=body.replace(marker,producer+marker)
        marker=' if(threadIdx.x==0){load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);load_input<MODE>(p,sm,bar,ch,mi,ni,64,1);}'
        assert body.count(marker)==1;body=body.replace(marker,'')
        start=body.index('int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);__syncthreads();')
        end=body.index('  fence_regs(v);wgmma_fence();',start)
        body=body[:start]+'int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);named_bar_sync(1+WG,128);\n'+body[end:]
        marker='});wgmma_commit();wgmma_wait<0>();fence_regs(v);'
        assert body.count(marker)==1;body=body.replace(marker,marker+'\n  named_bar_sync(1+WG,128);if(tid==0)mbar_arrive(bar+SLOTS+slot);')
        start=body.index(' // Both compute groups have now retired slot2.')
        end=body.index(' static_for<32>',start)
        body=body[:start]+' mbar_wait(bar+6,0);named_bar_sync(3,256);\n'+body[end:]
        marker=' consume<MODE>(p,sm,bar,ch,mi,ni);'
        assert body.count(marker)==1
        body=body.replace(marker,''' if(threadIdx.x>=256){produce_small<MODE>(p,sm,bar,ch,mi,ni);return;}
 consume<MODE>(p,sm,bar,ch,mi,ni);''')
        marker=' fence_proxy_async();__syncthreads();'
        assert body.count(marker)==1;body=body.replace(marker,' fence_proxy_async();named_bar_sync(3,256);')
        body=body.replace('__launch_bounds__(256,2)','__launch_bounds__(288,2)')
        if unroll:
            body=body.replace('for(int ki=0,it=0;ki<p.N;ki+=64,++it)', '#pragma unroll\n for(int ki=0,it=0;ki<384;ki+=64,++it)')
            body=body.replace('for(int it=0;it<p.N/64;++it)', '#pragma unroll\n  for(int it=0;it<6;++it)')
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_loader_warp_gp').replace('mw_wide_whole_tma_gp','mw_wide_loader_warp_gp');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_loader_warp_gp').kernel('mw_wide_loader_warp_gp');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,288,self.smem)))

    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(288,1,1),[self.params],self.smem)
