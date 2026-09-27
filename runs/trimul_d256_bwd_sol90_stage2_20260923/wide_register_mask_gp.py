"""Read the shared mask through L1 while the penultimate contraction runs."""
from wide_staged_epilogue_gp import StagedEpilogueGP
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class RegisterMaskGP:
    def __init__(self,plan):
        prior=StagedEpilogueGP(plan);self.__dict__.update(prior.__dict__)
        body=prior.source_text
        marker='float v[64]={};';assert body.count(marker)==1
        body=body.replace(marker,marker+'uint32_t masks[32];')
        marker='});wgmma_commit();wgmma_wait<0>();fence_regs(v);'
        assert body.count(marker)==1
        body=body.replace(marker,''' });wgmma_commit();
  if(ki==p.N-128){
   static_for<32>([&](auto jj){constexpr int q=decltype(jj)::value,j=q*2;
    int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
    const uint32_t* ptr=reinterpret_cast<const uint32_t*>(p.mask+size_t(mi+WG*64+r)*p.N+ni+c);
    asm volatile("ld.global.ca.u32 %0,[%1];":"=r"(masks[q]):"l"(ptr):"memory");
   });
  }
  wgmma_wait<0>();fence_regs(v);''')
        marker='mbar_arrive_expect_tx(bar+6,98304);';assert body.count(marker)==1
        body=body.replace(marker,'mbar_arrive_expect_tx(bar+6,65536);')
        marker=' if(threadIdx.x==0)load_epi<MODE,2>(p,sm,bar,ch,mi,ni);';assert body.count(marker)==1
        body=body.replace(marker,'')
        marker='mask=*reinterpret_cast<uint32_t*>(sm+65536+off)';assert body.count(marker)==1
        body=body.replace(marker,'mask=masks[j/2]')
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_register_mask_gp')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_register_mask_gp').kernel('mw_wide_register_mask_gp');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
