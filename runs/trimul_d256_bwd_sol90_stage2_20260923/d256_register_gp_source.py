"""Load each GP A fragment once and reuse it across both dW column halves."""
from pathlib import Path
from d256_deep_pair_source import DeepPairSource
from wide_mask_transform import mask_stage
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class RegisterGPSource:
    def __init__(self,plan,paired=False,slots=3,phase=0):
        p=plan.p;prior=plan.b7;root=Path(__file__).resolve().parent
        if paired:
            base=DeepPairSource(p,prior,False,phase,slots);self.__dict__.update(base.__dict__)
            body=base.source_text;gp='gp';self.threads=384;self.grid=16*prior.splits;original_name='mw_d256_deep_pair_source'
        else:
            body=mask_stage(prior.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
            gp='sm+DERIV';self.params=prior.params;self.smem=prior.smem;self.threads=256;self.grid=32*prior.splits;original_name='mw_d256_b7_tma'
        body=body.replace('constexpr int D=256',(root/'mma_rs_wide.cuh').read_text()+'\nconstexpr int D=256')
        marker='  fence_regs(dw0);fence_regs(dw1);wgmma_fence();'
        assert body.count(marker)==1
        body=body.replace(marker,f'''  uint32_t af[4][4];
  load_frag_bf16<4,8192>(af,smem_u32({gp}),warp*16,lane);
  static_for<4>([&](auto qq){{fence_regs(af[decltype(qq)::value]);}});
'''+marker)
        for half in (0,1):
            old=f'mma128_off<k*32,k*2048,0,1>(dw{half},a,'
            assert body.count(old)==1
            body=body.replace(old,f'mma128_rs_off<k*2048,1>(dw{half},af[k],')
        marker='wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'static_for<4>([&](auto qq){fence_regs(af[decltype(qq)::value]);});')
        body=body.replace(original_name,'mw_d256_register_gp_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={prior.splits}',f'-DPAIR_SLOTS={slots}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_register_gp_source').kernel('mw_d256_register_gp_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        assert self.registers*self.threads>=128*(32+224*(2 if paired else 1)),'insufficient dynamic register pool'

    def __call__(self):self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
