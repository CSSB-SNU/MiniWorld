"""A dedicated TMA producer feeds both full-N384 contraction compute groups."""
from d256_full_spatial_contract import FullSpatialContract
from multi_group_initial_pool import initial_pool_multi
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class ProducerSpatialContract(FullSpatialContract):
    def __init__(self,plan,consumer=232):
        super().__init__(plan,2,3,256)
        assert consumer in (224,232,240)
        body=self.source_text
        start=body.index('template<int MODE> TMN_DEVI void run(')
        end=body.index('extern "C" __global__',start)
        prefix,run,suffix=body[:start],body[start:end],body[end:]
        run=run.replace('wg=threadIdx.x/128','wg=threadIdx.x/128-1')
        marker=' if(threadIdx.x==0)for(int step=0;step<SLOTS-1;++step)load<MODE>(p,sm,bar,ch,mi,ni,step);'
        assert run.count(marker)==1
        run=run.replace(marker,f''' if(threadIdx.x<128){{
  setmaxnreg_dec<32>();
  if(tid==0)for(int step=0;step<6;++step){{int slot=step%SLOTS;
   if(step>=SLOTS)mbar_wait(bar+SLOTS+slot,((step/SLOTS)-1)&1);
   load<MODE>(p,sm,bar,ch,mi,ni,step);
  }}
  return;
 }}
 setmaxnreg_inc<{consumer}>();''')
        marker='  if(threadIdx.x==0 && step+SLOTS-1<6)load<MODE>(p,sm,bar,ch,mi,ni,step+SLOTS-1);'
        assert run.count(marker)==1;run=run.replace(marker,'')
        run=run.replace('__syncthreads();','named_bar_sync(wg+1,128);')
        marker='});wgmma_commit();wgmma_wait<0>();fence_regs(v0);fence_regs(v1);'
        assert run.count(marker)==1
        run=run.replace(marker,marker+'named_bar_sync(wg+1,128);if(tid==0)mbar_arrive(bar+SLOTS+slot);')
        marker=' }\n named_bar_sync(wg+1,128);\n static_for<'
        assert run.count(marker)==1
        run=run.replace(marker,' }\n named_bar_sync(3,256);\n static_for<')
        marker=' if(threadIdx.x==0){\n  #pragma unroll\n  for(int wm=0;wm<ROW_GROUPS;++wm){'
        assert run.count(marker)==1
        run=run.replace(marker,' if(tid==0){\n  {int wm=wg;')
        suffix=suffix.replace('__launch_bounds__(NT,1)',f'__maxnreg__({consumer})')
        suffix=suffix.replace('__launch_bounds__(NT,MIN_BLOCKS)',f'__maxnreg__({consumer})')
        assert '__maxnreg__' in suffix
        marker='for(int i=0;i<SLOTS;++i)mbar_init(bar+i,1);'
        assert suffix.count(marker)==1
        suffix=suffix.replace(marker,'for(int i=0;i<2*SLOTS;++i)mbar_init(bar+i,i<SLOTS?1:2);')
        # Keep mode dispatch as direct branches so every allocation path can be
        # checked conservatively before the resource metadata is adjusted.
        dispatch='''TMN_DEVI bool mode_is(int mode,int expected){
 int equal;asm volatile("{.reg .pred q;setp.eq.s32 q,%1,%2;selp.u32 %0,1,0,q;}"
 :"=r"(equal):"r"(mode),"r"(expected));return equal;
}
'''
        prefix+=dispatch
        for mode in range(3):suffix=suffix.replace(f'if(mode=={mode})',f'if(mode_is(mode,{mode}))')
        body=(prefix+run+suffix).replace('mw_d256_full_spatial_contract','mw_d256_producer_spatial_contract');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DROW_GROUPS=2','-DGRID_ORDER=1','-DMIN_BLOCKS=1','-DCONTRACT_SLOTS=3']
        cubin=T.compile_text(body,flags)
        initial=((32+2*consumer+23)//24)*8
        self.cubin,self.pool_metadata=initial_pool_multi(cubin,'mw_d256_producer_spatial_contract',initial,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_producer_spatial_contract').kernel('mw_d256_producer_spatial_contract')
        self.threads=384;self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        assert self.occupancy>0 and self.registers*self.threads>=128*(32+2*consumer)
