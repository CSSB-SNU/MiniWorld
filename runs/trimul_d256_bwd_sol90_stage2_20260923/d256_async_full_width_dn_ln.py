"""Independent TMA producer and three dNorm groups with exact full-width LN."""
from pathlib import Path
from wide_full_width_dn_ln import FullWidthDnLN
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class AsyncFullWidthDnLN(FullWidthDnLN):
    def __init__(self,plan,emit_dn=False,ln_rows=16):
        super().__init__(plan,emit_dn,ln_rows);assert plan.p.D==256
        root=Path(__file__).resolve().parent
        body=self.source_text.replace('NT=512','NT=384').replace(',WC=H/4','')
        a=body.index('extern "C" __global__ __launch_bounds__(NT,1)');b=body.index('struct Reduce',a)
        pre=body[:a];tail=body[b:];main=body[a:b]
        start=main.index(' extern __shared__')
        main='template<int WG> TMN_DEVI void consume(const Params& p){\n constexpr int WC=WG<2?192:128,CO=WG*192;\n'+main[start:]
        marker=' if(tid==0){for(int i=0;i<3;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();'
        assert main.count(marker)==1;main=main.replace(marker,'')
        main=main.replace('float v[WC/2]={};if(tid==0)load(p,sm,bar,row,col,0,0);','float v[WC/2]={};')
        main=main.replace('mbar_wait(bar+slot,(it/2)&1);__syncthreads();','mbar_wait(bar+slot,(it/2)&1);')
        main=main.replace('   if(tid==0&&ki+32<D)load(p,sm,bar,row,col,ki+32,1-slot);','')
        main=main.replace('wg*WC*64','CO*64').replace('col+wg*WC+','CO+')
        a='mma128_off<q*32,q*2048,0,1>(v,smem_desc(smem_u32(sm+slot*INPUT),16,512,2),smem_desc(smem_u32(sm+slot*INPUT+4096+CO*64),4096,1024,1),it>0||q>0);'
        assert main.count(a)==1
        main=main.replace(a,'if constexpr(WG<2)'+a.replace('mma128_off','mma192_off')+'else '+a)
        marker='});wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();'
        assert main.count(marker)==1
        main=main.replace(marker,'});wgmma_commit();wgmma_wait<0>();fence_regs(v);if(tid%128==0)mbar_arrive(bar+3+slot);')
        main=main.replace('__syncthreads();','named_bar_sync(1,NT);')
        producer='''
TMN_DEVI void produce(const Params& p,uint8_t* sm,uint64_t* bar){
 if(threadIdx.x!=384)return;
 int row=blockIdx.x*64;
 for(int ki=0,it=0;ki<D;ki+=32,++it){int slot=it%2;
  if(it>=2)mbar_wait(bar+3+slot,((it/2)-1)&1);
  load(p,sm,bar,row,0,ki,slot);
 }
}
extern "C" __global__ __launch_bounds__(512,1)
void mw_d256_async_full_width_dn_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 if(threadIdx.x==0){for(int i=0;i<5;++i)mbar_init(bar+i,i>=3?3:1);fence_barrier_init();}
 __syncthreads();
 if(threadIdx.x<384){
  setmaxnreg_inc<160>();
  if(threadIdx.x<128)consume<0>(p);else if(threadIdx.x<256)consume<1>(p);else consume<2>(p);
 }else{setmaxnreg_dec<32>();produce(p,sm,bar);}
}
'''
        body=pre+(root/'mma192_offset.cuh').read_text()+'\n'+main+producer+tail
        self.source_text=body;self.threads=512
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=256',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_d256_async_full_width_dn_ln')
        self.k=unit.kernel('mw_d256_async_full_width_dn_ln');self.k.set_max_dynamic_smem(self.smem)
        self.reduce_first=unit.kernel('mw_wide_stream_ln_reduce_first');self.reduce_last=unit.kernel('mw_wide_stream_ln_reduce_last')
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,512,self.smem)))
        assert self.registers>=128,'Runtime role budgets require128 initial registers per thread'
