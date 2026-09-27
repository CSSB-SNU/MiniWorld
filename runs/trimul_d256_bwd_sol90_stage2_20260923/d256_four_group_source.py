"""Separate TMA issuer from GP producer and the two dW consumers."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class D256FourGroupSource:
    def __init__(self,p,original,async_dw=False):
        self.p=p;self.original=original;self.splits=original.splits
        root=Path(__file__).resolve().parent;prior=root.parent/'trimul_d256_bwd_sol90_20260923'
        helpers=(root/'wide_source.cu').read_text().split('extern "C" __global__')[0]
        helpers=helpers.replace('BAR=DERIV+8192','BAR=DERIV+16384')
        helpers=helpers.replace('// MMA_HELPERS',(prior/'mma.cuh').read_text())
        helpers=helpers.replace('// PACKED_GLU',(root/'packed_glu.cuh').read_text().replace('s+32768','s+CH'))
        body=(root/'wide_pipe_source.cu').read_text().replace('// SOURCE_HELPERS',helpers)
        weight=''' if(tid==0){mbar_arrive_expect_tx(bar+2,CH);
  for(int c=0;c<D/64;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
 }
'''
        assert body.count(weight)==1;body=body.replace(weight,'')
        issue='  if(it>=2)mbar_wait(bar+5+slot,((it/2)-1)&1);\n  if(tid==0)load_input(p,sm,bar,row,rank,slot);'
        assert body.count(issue)==1;body=body.replace(issue,'')
        body=body.replace('int tid=threadIdx.x,lane=tid%32,warp=tid/32;','int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;')
        from wide_offset_source import offsets
        from wide_mask_transform import mask_stage
        body=mask_stage(offsets(body),'prefetch')
        if async_dw:
            fence='  static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(dw[c]);});wgmma_fence();'
            assert body.count(fence)==1;body=body.replace(fence,'')
            body=body.replace(' float dw[NC][32]={};',' float dw[NC][32]={};\n'+fence)
            marker='''  });wgmma_commit();wgmma_wait<0>();
  static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(dw[c]);});
  named_bar_sync(2+WG,128);if(tid==0)mbar_arrive(bar+5+slot);
 }
 static_for<NC>'''
            replacement='''  });wgmma_commit();
  if(it>0){wgmma_wait<1>();named_bar_sync(2+WG,128);if(tid==0)mbar_arrive(bar+5+(1-slot));}
 }
 wgmma_wait<0>();
 static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(dw[c]);});
 named_bar_sync(2+WG,128);
 if(tid==0 && end>begin)mbar_arrive(bar+5+(end-begin-1)%2);
 static_for<NC>'''
            assert body.count(marker)==1;body=body.replace(marker,replacement)
        a=body.index('extern "C" __global__ __launch_bounds__(384,1)')
        body=body[:a]+'''
extern "C" __global__ __launch_bounds__(512,1)
void mw_wide_four_group_source(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 if(threadIdx.x==0){for(int b=0;b<7;++b)mbar_init(bar+b,b>=5?2:1);fence_barrier_init();}
 __syncthreads();
 if(threadIdx.x<128){
  setmaxnreg_dec<32>();
  if(threadIdx.x==0){
   int rank=blockIdx.x%RANKS,split=blockIdx.x/RANKS,tiles=p.M/64;
   int begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
   mbar_arrive_expect_tx(bar+2,CH);
   for(int c=0;c<D/64;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
   for(int tile=begin,it=0;tile<end;++tile,++it){
    int slot=it%2;if(it>=2)mbar_wait(bar+5+slot,((it/2)-1)&1);
    load_input(p,sm,bar,tile*64,rank,slot);
   }
  }
 }else if(threadIdx.x<256){setmaxnreg_dec<96>();pipe_produce(p,sm,bar);}
 else{setmaxnreg_inc<192>();if(threadIdx.x<384)pipe_consume<0>(p,sm,bar);else pipe_consume<1>(p,sm,bar);}
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}',f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_four_group_source').kernel('mw_wide_four_group_source')
        self.smem=original.source_smem+8192;self.k.set_max_dynamic_smem(self.smem)
    def __call__(self):self.k.launch((self.p.D//8*self.splits,1,1),(512,1,1),[self.original.params],self.smem)
