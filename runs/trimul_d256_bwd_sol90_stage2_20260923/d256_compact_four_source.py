"""Aliased two-slot source with independent loader, GLU and two N128 dW groups."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from four_role_register_pool import initial_pool_four


class CompactFourSource:
    def __init__(self,plan,producer=56,consumer=88,static_slots=False,narrow_projection=False,split_dw=False,combined_loader=False,producer_slots=False,defer_output=False,runtime_group=False):
        p=plan.p;self.splits=plan.b7.splits;root=Path(__file__).resolve().parent
        assert (p.D,p.n)==(256,384)
        body=(root/'d256_aliased_pipe_source.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        if not combined_loader:
            start=body.index(' if(tid==0){mbar_arrive_expect_tx(bar+2,CH);');end=body.index(' mbar_wait(bar+2,0);',start)
            body=body[:start]+body[end:]
            start=body.index('  if(tile+1<end){');end=body.index('  wgmma_wait<0>();fence_regs(pre);',start)
            body=body[:start]+body[end:]
        else:
            body=body.replace('for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);','tma_load_3d(sm+WEIGHT,&p.w,bar+2,0,rank*64,0);')
        body=body.replace('int tid=threadIdx.x,lane=tid%32,w=tid/32','int tid=threadIdx.x%128,lane=tid%32,w=tid/32')
        marker='  wgmma_wait<0>();fence_regs(pre);'
        body=body.replace(marker,marker+'\n  uint32_t packed_pre[16];static_for<16>([&](auto jj){constexpr int j=decltype(jj)::value;packed_pre[j]=pack_bf16(pre[2*j],pre[2*j+1]);});')
        body=body.replace('pack_bf16(pre[q*8+j*2],pre[q*8+j*2+1])','packed_pre[q*4+j]').replace('pack_bf16(pre[(q+2)*8+j*2],pre[(q+2)*8+j*2+1])','packed_pre[(q+2)*4+j]')
        if narrow_projection:
            body=body.replace('TMN_DEVI void load_input(', (root/'mma32_offset.cuh').read_text()+'\nTMN_DEVI void load_input(')
            start=body.index('  float pre[32]={};');end=body.index('\n  int ra=w*16+lane/4;',start)
            body=body[:start]+'''  uint32_t packed_pre[16];
  static_for<2>([&](auto hh){constexpr int half=decltype(hh)::value;
   float pre[16]={};fence_regs(pre);wgmma_fence();
   static_for<16>([&](auto kk){constexpr int k=decltype(kk)::value;
    mma32_off<(k/4)*8192+(k%4)*32,(k/4)*8192+(k%4)*32+half*4096,0,0>(pre,smem_desc(smem_u32(xn),16,1024,1),smem_desc(smem_u32(sm+WEIGHT),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(pre);
   static_for<8>([&](auto jj){constexpr int j=decltype(jj)::value;packed_pre[half*8+j]=pack_bf16(pre[2*j],pre[2*j+1]);});
  });
''' +body[end:]
            if combined_loader:
                marker='\n  int ra=w*16+lane/4;'
                body=body.replace(marker,'''  if(tile+1<end){
   if(it>=1)mbar_wait(bar+5+(1-slot),((it-1)/2)&1);
   if(tid==0)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);
  }
''' +marker)
        if not combined_loader:
            body=body.replace('tma_store_commit();mbar_arrive(bar+3+slot);tma_store_wait_all();','tma_store_commit();mbar_arrive(bar+3+slot);tma_store_wait_all();mbar_arrive(bar+5+slot);')
        if producer_slots:
            start=body.index('TMN_DEVI void produce(');end=body.index('TMN_DEVI void consume(',start)
            fragment=body[start:end]
            marker=' for(int tile=begin,it=0;tile<end;++tile,++it){\n  int slot=it%2,row=tile*64;'
            assert fragment.count(marker)==1
            fragment=fragment.replace(marker,' for(int base=0;base<end-begin;base+=2){static_for<2>([&](auto ss){constexpr int slot=decltype(ss)::value;int it=base+slot,tile=begin+it;\n  int row=tile*64;')
            assert fragment.count('\n }\n}')==1
            fragment=fragment.replace('\n }\n}','\n });}\n}')
            body=body[:start]+fragment+body[end:]
        body=body.replace('TMN_DEVI void consume(','template<int WG> TMN_DEVI void consume(')
        body=body.replace('float dw0[64]={},dw1[64]={};','float dw0[64]={};').replace('fence_regs(dw0);fence_regs(dw1);','fence_regs(dw0);')
        old='   mma128_off<k*32,k*2048,0,1>(dw1,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+16384),8192,1024,1),it>0||k>0);'
        assert body.count(old)==1;body=body.replace(old,'')
        body=body.replace('mma128_off<k*32,k*2048,0,1>(dw0,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn),8192,1024,1)','mma128_off<k*32,k*2048,0,1>(dw0,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+WG*16384),8192,1024,1)')
        body=body.replace('named_bar_sync(2,128)','named_bar_sync(2+WG,128)').replace('p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];','p.part[ix+WG*128]=dw0[j];')
        if split_dw:
            body=body.replace('float dw0[64]={};','float dw0[32]={},dw1[32]={};').replace('fence_regs(dw0);','fence_regs(dw0);fence_regs(dw1);')
            marker='mma128_off<k*32,k*2048,0,1>(dw0,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+WG*16384),8192,1024,1),it>0||k>0);'
            assert body.count(marker)==1
            body=body.replace(marker,'mma64_off<k*32,k*2048,0,1>(dw0,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+WG*16384),8192,1024,1),it>0||k>0);\n   mma64_off<k*32,k*2048+8192,0,1>(dw1,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+WG*16384),8192,1024,1),it>0||k>0);')
            body=body.replace('static_for<64>([&](auto jj)','static_for<32>([&](auto jj)').replace('p.part[ix+WG*128]=dw0[j];','p.part[ix+WG*128]=dw0[j];p.part[ix+WG*128+64]=dw1[j];')
        if static_slots:
            start=body.index('template<int WG> TMN_DEVI void consume(')
            consumer_body=body[start:]
            marker=' for(int tile=begin,it=0;tile<end;++tile,++it){\n  int slot=it%2;'
            assert consumer_body.count(marker)==1
            consumer_body=consumer_body.replace(marker,' for(int base=0;base<end-begin;base+=2){static_for<2>([&](auto ss){constexpr int slot=decltype(ss)::value;int it=base+slot;')
            marker='\n }\n static_for<'+('32' if split_dw else '64')+'>'
            assert consumer_body.count(marker)==1
            consumer_body=consumer_body.replace(marker,'\n });}\n static_for<'+('32' if split_dw else '64')+'>')
            body=body[:start]+consumer_body
        if defer_output:
            assert static_slots
            start=body.index('template<int WG> TMN_DEVI void consume(');end=body.index('extern "C" __global__',start)
            fragment=body[start:end]
            fragment=fragment.replace('int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;','int tid=threadIdx.x%128;')
            marker=' int rank=blockIdx.x%32,split=blockIdx.x/32,tiles=p.M/64;\n int begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;'
            assert fragment.count(marker)==1;fragment=fragment.replace(marker,'')
            fragment=fragment.replace('base<end-begin','base<2304/WEIGHT_SPLITS')
            marker='\n static_for<'+('32' if split_dw else '64')+'>'
            assert fragment.count(marker)==1
            fragment=fragment.replace(marker,'''\n // The runtime shared allocation is less than 1MB, so this dependency is zero.
 // Keep output coordinate lifetimes after the final accumulator completion.
 int dep=zero_dep(__float_as_uint(dw0[0]));int etid=threadIdx.x%128+dep,block=blockIdx.x+dep;
 int lane=etid%32,warp=etid/32,rank=block%32,split=block/32;
''' +marker)
            body=body[:start]+fragment+body[end:]
        marker=' for(int c=0;c<4;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);'
        assert body.count(marker)==1;body=body.replace(marker,' tma_load_3d(sm+slot*INPUT,&p.xn,bar+slot,0,row,0);')
        body=body[:body.index('extern "C" __global__')]+'''
extern "C" __global__ __maxnreg__(CONSUMER_REGS)
void mw_d256_compact_four_source(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 if(threadIdx.x==0){for(int i=0;i<7;++i)mbar_init(bar+i,i>=5?3:1);fence_barrier_init();}__syncthreads();
 if(threadIdx.x<128){
  setmaxnreg_dec<24>();
  if(threadIdx.x==0){
   int rank=blockIdx.x%32,split=blockIdx.x/32,tiles=p.M/64;
   int begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
   mbar_arrive_expect_tx(bar+2,CH);tma_load_3d(sm+WEIGHT,&p.w,bar+2,0,rank*64,0);
   for(int tile=begin,it=0;tile<end;++tile,++it){
    int slot=it%2;if(it>=2)mbar_wait(bar+5+slot,((it/2)-1)&1);
    load_input(p,sm,bar,tile*64,rank,slot);
   }
  }
 }else if(threadIdx.x<256){setmaxnreg_dec<PRODUCER_REGS>();produce(p,sm,bar);}
 else{setmaxnreg_inc<CONSUMER_REGS>();if(threadIdx.x<384)consume<0>(p,sm,bar);else consume<1>(p,sm,bar);}
}
'''
        if combined_loader:
            start=body.index(' if(threadIdx.x<128){',body.index('extern "C" __global__'))
            body=body[:start]+''' if(threadIdx.x<128){setmaxnreg_dec<PRODUCER_REGS>();produce(p,sm,bar);}
 else{setmaxnreg_inc<CONSUMER_REGS>();if(threadIdx.x<256)consume<0>(p,sm,bar);else consume<1>(p,sm,bar);}
}
'''
            body=body.replace('i>=5?3:1','i>=5?2:1')
        if runtime_group:
            assert combined_loader
            body=body.replace('template<int WG> TMN_DEVI void consume(const Params& p,uint8_t* sm,uint64_t* bar){','TMN_DEVI void consume(const Params& p,uint8_t* sm,uint64_t* bar){\n const int WG=(threadIdx.x-128)/128;')
            body=body.replace('if(threadIdx.x<256)consume<0>(p,sm,bar);else consume<1>(p,sm,bar);','consume(p,sm,bar);')
        body=body.replace('tiles=p.M/64','tiles=147456/64');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}',f'-DPRODUCER_REGS={producer}',f'-DCONSUMER_REGS={consumer}']
        cubin=T.compile_text(body,flags)
        if combined_loader:
            from three_role_register_pool import initial_pool_three
            initial=((producer+consumer*2+23)//24)*8;self.threads=384
            self.cubin,self.pool_metadata=initial_pool_three(cubin,'mw_d256_compact_four_source',initial,producer,consumer)
        else:
            initial=((24+producer+consumer*2+31)//32)*8;self.threads=512
            self.cubin,self.pool_metadata=initial_pool_four(cubin,'mw_d256_compact_four_source',initial,producer,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_compact_four_source').kernel('mw_d256_compact_four_source');self.smem=115072;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();fields=plan.b7.params.fields
        tm=lambda t,rows:L.tensor_map(t,[64,64,4],dims=[64,rows,4],strides_bytes=[512,128],swizzle='128B',l2='256B')
        self.params=L.Struct([tm(p.xn,p.M),tm(p.w1,2048),*fields[2:9],*fields[-2:]])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(self.threads,1,1),[self.params],self.smem)
