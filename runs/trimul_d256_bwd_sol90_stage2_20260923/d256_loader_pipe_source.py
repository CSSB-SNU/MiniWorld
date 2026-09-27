"""A loader warp feeds separate projection/GLU and full-column dW groups."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from loader_warp_register_pool import initial_pool_loader_warp


class LoaderPipeSource:
    def __init__(self,plan,full_column=False):
        p=plan.p;self.p=p;self.splits=plan.b7.splits
        root=Path(__file__).resolve().parent
        body=(root/'d256_aliased_pipe_source.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        start=body.index(' if(tid==0){mbar_arrive_expect_tx(bar+2,CH);')
        end=body.index(' mbar_wait(bar+2,0);',start)
        body=body[:start]+body[end:]
        start=body.index('  if(tile+1<end){')
        end=body.index('  wgmma_wait<0>();fence_regs(pre);',start)
        body=body[:start]+body[end:]
        marker='tma_store_commit();mbar_arrive(bar+3+slot);tma_store_wait_all();'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'mbar_arrive(bar+5+slot);')
        marker='  wgmma_wait<0>();fence_regs(pre);'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'\n  uint32_t packed_pre[16];static_for<16>([&](auto jj){constexpr int j=decltype(jj)::value;packed_pre[j]=pack_bf16(pre[2*j],pre[2*j+1]);});')
        body=body.replace('pack_bf16(pre[q*8+j*2],pre[q*8+j*2+1])','packed_pre[q*4+j]').replace('pack_bf16(pre[(q+2)*8+j*2],pre[(q+2)*8+j*2+1])','packed_pre[(q+2)*4+j]')
        loader='''TMN_DEVI void load_source(const Params& p,uint8_t* sm,uint64_t* bar){
 if(threadIdx.x==256){
  int rank=blockIdx.x%32,split=blockIdx.x/32,tiles=p.M/64;
  int begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
  mbar_arrive_expect_tx(bar+2,CH);
  for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
  for(int tile=begin,it=0;tile<end;++tile,++it){
   int slot=it%2;
   if(it>=2)mbar_wait(bar+5+slot,((it/2)-1)&1);
   load_input(p,sm,bar,tile*64,rank,slot);
  }
 }
}
'''
        body=body.replace('TMN_DEVI void produce(',loader+'TMN_DEVI void produce(')
        if full_column:
            helper=(root/'mma256.cuh').read_text()
            helper=helper.replace('template<int TA,int TB>','template<int OA,int OB,int TA,int TB>').replace('void mma256(','void mma256_off(')
            helper=helper.replace('{.reg .pred p;setp.ne.b32 p,%130,0;', '{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%128,%133;add.u64 bx,%129,%134;setp.ne.b32 p,%130,0;')
            helper=helper.replace(',%128,%129,p,1,1,',',ax,bx,p,1,1,').replace('"n"(TB));','"n"(TB),"n"(OA>>4),"n"(OB>>4));')
            body=body.replace('TMN_DEVI void consume(',helper+'\nTMN_DEVI void consume(')
            body=body.replace('float dw0[64]={},dw1[64]={};','float dw[128]={};').replace('fence_regs(dw0);fence_regs(dw1);','fence_regs(dw);')
            old='mma128_off<k*32,k*2048,0,1>(dw0,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);'
            assert body.count(old)==1
            body=body.replace(old,'mma256_off<k*32,k*2048,0,1>(dw,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);')
            old='mma128_off<k*32,k*2048,0,1>(dw1,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+16384),8192,1024,1),it>0||k>0);'
            assert body.count(old)==1;body=body.replace(old,'')
            body=body.replace('static_for<64>([&](auto jj)','static_for<128>([&](auto jj)').replace('p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];','p.part[ix]=dw[j];')
        body=body.replace('__launch_bounds__(256,2)','__maxnreg__(152)').replace('mbar_init(bar+i,1)','mbar_init(bar+i,i>=5?2:1)')
        body=body.replace('setmaxnreg_dec<PRODUCER_REGS>()','setmaxnreg_dec<64>()')
        body=body.replace('else{setmaxnreg_inc<256-PRODUCER_REGS>();consume(p,sm,bar);}','else if(threadIdx.x<256){setmaxnreg_inc<152>();consume(p,sm,bar);}\n else{setmaxnreg_dec<24>();load_source(p,sm,bar);}')
        assert p.n==384
        body=body.replace('tiles=p.M/64','tiles=147456/64')
        body=body.replace('mw_d256_aliased_pipe_source','mw_d256_loader_pipe_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags)
        self.cubin,self.pool_metadata=initial_pool_loader_warp(cubin,'mw_d256_loader_pipe_source')
        self.k=T.load_unit(str(self.cubin),'mw_d256_loader_pipe_source').kernel('mw_d256_loader_pipe_source');self.smem=115072;self.k.set_max_dynamic_smem(self.smem)
        fields=plan.b7.params.fields;assert len(fields)==15,len(fields)
        self.params=T._launch_module().Struct([*fields[:9],*fields[-2:]])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))
        print('LOADER_PIPE_RESOURCE',dict(registers=self.registers,local_bytes=self.local_bytes,occupancy=self.occupancy,smem=self.smem,smem_zero_occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,0))),pool=self.pool_metadata),flush=True)
        assert self.occupancy==2 and self.registers==80,(self.occupancy,self.registers)

    def __call__(self):self.k.launch((32*self.splits,1,1),(384,1,1),[self.params],self.smem)
