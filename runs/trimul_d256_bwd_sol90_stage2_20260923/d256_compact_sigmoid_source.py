"""Reuse retired incoming-gradient storage for GP and keep an exact LUT resident."""
import torch
from initial_register_pool import initial_pool
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


HELPER=r'''
TMN_DEVI float source_sigmoid(unsigned raw,const unsigned* table){
 unsigned mag=raw&32767u;
 #if LOOKUP_MODE == 1
 if(mag>=0x3880u && mag<0x4080u){
  unsigned word=table[mag-0x3880u];float pos=__uint_as_float(0x3f000000u|(word&0x7fffffu));
  if(raw&32768u)return __uint_as_float(__float_as_uint(__fsub_rn(1.f,pos))+(int(word)>>23));
  return pos;
 }
 #elif LOOKUP_MODE == 2
 if(mag>=0x3c80u && mag<0x4080u)return __uint_as_float(table[(raw>>15)*1024u+mag-0x3c80u]);
 #endif
 return math::sigmoid(__uint_as_float(raw<<16));
}
'''
BUILD=r'''
extern "C" __global__ void mw_build_packed_sigmoid(unsigned* table,unsigned* errors){
 unsigned i=blockIdx.x*256+threadIdx.x;
 #if LOOKUP_MODE == 1
 unsigned raw=0x3880u+i;float pos=math::sigmoid(__uint_as_float(raw<<16));
 float neg=math::sigmoid(__uint_as_float((raw|32768u)<<16));
 int delta=int(__float_as_uint(neg))-int(__float_as_uint(__fsub_rn(1.f,pos)));
 if(delta < -256 || delta>255 || (__float_as_uint(pos)&0xff800000u)!=0x3f000000u)atomicAdd(errors,1u);
 table[i]=(__float_as_uint(pos)&0x7fffffu)|(unsigned(delta)<<23);
 #else
 unsigned raw=((i/1024)<<15)+0x3c80u+(i%1024);
 table[i]=__float_as_uint(math::sigmoid(__uint_as_float(raw<<16)));
 #endif
}
extern "C" __global__ void mw_check_packed_sigmoid(const unsigned* table,unsigned* errors){
 unsigned raw=blockIdx.x*256+threadIdx.x;
 if(__float_as_uint(source_sigmoid(raw,table))!=__float_as_uint(math::sigmoid(__uint_as_float(raw<<16))))atomicAdd(errors,1u);
}
'''


class CompactSigmoidSource:
    def __init__(self,plan,mode=1,cap=208):
        prior=plan.pool_source;self.__dict__.update(prior.__dict__)
        body=prior.source_text
        self.table=torch.empty(2048,device=plan.p.x.device,dtype=torch.int32)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}',f'-DLOOKUP_MODE={mode}']
        builder=T.compile_text('#include "tmn_kernels.cuh"\nusing namespace tmn;using namespace tmn::sm90;\n'+HELPER+BUILD,flags)
        self.builder=T.load_unit(str(builder),'mw_build_packed_sigmoid')
        check=torch.zeros((),device=self.table.device,dtype=torch.int32)
        self.builder.kernel('mw_build_packed_sigmoid').launch((8,1,1),(256,1,1),[self.table,check],0)
        self.builder.kernel('mw_check_packed_sigmoid').launch((256,1,1),(256,1,1),[self.table,check],0)
        self.table_errors=int(check.item());assert self.table_errors==0,self.table_errors
        body=body.replace('INPUT=36864,WEIGHT=73728,DERIV=106496','INPUT=32768,WEIGHT=65536,DERIV=98304')
        body=body.replace('sm+114688','sm+106496').replace('sm+114816','sm+106624')
        marker=' tma_load_2d(sm+slot*INPUT+CH,rank<16?&p.dl:&p.dr,bar+slot,row,(rank%16)*32);'
        assert body.count(marker)==1;body=body.replace(marker,'')
        body=body.replace('smem_u32(s+32768)','smem_u32(sg)')
        marker='  float pre[32]={};';assert body.count(marker)==1
        body=body.replace(marker,'''  if(tid==0){mbar_arrive_expect_tx(bar+5,4096);tma_load_2d(sm+DERIV,rank<16?&p.dl:&p.dr,bar+5,row,(rank%16)*32);}
'''+marker)
        marker='  int ra=warp*16+lane/4;';assert body.count(marker)==1
        body=body.replace(marker,'  mbar_wait(bar+5,it&1);named_bar_sync(1,128);\n'+marker)
        body=body.replace('i<5;++i)mbar_init','i<6;++i)mbar_init')
        if mode:
            body=body.replace('float* part;int M;','float* part;const unsigned* lut;int M;')
            body=body.replace('TMN_DEVI void packed_glu(',HELPER+'\nTMN_DEVI void packed_glu(')
            body=body.replace('uint32_t ma,uint32_t mb){','uint32_t ma,uint32_t mb,const unsigned* table){')
            body=body.replace('packed_glu(pre,xn,sm+DERIV,ma,mb);','packed_glu(pre,xn,sm+DERIV,ma,mb,reinterpret_cast<unsigned*>(sm+106880));')
            marker='float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);'
            assert body.count(marker)==1
            body=body.replace(marker,'float ga=source_sigmoid(gr&65535u,table),gb=source_sigmoid(gr>>16,table),pa=bf16lo(pr),pb=bf16hi(pr);')
            marker=' if(threadIdx.x==0){for(int i=0;i<6;++i)mbar_init(bar+i,1);fence_barrier_init();}'
            assert body.count(marker)==1
            body=body.replace(marker,' for(int i=threadIdx.x;i<2048;i+=256)reinterpret_cast<unsigned*>(sm+106880)[i]=p.lut[i];\n'+marker)
            fields=prior.params.fields.copy();fields.insert(-1,self.table);self.params=T._launch_module().Struct(fields)
        body=body.replace('setmaxnreg_inc<208>()',f'setmaxnreg_inc<{cap}>()')
        body=body.replace('mw_d256_register_budget_source','mw_d256_compact_sigmoid_source');self.source_text=body
        cubin=T.compile_text(body,flags)
        self.cubin,self.pool_metadata=initial_pool(cubin,'mw_d256_compact_sigmoid_source',(32+cap)//2,cap)
        self.k=T.load_unit(str(self.cubin),'mw_d256_compact_sigmoid_source').kernel('mw_d256_compact_sigmoid_source');self.smem=115072 if mode else 106880;self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))

    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
