"""Whole XN/weight transfers and N256 dW in the aliased two-group pipeline."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from two_role_register_pool import initial_pool_roles


class WholeAliasedSource:
    def __init__(self,plan,producer=96):
        p=plan.p;self.splits=plan.b7.splits;consumer=256-producer
        root=Path(__file__).resolve().parent
        body=(root/'d256_aliased_pipe_source.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        marker=' for(int c=0;c<4;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);'
        assert body.count(marker)==1;body=body.replace(marker,' tma_load_3d(sm+slot*INPUT,&p.xn,bar+slot,0,row,0);')
        marker='  for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);'
        assert body.count(marker)==1;body=body.replace(marker,'  tma_load_3d(sm+WEIGHT,&p.w,bar+2,0,rank*64,0);')
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
        body=body.replace('__launch_bounds__(256,2)',f'__maxnreg__({consumer})').replace('setmaxnreg_dec<PRODUCER_REGS>()',f'setmaxnreg_dec<{producer}>()').replace('setmaxnreg_inc<256-PRODUCER_REGS>()',f'setmaxnreg_inc<{consumer}>()')
        body=body.replace('mw_d256_aliased_pipe_source','mw_d256_whole_aliased_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        cubin=T.compile_text(body,flags)
        self.cubin,self.pool_metadata=initial_pool_roles(cubin,'mw_d256_whole_aliased_source',producer,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_whole_aliased_source').kernel('mw_d256_whole_aliased_source');self.smem=115072;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();fields=plan.b7.params.fields
        tm=lambda t,rows:L.tensor_map(t,[64,64,4],dims=[64,rows,4],strides_bytes=[512,128],swizzle='128B',l2='256B')
        self.params=L.Struct([tm(p.xn,p.M),tm(p.w1,2048),*fields[2:9],*fields[-2:]])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        assert self.occupancy==2 and self.registers==128

    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
