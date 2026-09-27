"""Full-column dW WGMMA with the original producer and register redistribution."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class RegisterLayoutSource:
    def __init__(self,plan,offset=True,patch=False):
        prior=plan.b7;self.__dict__.update(prior.__dict__)
        root=Path(__file__).resolve().parent
        body=mask_stage(prior.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        helper=(root/'mma256.cuh').read_text()
        if offset:
            helper=helper.replace('template<int TA,int TB>','template<int OA,int OB,int TA,int TB>').replace('void mma256(','void mma256_off(')
            helper=helper.replace('{.reg .pred p;setp.ne.b32 p,%130,0;', '{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%128,%133;add.u64 bx,%129,%134;setp.ne.b32 p,%130,0;')
            helper=helper.replace(',%128,%129,p,1,1,',',ax,bx,p,1,1,').replace('"n"(TB));','"n"(TB),"n"(OA>>4),"n"(OB>>4));')
        body=body.replace('TMN_DEVI void compute_source',helper+'\nTMN_DEVI void compute_source')
        body=body.replace('float dw0[64]={},dw1[64]={};','float dw[128]={};').replace('fence_regs(dw0);fence_regs(dw1);','fence_regs(dw);')
        old='mma128_off<k*32,k*2048,0,1>(dw0,a,smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);'
        assert body.count(old)==1
        new=('mma256_off<k*32,k*2048,0,1>(dw,a,smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);' if offset else
             'mma256<0,1>(dw,a+((k*32)>>4),smem_desc(smem_u32(xn),8192,1024,1)+((k*2048)>>4),it>0||k>0);')
        body=body.replace(old,new)
        old='mma128_off<k*32,k*2048,0,1>(dw1,a,smem_desc(smem_u32(xn+16384),8192,1024,1),it>0||k>0);'
        assert body.count(old)==1;body=body.replace(old,'')
        body=body.replace(' for(int j=0;j<64;++j){',' static_for<128>([&](auto jj){constexpr int j=decltype(jj)::value;')
        body=body.replace('  p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];\n }','  p.part[ix]=dw[j];\n });')
        body=body.replace('mw_d256_b7_tma','mw_d256_register_layout_source')
        body=body.replace('__launch_bounds__(256,2)','__launch_bounds__(256,1)')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        if patch:
            from initial_register_pool import initial_pool
            self.cubin,self.pool_metadata=initial_pool(self.cubin,'mw_d256_register_layout_source')
        self.k=T.load_unit(str(self.cubin),'mw_d256_register_layout_source').kernel('mw_d256_register_layout_source');self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        assert self.registers*256>=128*(32+224)
        if patch:assert self.registers==128 and self.occupancy==2 and self.local_bytes==0
    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
