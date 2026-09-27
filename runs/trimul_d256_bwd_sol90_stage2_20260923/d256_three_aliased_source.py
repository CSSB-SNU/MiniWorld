"""One packed-GP group plus two half-column dW groups in two-CTA shared storage."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class ThreeAliasedSource:
    def __init__(self,plan,producer_regs=64,tile_n=128):
        p=plan.p;self.p=p;self.splits=plan.b7.splits
        assert producer_regs in (64,80)
        consumer_regs=(240-producer_regs)//2
        root=Path(__file__).resolve().parent
        body=(root/'d256_aliased_pipe_source.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        marker='  wgmma_wait<0>();fence_regs(pre);'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'''
  uint32_t prepack[16];
  #pragma unroll
  for(int j=0;j<16;++j)prepack[j]=pack_bf16(pre[j*2],pre[j*2+1]);
''')
        body=body.replace('pack_bf16(pre[q*8+j*2],pre[q*8+j*2+1])','prepack[q*4+j]')
        body=body.replace('pack_bf16(pre[(q+2)*8+j*2],pre[(q+2)*8+j*2+1])','prepack[(q+2)*4+j]')
        body=body.replace('TMN_DEVI void consume(','template<int WG> TMN_DEVI void consume(')
        body=body.replace('float dw0[64]={},dw1[64]={};','float dw[64]={};')
        body=body.replace('fence_regs(dw0);fence_regs(dw1);','fence_regs(dw);')
        body=body.replace('named_bar_sync(2,128);','named_bar_sync(2+WG,128);')
        a='mma128_off<k*32,k*2048,0,1>(dw0,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);'
        assert body.count(a)==1
        body=body.replace(a,'mma128_off<k*32,k*2048,0,1>(dw,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+WG*16384),8192,1024,1),it>0||k>0);')
        a='mma128_off<k*32,k*2048,0,1>(dw1,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+16384),8192,1024,1),it>0||k>0);'
        assert body.count(a)==1;body=body.replace(a,'')
        body=body.replace('p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];','p.part[ix+WG*128]=dw[j];')
        assert tile_n in (64,128)
        if tile_n==64:
            body=body.replace('float dw[64]={};','float dw0[32]={},dw1[32]={};')
            body=body.replace('fence_regs(dw);','fence_regs(dw0);fence_regs(dw1);')
            a='mma128_off<k*32,k*2048,0,1>(dw,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+WG*16384),8192,1024,1),it>0||k>0);'
            assert body.count(a)==1
            b='mma64_off<k*32,k*2048,0,1>(dw0,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+WG*16384),8192,1024,1),it>0||k>0);'
            b+='mma64_off<k*32,k*2048,0,1>(dw1,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+WG*16384+8192),8192,1024,1),it>0||k>0);'
            body=body.replace(a,b).replace('static_for<64>([&](auto jj)','static_for<32>([&](auto jj)')
            body=body.replace('p.part[ix+WG*128]=dw[j];','p.part[ix+WG*128]=dw0[j];p.part[ix+WG*128+64]=dw1[j];')
        body=body.replace('__launch_bounds__(256,2)','__launch_bounds__(384,2)')
        body=body.replace('mbar_init(bar+i,1)','mbar_init(bar+i,i>=5?2:1)')
        body=body.replace('setmaxnreg_dec<PRODUCER_REGS>();',f'setmaxnreg_dec<{producer_regs}>();')
        body=body.replace('setmaxnreg_inc<256-PRODUCER_REGS>();consume(p,sm,bar);',
                          f'setmaxnreg_inc<{consumer_regs}>();if(threadIdx.x<256)consume<0>(p,sm,bar);else consume<1>(p,sm,bar);')
        body=body.replace('mw_d256_aliased_pipe_source','mw_d256_three_aliased_source')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_three_aliased_source').kernel('mw_d256_three_aliased_source');self.smem=115072;self.k.set_max_dynamic_smem(self.smem)
        fields=plan.b7.params.fields
        self.params=T._launch_module().Struct([*fields[:9],*fields[13:]])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))
    def __call__(self):self.k.launch((32*self.splits,1,1),(384,1,1),[self.params],self.smem)
