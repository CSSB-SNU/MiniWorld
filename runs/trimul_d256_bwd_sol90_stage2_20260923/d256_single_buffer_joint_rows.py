"""Two compute groups share one aliased 128-row tile in two resident CTAs."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
class SingleBufferJointRows:
    def __init__(self,plan,packed=False):
        self.p=plan.p;self.splits=plan.b7.splits
        root=Path(__file__).resolve().parent
        body=(root/'d256_joint_rows_source.cu').read_text()
        body=body.replace('ONE=36864,INPUT=2*ONE,WEIGHT=2*INPUT,DERIV=WEIGHT+32768,BARS=DERIV+16384','ONE=40960,INPUT=2*ONE,WEIGHT=INPUT,BARS=WEIGHT+32768')
        begin=body.index('TMN_DEVI void produce(');end=body.index('template<int WG>',begin)
        body=body[:begin]+'''
TMN_DEVI void load_rows(const Params& p,uint8_t* sm,uint64_t* bar,int rank,int row){
 mbar_arrive_expect_tx(bar,73728+256);
 for(int wg=0;wg<2;++wg){uint8_t* dst=sm+wg*ONE;
  for(int c=0;c<4;++c)tma_load_2d(dst+c*8192,&p.xn,bar,c*64,row+wg*64);
  tma_load_2d(dst+32768,rank<16?&p.dl:&p.dr,bar,row+wg*64,(rank%16)*32);
 }
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],256,[%2];"::"r"(smem_u32(sm+BARS+128)),"l"(p.mask+row),"r"(smem_u32(bar)):"memory");
}
'''+body[end:]
        body=body.replace('int slot=it%2,row=tile*128+WG*64;uint8_t* xn=sm+slot*INPUT+WG*ONE;', 'int row=tile*128+WG*64;uint8_t* xn=sm+WG*ONE;\n  if(threadIdx.x==0)load_rows(p,sm,bar,rank,tile*128);')
        body=body.replace('mbar_wait(bar+slot,(it/2)&1)','mbar_wait(bar,it&1)')
        body=body.replace('sm+BARS+128+slot*256','sm+BARS+128')
        body=body.replace('sm+DERIV+WG*8192','sm+WG*ONE+32768').replace('sm+DERIV+src*8192','sm+src*ONE+32768')
        body=body.replace('sm+slot*INPUT+src*ONE+WG*16384','sm+src*ONE+WG*16384')
        body=body.replace('  if(threadIdx.x==128)mbar_arrive(bar+3+slot);','')
        begin=body.index('extern "C" __global__')
        body=body[:begin]+'''
extern "C" __global__ __launch_bounds__(256,2) void mw_d256_single_buffer_joint_rows(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BARS);
 if(threadIdx.x==0){
  for(int i=0;i<3;++i)mbar_init(bar+i,1);fence_barrier_init();
  mbar_arrive_expect_tx(bar+2,32768);
  for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,(blockIdx.x%32)*64);
 }
 __syncthreads();
 if(threadIdx.x<128)consume<0>(p,sm,bar);else consume<1>(p,sm,bar);
}
'''
        glu=(root/'packed_glu.cuh').read_text()
        if packed:
            glu=glu.replace('float (&a)[32]','uint32_t (&a)[16]')
            glu=glu.replace('pack_bf16(a[q*8+j*2],a[q*8+j*2+1])','a[q*4+j]')
            glu=glu.replace('pack_bf16(a[(q+2)*8+j*2],a[(q+2)*8+j*2+1])','a[(q+2)*4+j]')
            marker='  int ra=warp*16+lane/4;'
            body=body.replace(marker,'  uint32_t prepack[16];\n  static_for<16>([&](auto jj){constexpr int j=decltype(jj)::value;prepack[j]=pack_bf16(pre[2*j],pre[2*j+1]);});\n'+marker)
            body=body.replace('packed_glu(pre,xn,','packed_glu(prepack,xn,')
        body=body.replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text()).replace('// PACKED_HELPER',glu)
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_single_buffer_joint_rows').kernel('mw_d256_single_buffer_joint_rows');self.smem=115072;self.k.set_max_dynamic_smem(self.smem)
        f=plan.b7.params.fields;self.params=T._launch_module().Struct([*f[:9],*f[13:]])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
    def __call__(self):self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
