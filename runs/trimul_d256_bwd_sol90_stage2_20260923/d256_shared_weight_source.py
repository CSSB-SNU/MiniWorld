"""Independent row-split consumers share a resident projection-weight tile."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class SharedWeightSource:
    def __init__(self,original):
        self.__dict__.update(original.__dict__);root=Path(__file__).resolve().parent
        packed=(root/'packed_glu.cuh').read_text().replace('packed_glu','packed_cluster_glu').replace('smem_u32(s+32768)','smem_u32(s)')
        body=(root/'source_pairs.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text()).replace('// PACKED_HELPER',packed)
        body=body.replace('INPUT=40960,WEIGHT=81920,DERIV=147456,BARS=163840','INPUT=36864,WEIGHT=147456,DERIV=180224,BARS=196608')
        body=body.replace('const bf* mask;float* part;','const bf* mask;bf* global_gp[4];float* part;')
        begin=body.index('TMN_DEVI void producer(');end=body.index('template<int WG> TMN_DEVI void consumer(',begin)
        body=body[:begin]+'''
TMN_DEVI void producer(const Params& p,uint8_t* sm,uint64_t* b){
 int wg=threadIdx.x/32;
 if(wg>=2||threadIdx.x%32)return;
 int rank=blockIdx.x%32,split=(blockIdx.x/32)*2+wg;
 int begin=(p.M/64)*split/WEIGHT_SPLITS,end=(p.M/64)*(split+1)/WEIGHT_SPLITS;
 if(wg==0){
  mbar_arrive_expect_tx(b+4,32768);
  for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,b+4,c*64,rank*64);
 }
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2;uint8_t* input=sm+(wg*2+slot)*INPUT;uint64_t* ready=b+wg*2+slot;
  if(it>=2)mbar_wait(b+5+wg*2+slot,((it/2)-1)&1);
  mbar_arrive_expect_tx(ready,INPUT+128);
  for(int c=0;c<4;++c)tma_load_2d(input+c*8192,&p.xn,ready,c*64,tile*64);
  tma_load_2d(input+32768,rank<16?&p.dl:&p.dr,ready,tile*64,(rank%16)*32);
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],128,[%2];"::"r"(smem_u32(sm+BARS+128+(wg*2+slot)*128)),"l"(p.mask+tile*64),"r"(smem_u32(ready)):"memory");
 }
}
'''+body[end:]
        marker='rank=(blockIdx.x%16)*2+WG,split=blockIdx.x/16';assert body.count(marker)==1
        body=body.replace(marker,'rank=blockIdx.x%32,split=(blockIdx.x/32)*2+WG')
        body=body.replace('mbar_wait(b+2,0)','mbar_wait(b+4,0)')
        body=body.replace('uint8_t* xn=sm+slot*INPUT','uint8_t* xn=sm+(WG*2+slot)*INPUT')
        body=body.replace('mbar_wait(b+slot,(it/2)&1)','mbar_wait(b+WG*2+slot,(it/2)&1)')
        body=body.replace('sm+WEIGHT+WG*32768','sm+WEIGHT')
        body=body.replace('xn+32768+WG*4096','xn+32768')
        body=body.replace('p.mask[row+ra+8]','reinterpret_cast<bf*>(sm+BARS+128+(WG*2+slot)*128)[ra+8]')
        body=body.replace('p.mask[row+ra]','reinterpret_cast<bf*>(sm+BARS+128+(WG*2+slot)*128)[ra]')
        body=body.replace('mbar_arrive(b+3+slot)','mbar_arrive(b+5+WG*2+slot)')
        marker='for(int i=0;i<5;++i)mbar_init(b+i,i>=3?2:1);';assert body.count(marker)==1
        body=body.replace(marker,'for(int i=0;i<9;++i)mbar_init(b+i,1);')
        body=body.replace('mw_d256_b7_pairs','mw_d256_shared_weight_source')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_shared_weight_source').kernel('mw_d256_shared_weight_source')
        self.smem=196608+128+512;self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):self.k.launch((16*self.splits,1,1),(384,1,1),[self.params],self.smem)
