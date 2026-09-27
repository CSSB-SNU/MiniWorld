"""Overlap direct-prefix gradient loads and protected TMA stores."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

class PipelinedGateEpi:
    def __init__(self,plan,slots=2):
        prior=plan.prefix_gate;self.__dict__.update(prior.__dict__)
        assert slots in (1,2)
        body=Path(__file__).with_name('prefix_gate_epi.cu').read_text()
        if slots==1:
            marker='  mbar_wait(bar,it&1);__syncthreads();'
            assert body.count(marker)==1
            body=body.replace(marker,'  mbar_wait(bar,it&1);if(tid==0 && it>0)tma_store_wait_read<0>();__syncthreads();')
            body=body.replace('tma_store_commit();tma_store_wait_read<0>();','tma_store_commit();')
        else:
            body=body.replace('BAR=49152','BAR=81920')
            body=body.replace('mbar_init(bar,1);','mbar_init(bar,1);mbar_init(bar+1,1);')
            a=body.index(' for(int tile=');b=body.index('  #pragma unroll',a)
            body=body[:a]+'''
 auto load=[&](int tile,int slot){
  int row=(tile/(D/64))*64,col=(tile%(D/64))*64;uint8_t* in=sm+slot*32768;
  mbar_arrive_expect_tx(bar+slot,32768);
  tma_load_2d(in,&p.proj,bar+slot,col,row);tma_load_2d(in+8192,&p.gate,bar+slot,col,row);
  tma_load_2d(in+16384,&p.dy,bar+slot,col,row);tma_load_2d(in+24576,&p.ds,bar+slot,col,row%p.N);
 };
 if(tid==0 && blockIdx.x<total)load(blockIdx.x,0);
 for(int tile=blockIdx.x,it=0;tile<total;tile+=gridDim.x,++it){
  int row=(tile/(D/64))*64,col=(tile%(D/64))*64,slot=it&1;uint8_t* in=sm+slot*32768;
  mbar_wait(bar+slot,(it/2)&1);
  if(tid==0){if(tile+gridDim.x<total)load(tile+gridDim.x,slot^1);if(it>0)tma_store_wait_read<0>();}
  __syncthreads();
'''+body[b:]
            body=body.replace('sm+32768+off','sm+65536+off').replace('sm+40960','sm+73728')
            body=body.replace('sm+32768)','sm+65536)')
            for offset in ('24576+off','16384+off','8192+off','off'):
                body=body.replace('sm+'+offset,'in+'+offset)
            body=body.replace('tma_store_commit();tma_store_wait_read<0>();','tma_store_commit();')
        body=body.replace('mw_prefix_gate_epi','mw_prefix_gate_pipeline')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={self.p.D}']
        if self.p.D!=512:flags.append('-DTMN_SIGMOID_TANH=1')
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_prefix_gate_pipeline').kernel('mw_prefix_gate_pipeline')
        self.smem=(49152 if slots==1 else 81920)+128;self.k.set_max_dynamic_smem(self.smem)
        drv=self.k.unit.drv
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),128,self.smem)))
        self.grid=torch.cuda.get_device_properties(self.p.x.device).multi_processor_count*self.occupancy
    def __call__(self):self.k.launch((self.grid,1,1),(128,1,1),[self.params],self.smem)
    def launch(self,*args,**kwargs):self()
