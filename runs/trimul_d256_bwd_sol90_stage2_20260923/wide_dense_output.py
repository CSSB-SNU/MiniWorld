"""Explicit output schedule: exact forward LN, dense GEMMs, native epilogue."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class DenseOutput:
    def __init__(self,f,p,products,threads=128,packed_unroll=1):
        self.p=p;self.f=f;self.proj,self.gate=products;self.y=f.output.y
        self.wp,self.wg=f.leaves[6],f.leaves[5];d=p.D;h=2*d
        self.threads=threads;self.smem=128*h+128
        body=(F.R/'output.cu').read_text()
        if packed_unroll!=1:
            assert packed_unroll in (4,8,16)
            body=body.replace('#pragma unroll 1',f'#pragma unroll {packed_unroll}')
        a=body.index('struct Params ');b=body.index('\nTMN_DEVI float rd',a)
        body=body[:a]+'struct Params {CUtensorMap tri,norm;const float *gamma,*beta;float *mu,*rs;int M;};'+body[b:]
        a=body.index('template<bool GATE>');b=body.index('extern "C" __global__',a)
        body=body[:a]+body[b:]
        body=body.replace('mw_wide_output(', 'mw_wide_dense_norm(')
        body=body.replace('uint8_t* sx=sm;uint8_t* buf=sm+XBYTES;\n float* stats=reinterpret_cast<float*>(buf+2*STAGE);uint64_t* bar=reinterpret_cast<uint64_t*>(stats+128);','uint8_t* sx=sm;uint64_t* bar=reinterpret_cast<uint64_t*>(sm+XBYTES);')
        body=body.replace('float rs=rsqrtf(sumwarp(s)/H+1e-5f);','float rs=rsqrtf(sumwarp(s)/H+1e-5f);if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}')
        a=body.index('  for(int col=0;')
        body=body[:a]+'''
  if(tid==0){for(int c=0;c<H;c+=64){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.norm),"r"(smem_u32(sx+c*128)),"r"(c),"r"(row):"memory");
  }tma_store_commit();tma_store_wait_all();}
  __syncthreads();
 }
}
struct Epi {const bf *proj,*gate,*x,*ds;bf *y;int elements,period;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_dense_output_epi(__grid_constant__ const Epi p){
 for(int i=blockIdx.x*256+threadIdx.x;i<p.elements/2;i+=gridDim.x*256){
  uint32_t pr=reinterpret_cast<const uint32_t*>(p.proj)[i],ga=reinterpret_cast<const uint32_t*>(p.gate)[i];
  uint32_t xx=reinterpret_cast<const uint32_t*>(p.x)[i],ds=reinterpret_cast<const uint32_t*>(p.ds)[i%(p.period/2)];
  float a=bf16lo(pr)*math::sigmoid(bf16lo(ga)),b=bf16hi(pr)*math::sigmoid(bf16hi(ga));
  reinterpret_cast<uint32_t*>(p.y)[i]=pack_bf16(fmaf(a,bf16lo(ds),bf16lo(xx)),fmaf(b,bf16hi(ds),bf16hi(xx)));
 }
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0','-DTMN_SIGMOID_TANH=1',
               f'-DWIDTH={d}',f'-DGROUPS={threads//128}','-DKCHUNK=1']
        self.cubin=T.compile_text(body,flags)
        unit=T.load_unit(str(self.cubin),'mw_wide_dense_norm')
        self.norm=unit.kernel('mw_wide_dense_norm');self.norm.set_max_dynamic_smem(self.smem)
        self.epi=unit.kernel('mw_wide_dense_output_epi')
        drv=self.norm.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.norm.handle)),threads,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module()
        self.np=L.Struct([F.tm(p.tri,[64,64],[p.M,h],[p.M*2]),p.maps[3],f.leaves[9],f.leaves[10],p.floats[5],p.floats[6],p.M])
        self.ep=L.Struct([self.proj,self.gate,p.x,p.ds,self.y,p.M*d,p.n*d])

    def __call__(self):
        p=self.p
        self.norm.launch((self.grid,1,1),(self.threads,1,1),[self.np],self.smem)
        torch.mm(p.tensors[6],self.wp.t(),out=self.proj)
        torch.mm(p.xn.reshape(p.M,p.D),self.wg.t(),out=self.gate)
        self.epi.launch((1056,1,1),(256,1,1),[self.ep],0)
        return self.y
