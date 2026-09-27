"""Multicast normalized inputs between channel CTAs of the D256 source."""
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage

class D256MulticastSource:
    def __init__(self,original,cluster=2):
        self.__dict__.update(original.__dict__);assert cluster in (2,4)
        body=mask_stage(original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        body=body.replace('#include "tmn_kernels.cuh"','#include "tmn_kernels.cuh"\n#include <cooperative_groups.h>')
        begin=body.index('TMN_DEVI void load_input(');end=body.index('\n}',begin)+2
        body=body[:begin]+'''
TMN_DEVI uint32_t d256_remote(const void* ptr,int rank){uint32_t v;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(v):"r"(smem_u32(ptr)),"r"(rank));return v;}
TMN_DEVI void d256_signal(uint64_t* bar){asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(d256_remote(bar,0)):"memory");}
TMN_DEVI void d256_cluster_wait(uint64_t* bar,int phase){uint32_t v;do{asm volatile("{.reg .pred P;mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P,[%1],%2;selp.u32 %0,1,0,P;}":"=r"(v):"r"(smem_u32(bar)),"r"(phase):"memory");}while(!v);}
TMN_DEVI void d256_multicast(void* dst,const CUtensorMap* map,uint64_t* bar,int c,int row){
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%3,%4}],[%2],%5;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c),"r"(row),"h"(uint16_t((1<<SOURCE_CLUSTER)-1)):"memory");
}
TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int row,int rank,int slot){
 int begin=(p.M/64)*(blockIdx.x/32)/WEIGHT_SPLITS,it=row/64-begin;
 mbar_arrive_expect_tx(bar+slot,INPUT+128);
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],128,[%2];"::"r"(smem_u32(sm+114816+slot*128)),"l"(p.mask+row),"r"(smem_u32(bar+slot)):"memory");
 tma_load_2d(sm+slot*INPUT+CH,rank<16?&p.dl:&p.dr,bar+slot,row,(rank%16)*32);
 // The leader may multicast only after every target armed its local barrier.
 d256_signal(bar+5+slot);
 if(rank%SOURCE_CLUSTER==0){
  d256_cluster_wait(bar+5+slot,(it/2)&1);
  for(int c=0;c<4;++c)d256_multicast(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);
 }
}
'''+body[end:]
        marker='for(int i=0;i<5;++i)mbar_init(bar+i,1);';assert body.count(marker)==1
        body=body.replace(marker,'for(int i=0;i<7;++i)mbar_init(bar+i,i>=5?SOURCE_CLUSTER:1);')
        marker=' __syncthreads();\n if(threadIdx.x<128)';assert body.count(marker)==1
        body=body.replace(marker,' __syncthreads();cooperative_groups::this_cluster().sync();\n if(threadIdx.x<128)')
        end=body.rfind('\n}');body=body[:end]+'\n cooperative_groups::this_cluster().sync();'+body[end:]
        body=body.replace('mw_d256_b7_tma','mw_d256_multicast_source')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}',f'-DSOURCE_CLUSTER={cluster}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_multicast_source').kernel('mw_d256_multicast_source');self.k.set_max_dynamic_smem(self.smem)
        self.drv=self.k.unit.drv;d=self.drv.d;L=T._launch_module();self.args=L._Packed([self.params])
        attr=d.CUlaunchAttribute();attr.id=d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x=cluster;attr.value.clusterDim.y=1;attr.value.clusterDim.z=1
        cfg=d.CUlaunchConfig();cfg.gridDimX=32*self.splits;cfg.gridDimY=1;cfg.gridDimZ=1
        cfg.blockDimX=self.source_threads;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=self.smem;cfg.attrs=[attr];cfg.numAttrs=1
        self.cfg=cfg;self.attr=attr
        self.active_clusters=int(self.drv._unwrap('cuOccupancyMaxActiveClusters',d.cuOccupancyMaxActiveClusters(d.CUfunction(int(self.k.handle)),cfg)))
        assert self.active_clusters>0
    def __call__(self):
        self.cfg.hStream=self.drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream))
        self.drv._unwrap('cuLaunchKernelEx',self.drv.d.cuLaunchKernelEx(self.cfg,self.drv.d.CUfunction(int(self.k.handle)),ctypes.addressof(self.args.array),0))
