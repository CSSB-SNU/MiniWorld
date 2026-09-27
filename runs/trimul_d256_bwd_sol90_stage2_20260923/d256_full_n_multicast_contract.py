"""Pair full-N384 row CTAs and multicast their 48KB B operand per K step."""
import ctypes
import torch
from d256_one_group_producer_contract import OneGroupProducerContract
from two_role_modes_register_pool import initial_pool_modes
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class FullNMulticastContract(OneGroupProducerContract):
    def __init__(self,plan,consumer=224):
        super().__init__(plan,consumer);producer=256-consumer;body=self.source_text
        body=body.replace('#include "tmn_kernels.cuh"','#include "tmn_kernels.cuh"\n#include <cooperative_groups.h>')
        helper='''
TMN_DEVI uint32_t remote_ready(const void* ptr){uint32_t v;asm volatile("mapa.shared::cluster.u32 %0,%1,0;":"=r"(v):"r"(smem_u32(ptr)));return v;}
TMN_DEVI void arm_peer(uint64_t* ptr){asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote_ready(ptr)):"memory");}
TMN_DEVI void wait_peer(uint64_t* ptr,int phase){uint32_t v;do{asm volatile("{.reg .pred P;mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P,[%1],%2;selp.u32 %0,1,0,P;}":"=r"(v):"r"(smem_u32(ptr)),"r"(phase):"memory");}while(!v);}
TMN_DEVI void multicast_b(void* dst,const CUtensorMap* map,uint64_t* bar,int x,int y,int z){
 asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%3,%4,%5}],[%2],%6;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(x),"r"(y),"r"(z),"h"(uint16_t(3)):"memory");
}
'''
        body=body.replace('template<int MODE> TMN_DEVI void load(',helper+'template<int MODE> TMN_DEVI void load(')
        marker=' mbar_arrive_expect_tx(bar+slot,INPUT);';assert body.count(marker)==1;body=body.replace(marker,'')
        marker=' tma_load_3d(sm+slot*INPUT+ROW_GROUPS*8192,p.b+MODE,bar+slot,TB?0:ki,TB?ch*N+ki:0,TB?ni/64:(ch*N+ni)/64);';assert body.count(marker)==1
        body=body.replace(marker,''' if((blockIdx.x&1)==0){
  if(step>=SLOTS)wait_peer(bar+2*SLOTS+slot,((step/SLOTS)-1)&1);
  multicast_b(sm+slot*INPUT+ROW_GROUPS*8192,p.b+MODE,bar+slot,TB?0:ki,TB?ch*N+ki:0,TB?ni/64:(ch*N+ni)/64);
 }''')
        marker='if(tid==0)mbar_arrive(bar+SLOTS+slot);';assert body.count(marker)==1
        body=body.replace(marker,'''if(tid==0){
  if(step+SLOTS<6){mbar_arrive_expect_tx(bar+slot,INPUT);arm_peer(bar+2*SLOTS+slot);}
  mbar_arrive(bar+SLOTS+slot);
 }''')
        start=body.index(' constexpr int MT=N/(64*ROW_GROUPS),TILES=MT;')
        end=body.index(' if(threadIdx.x==0){for(int i=0;i<2*SLOTS;',start)
        body=body[:start]+''' constexpr int PAIRS=3;
 int pair=blockIdx.x/2,half=pair/(2*D*PAIRS),rem=pair%(2*D*PAIRS);
 int ch=rem/(2*PAIRS),mode=2*half+rem%2,tile=(rem/2)%PAIRS;
 int mi=(tile*2+(blockIdx.x&1))*64,ni=0;
'''+body[end:]
        marker='for(int i=0;i<2*SLOTS;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();';assert body.count(marker)==1
        body=body.replace(marker,'for(int i=0;i<3*SLOTS;++i)mbar_init(bar+i,i<2*SLOTS?1:2);fence_barrier_init();for(int i=0;i<SLOTS;++i)mbar_arrive_expect_tx(bar+i,INPUT); }__syncthreads();cooperative_groups::this_cluster().sync();')
        end=body.rfind('\n}');body=body[:end]+'\n cooperative_groups::this_cluster().sync();'+body[end:]
        body=body.replace('mw_d256_one_group_producer_contract','mw_d256_full_n_multicast_contract');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DROW_GROUPS=1','-DGRID_ORDER=1','-DMIN_BLOCKS=1','-DCONTRACT_SLOTS=2']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool_modes(cubin,'mw_d256_full_n_multicast_contract',producer,consumer)
        self.k=T.load_unit(str(self.cubin),'mw_d256_full_n_multicast_contract').kernel('mw_d256_full_n_multicast_contract');self.k.set_max_dynamic_smem(self.smem)
        self.drv=self.k.unit.drv;d=self.drv.d;fn=d.CUfunction(int(self.k.handle))
        query=lambda key:int(self.drv._unwrap('cuFuncGetAttribute',d.cuFuncGetAttribute(getattr(d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(self.drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        self.args=T._launch_module()._Packed([self.params]);attr=d.CUlaunchAttribute();attr.id=d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x=2;attr.value.clusterDim.y=1;attr.value.clusterDim.z=1
        cfg=d.CUlaunchConfig();cfg.gridDimX=self.grid;cfg.gridDimY=1;cfg.gridDimZ=1;cfg.blockDimX=256;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=self.smem;cfg.attrs=[attr];cfg.numAttrs=1
        self.cfg=cfg;self.attr=attr
        self.active_clusters=int(self.drv._unwrap('cuOccupancyMaxActiveClusters',d.cuOccupancyMaxActiveClusters(fn,cfg)))

    def __call__(self):
        self.cfg.hStream=self.drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream))
        self.drv._unwrap('cuLaunchKernelEx',self.drv.d.cuLaunchKernelEx(self.cfg,self.drv.d.CUfunction(int(self.k.handle)),ctypes.addressof(self.args.array),0))
