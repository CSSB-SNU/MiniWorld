"""Multicast credits originate at consumer retirement, prearming the next tile.

Unlike the earlier producer-to-producer handshake, every target barrier is armed
before its consumer announces that the input slot can be overwritten.
"""
import ctypes
import torch
from initial_register_pool import initial_pool
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class PrearmedMulticastSource:
    def __init__(self,plan,cluster=2):
        prior=plan.pool_source;self.__dict__.update(prior.__dict__)
        assert cluster in (0,2,4);self.cluster=cluster
        body=prior.source_text
        helper='''
TMN_DEVI void load_whole(void* dst,const CUtensorMap* map,uint64_t* bar,int row){
 asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{0,%3,0}],[%2];"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(row):"memory");
}
'''
        marker='TMN_DEVI void load_input(';assert body.count(marker)==1
        body=body.replace(marker,helper+marker)
        marker='for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);'
        assert body.count(marker)==1;body=body.replace(marker,'load_whole(sm+WEIGHT,&p.w,bar+2,rank*64);')
        marker='for(int c=0;c<4;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);'
        assert body.count(marker)==1
        body=body.replace(marker,'load_whole(sm+slot*INPUT,&p.xn,bar+slot,row);')
        if cluster:
            body=body.replace('#include "tmn_kernels.cuh"','#include "tmn_kernels.cuh"\n#include <cooperative_groups.h>')
            helper='''
TMN_DEVI uint32_t source_remote(const void* ptr){uint32_t v;asm volatile("mapa.shared::cluster.u32 %0,%1,0;":"=r"(v):"r"(smem_u32(ptr)));return v;}
TMN_DEVI void source_credit(uint64_t* bar){asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(source_remote(bar)):"memory");}
TMN_DEVI void source_wait(uint64_t* bar,int phase){uint32_t v;do{asm volatile("{.reg .pred P;mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P,[%1],%2;selp.u32 %0,1,0,P;}":"=r"(v):"r"(smem_u32(bar)),"r"(phase):"memory");}while(!v);}
TMN_DEVI void multicast_whole(void* dst,const CUtensorMap* map,uint64_t* bar,int row){
 asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{0,%3,0}],[%2],%4;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(row),"h"(uint16_t((1<<SOURCE_CLUSTER)-1)):"memory");
}
'''
            body=body.replace('TMN_DEVI void load_input(',helper+'TMN_DEVI void load_input(')
            marker=' mbar_arrive_expect_tx(bar+slot,INPUT+128);';assert body.count(marker)==1;body=body.replace(marker,'')
            marker='load_whole(sm+slot*INPUT,&p.xn,bar+slot,row);';assert body.count(marker)==1
            body=body.replace(marker,'''if(rank%SOURCE_CLUSTER==0){
  int begin=(p.M/64)*(blockIdx.x/32)/WEIGHT_SPLITS,it=row/64-begin;
  if(it>=2)source_wait(bar+5+slot,((it/2)-1)&1);
  multicast_whole(sm+slot*INPUT,&p.xn,bar+slot,row);
 }''')
            marker='if(tid==0)mbar_arrive(bar+3+slot);';assert body.count(marker)==1
            body=body.replace(marker,'''if(tid==0){
  if(tile+2<end){mbar_arrive_expect_tx(bar+slot,INPUT+128);source_credit(bar+5+slot);}
  mbar_arrive(bar+3+slot);
 }''')
            marker='for(int i=0;i<5;++i)mbar_init(bar+i,1);fence_barrier_init();';assert body.count(marker)==1
            body=body.replace(marker,'for(int i=0;i<7;++i)mbar_init(bar+i,i>=5?SOURCE_CLUSTER:1);fence_barrier_init();mbar_arrive_expect_tx(bar,INPUT+128);mbar_arrive_expect_tx(bar+1,INPUT+128);')
            marker=' __syncthreads();\n if(threadIdx.x<128)';assert body.count(marker)==1
            body=body.replace(marker,' __syncthreads();cooperative_groups::this_cluster().sync();\n if(threadIdx.x<128)')
            end=body.rfind('\n}');body=body[:end]+'\n cooperative_groups::this_cluster().sync();'+body[end:]
        body=body.replace('mw_d256_register_budget_source','mw_d256_prearmed_multicast_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}',f'-DSOURCE_CLUSTER={cluster}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool(cubin,'mw_d256_prearmed_multicast_source',120,208)
        self.k=T.load_unit(str(self.cubin),'mw_d256_prearmed_multicast_source').kernel('mw_d256_prearmed_multicast_source');self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();p=plan.p;fields=prior.params.fields.copy()
        fields[0]=L.tensor_map(p.xn,[64,64,4],dims=[64,p.M,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        fields[1]=L.tensor_map(p.w1,[64,64,4],dims=[64,2048,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        self.params=L.Struct(fields);self.drv=self.k.unit.drv;d=self.drv.d;fn=d.CUfunction(int(self.k.handle))
        query=lambda key:int(self.drv._unwrap('cuFuncGetAttribute',d.cuFuncGetAttribute(getattr(d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(self.drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        if cluster:
            self.args=L._Packed([self.params]);attr=d.CUlaunchAttribute();attr.id=d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attr.value.clusterDim.x=cluster;attr.value.clusterDim.y=1;attr.value.clusterDim.z=1
            cfg=d.CUlaunchConfig();cfg.gridDimX=32*self.splits;cfg.gridDimY=1;cfg.gridDimZ=1
            cfg.blockDimX=256;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=self.smem;cfg.attrs=[attr];cfg.numAttrs=1
            self.cfg=cfg;self.attr=attr
            self.active_clusters=int(self.drv._unwrap('cuOccupancyMaxActiveClusters',d.cuOccupancyMaxActiveClusters(fn,cfg)))
        else:self.active_clusters=0

    def __call__(self):
        if not self.cluster:return self.k.launch((32*self.splits,1,1),(256,1,1),[self.params],self.smem)
        self.cfg.hStream=self.drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream))
        self.drv._unwrap('cuLaunchKernelEx',self.drv.d.cuLaunchKernelEx(self.cfg,self.drv.d.CUfunction(int(self.k.handle)),ctypes.addressof(self.args.array),0))
