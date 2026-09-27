"""Share A across adjacent N tiles, allowing partial groups inside each cluster."""
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class GroupedMulticastGP:
    def __init__(self,plan,cluster=4):
        assert (plan.p.D,plan.p.n)==(512,384) and cluster in (0,2,4,8)
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__);self.cluster=cluster
        body=prior.source_text
        start=body.index(' int tiles=p.N/128;int half=');end=body.index('\n int mi=',start)
        body=body[:start]+''' int tiles=p.N/128;int half=blockIdx.x/(2*D*tiles*tiles),rem=blockIdx.x%(2*D*tiles*tiles),ch=rem/(2*tiles*tiles),mode=2*half+(rem/(tiles*tiles))%2,tile=rem%(tiles*tiles);'''+body[end:]
        if cluster:
            body=body.replace('#include "tmn_kernels.cuh"','#include "tmn_kernels.cuh"\n#include <cooperative_groups.h>')
            helper='''
TMN_DEVI int cluster_root(){int lane=blockIdx.x%GP_CLUSTER,ni=blockIdx.x%3;return lane>=ni?lane-ni:0;}
TMN_DEVI int cluster_width(){int root=cluster_root(),base=blockIdx.x-blockIdx.x%GP_CLUSTER;return min(GP_CLUSTER-root,3-(base+root)%3);}
TMN_DEVI uint32_t remote_ready(const void* ptr,int root){uint32_t v;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(v):"r"(smem_u32(ptr)),"r"(root));return v;}
TMN_DEVI void announce_ready(uint64_t* bar,int root){asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote_ready(bar,root)):"memory");}
TMN_DEVI void wait_ready(uint64_t* bar,int phase){uint32_t v;do{asm volatile("{.reg .pred P;mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P,[%1],%2;selp.u32 %0,1,0,P;}":"=r"(v):"r"(smem_u32(bar)),"r"(phase):"memory");}while(!v);}
TMN_DEVI void multicast4(void* dst,const CUtensorMap* map,uint64_t* bar,int c2,int c3,uint16_t mask){
 asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{0,0,%3,%4}],[%2],%5;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c2),"r"(c3),"h"(mask):"memory");
}
'''
            body=body.replace('template<int MODE> TMN_DEVI void load_input(',helper+'template<int MODE> TMN_DEVI void load_input(')
            marker=' load4(sm+slot*INPUT,p.a+MODE,bar+slot,(TA?mi:ki)/64,(ch*p.N+(TA?ki:mi))/64);'
            assert body.count(marker)==1
            body=body.replace(marker,''' int root=cluster_root();announce_ready(bar+7+slot,root);
 if(int(blockIdx.x%GP_CLUSTER)==root){
  wait_ready(bar+7+slot,(ki/64/SLOTS)&1);
  multicast4(sm+slot*INPUT,p.a+MODE,bar+slot,(TA?mi:ki)/64,(ch*p.N+(TA?ki:mi))/64,uint16_t(((1<<cluster_width())-1)<<root));
 }
''')
            marker='for(int i=0;i<7;++i)mbar_init(bar+i,(i>=3&&i<6)?2:1);fence_barrier_init();}__syncthreads();'
            assert body.count(marker)==1
            body=body.replace(marker,'for(int i=0;i<10;++i)mbar_init(bar+i,i>=7?cluster_width():(i>=3&&i<6)?2:1);fence_barrier_init();}__syncthreads();cooperative_groups::this_cluster().sync();')
            end=body.rfind('\n}');body=body[:end]+'\n cooperative_groups::this_cluster().sync();'+body[end:]
        body=body.replace('mw_wide_loader_warp_gp','mw_wide_grouped_multicast_gp');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=512',f'-DGP_CLUSTER={cluster}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_grouped_multicast_gp').kernel('mw_wide_grouped_multicast_gp');self.k.set_max_dynamic_smem(self.smem)
        self.drv=self.k.unit.drv;d=self.drv.d;fn=d.CUfunction(int(self.k.handle));self.grid=4*512*9
        query=lambda key:int(self.drv._unwrap('cuFuncGetAttribute',d.cuFuncGetAttribute(getattr(d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(self.drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,288,self.smem)))
        if cluster:
            self.args=T._launch_module()._Packed([self.params]);attr=d.CUlaunchAttribute();attr.id=d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attr.value.clusterDim.x=cluster;attr.value.clusterDim.y=1;attr.value.clusterDim.z=1
            cfg=d.CUlaunchConfig();cfg.gridDimX=self.grid;cfg.gridDimY=1;cfg.gridDimZ=1;cfg.blockDimX=288;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=self.smem;cfg.attrs=[attr];cfg.numAttrs=1
            self.cfg=cfg;self.attr=attr;self.active_clusters=int(self.drv._unwrap('cuOccupancyMaxActiveClusters',d.cuOccupancyMaxActiveClusters(fn,cfg)))
        else:self.active_clusters=0

    def __call__(self):
        if not self.cluster:return self.k.launch((self.grid,1,1),(288,1,1),[self.params],self.smem)
        self.cfg.hStream=self.drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream))
        self.drv._unwrap('cuLaunchKernelEx',self.drv.d.cuLaunchKernelEx(self.cfg,self.drv.d.CUfunction(int(self.k.handle)),ctypes.addressof(self.args.array),0))
