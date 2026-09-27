"""Two fixed-contraction row CTAs share B via whole-map TMA multicast."""
import ctypes
import torch
from d256_whole_fixed_contract import WholeFixedContract
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class MulticastFixedContract(WholeFixedContract):
    def __init__(self, plan, slots=3):
        super().__init__(plan, 1, slots)
        body = self.source_text.replace('#include "tmn_kernels.cuh"', '#include "tmn_kernels.cuh"\n#include <cooperative_groups.h>')
        helpers = '''
TMN_DEVI uint32_t remote_addr(const void* ptr){uint32_t v;asm volatile("mapa.shared::cluster.u32 %0,%1,0;":"=r"(v):"r"(smem_u32(ptr)));return v;}
TMN_DEVI void armed(uint64_t* ptr){asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote_addr(ptr)):"memory");}
TMN_DEVI void wait_armed(uint64_t* ptr,int phase){uint32_t v;do{asm volatile("{.reg .pred P;mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P,[%1],%2;selp.u32 %0,1,0,P;}":"=r"(v):"r"(smem_u32(ptr)),"r"(phase):"memory");}while(!v);}
TMN_DEVI void multicast_b(void* dst,const CUtensorMap* map,uint64_t* bar,int x,int y,int z){
 asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%3,%4,%5}],[%2],%6;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(x),"r"(y),"r"(z),"h"(uint16_t(3)):"memory");
}
'''
        marker = 'template<int MODE> TMN_DEVI void load('
        assert body.count(marker) == 1
        body = body.replace(marker, helpers + marker)
        old = ' tma_load_3d(sm+slot*INPUT+ROW_GROUPS*8192,p.b+MODE,bar+slot,TB?0:ki,TB?ch*N+ki:0,TB?ni/64:(ch*N+ni)/64);'
        assert body.count(old) == 1
        body = body.replace(old, '''
 armed(bar+SLOTS+slot);
 if((blockIdx.x&1)==0){
  wait_armed(bar+SLOTS+slot,(step/SLOTS)&1);
  multicast_b(sm+slot*INPUT+ROW_GROUPS*8192,p.b+MODE,bar+slot,TB?0:ki,TB?ch*N+ki:0,TB?ni/64:(ch*N+ni)/64);
 }''')
        start = body.index(' constexpr int MT=N/(64*ROW_GROUPS),TILES=MT*3;')
        end = body.index(' if(threadIdx.x==0){for(int i=0;i<SLOTS;', start)
        body = body[:start] + ''' constexpr int PAIRS=9;
 int pair=blockIdx.x/2,half=pair/(2*D*PAIRS),rem=pair%(2*D*PAIRS);
 int ch=rem/(2*PAIRS),mode=2*half+rem%2,tile=(rem/2)%PAIRS;
 int mi=((tile/3)*2+(blockIdx.x&1))*64,ni=(tile%3)*128;
''' + body[end:]
        old = 'for(int i=0;i<SLOTS;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();'
        assert body.count(old) == 1
        body = body.replace(old, 'for(int i=0;i<2*SLOTS;++i)mbar_init(bar+i,i<SLOTS?1:2);fence_barrier_init();}__syncthreads();cooperative_groups::this_cluster().sync();')
        end = body.rfind('\n}')
        body = body[:end] + '\n cooperative_groups::this_cluster().sync();' + body[end:]
        body = body.replace('mw_d256_whole_fixed_contract', 'mw_d256_multicast_fixed_contract')
        self.source_text = body
        flags = ['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DROW_GROUPS=1','-DGRID_ORDER=1',f'-DMIN_BLOCKS={4 if slots==2 else 3}',f'-DCONTRACT_SLOTS={slots}']
        self.cubin = T.compile_text(body, flags)
        self.k = T.load_unit(str(self.cubin), 'mw_d256_multicast_fixed_contract').kernel('mw_d256_multicast_fixed_contract')
        self.k.set_max_dynamic_smem(self.smem)
        drv = self.k.unit.drv
        d = drv.d
        self.drv = drv
        fn = d.CUfunction(int(self.k.handle))
        query = lambda n: int(drv._unwrap('cuFuncGetAttribute', d.cuFuncGetAttribute(getattr(d.CUfunction_attribute,n),fn)))
        self.registers = query('CU_FUNC_ATTRIBUTE_NUM_REGS')
        self.local_bytes = query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy = int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        self.args = T._launch_module()._Packed([self.params])
        attr = d.CUlaunchAttribute()
        attr.id = d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x = 2
        attr.value.clusterDim.y = attr.value.clusterDim.z = 1
        cfg = d.CUlaunchConfig()
        cfg.gridDimX = self.grid
        cfg.gridDimY = cfg.gridDimZ = 1
        cfg.blockDimX = self.threads
        cfg.blockDimY = cfg.blockDimZ = 1
        cfg.sharedMemBytes = self.smem
        cfg.attrs = [attr]
        cfg.numAttrs = 1
        self.cfg, self.attr = cfg, attr
        self.active_clusters = int(drv._unwrap('cuOccupancyMaxActiveClusters',d.cuOccupancyMaxActiveClusters(fn,cfg)))
        assert self.active_clusters > 0

    def __call__(self):
        self.cfg.hStream = self.drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream))
        self.drv._unwrap('cuLaunchKernelEx',self.drv.d.cuLaunchKernelEx(self.cfg,self.drv.d.CUfunction(int(self.k.handle)),ctypes.addressof(self.args.array),0))
