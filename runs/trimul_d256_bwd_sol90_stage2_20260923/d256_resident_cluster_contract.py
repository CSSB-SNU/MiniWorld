"""Load each B column panel once per three-row cluster, then exchange it by DSM."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class ResidentClusterContract:
    def __init__(self,plan):
        p=plan.p;assert (p.D,p.n)==(256,384)
        root=Path(__file__).resolve().parent
        body=(root/'d256_resident_cluster_contract.cu').read_text().replace('// MMA_HELPER',(root/'mma_offset.cuh').read_text())
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc')]
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_resident_cluster_contract').kernel('mw_d256_resident_cluster_contract')
        self.smem=229504;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();ab=p.front.ab;d=256;h=512;n=384
        aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        tm=lambda t:L.tensor_map(t,[64,64],dims=[n,n*d],strides_bytes=[n*2],swizzle='128B',l2='256B')
        out=lambda t:L.tensor_map(t,[64,64],dims=[n,n*h],strides_bytes=[n*2],swizzle='128B',l2='128B')
        self.params=L.Struct([*[tm(t) for t in aa],*[tm(t) for t in bb],out(p.dl),out(p.dr)])
        self.drv=self.k.unit.drv;drv=self.drv;d=drv.d;fn=d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',d.cuFuncGetAttribute(getattr(d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
        self.args=L._Packed([self.params]);self.attr=d.CUlaunchAttribute();self.attr.id=d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        self.attr.value.clusterDim.x=3;self.attr.value.clusterDim.y=1;self.attr.value.clusterDim.z=1
        cfg=d.CUlaunchConfig();cfg.gridDimX=4*256*3;cfg.gridDimY=1;cfg.gridDimZ=1;cfg.blockDimX=256;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=self.smem;cfg.attrs=[self.attr];cfg.numAttrs=1
        self.cfg=cfg;self.active_clusters=int(drv._unwrap('cuOccupancyMaxActiveClusters',d.cuOccupancyMaxActiveClusters(fn,cfg)))

    def __call__(self):
        self.cfg.hStream=self.drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream))
        self.drv._unwrap('cuLaunchKernelEx',self.drv.d.cuLaunchKernelEx(self.cfg,self.drv.d.CUfunction(int(self.k.handle)),ctypes.addressof(self.args.array),0))
