from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class ClusterReduceDnLN:
    def __init__(self,plan,emit_dn=False):
        p=plan.p;d=p.D;h=2*d;self.p=p;self.cluster=4 if d==256 else 8
        root=Path(__file__).resolve().parent
        body=(root/'cluster_reduce_dn_ln.cu').read_text().replace('// MMA_HELPERS',(root.parent/'trimul_d256_bwd_sol90_20260923/mma.cuh').read_text())
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DEMIT_DN={int(emit_dn)}']
        self.cubin=T.compile_text(body,flags);unit=T.load_unit(str(self.cubin),'mw_cluster_reduce_dn_ln')
        self.k=unit.kernel('mw_cluster_reduce_dn_ln');self.finish=unit.kernel('mw_cluster_reduce_dn_ln_finish');self.smem=90112+640;self.k.set_max_dynamic_smem(self.smem)
        self.partial=torch.empty((p.M//64,2,h),device=p.x.device,dtype=torch.float32)
        L=T._launch_module()
        tm=lambda t,fast,slow:L.tensor_map(t,[64,64],dims=[fast,slow],strides_bytes=[fast*2],swizzle='128B',l2='128B')
        self.params=L.Struct([tm(p.tensors[7],d,p.M),tm(plan.b1.wp,h,d),tm(p.tri,p.M,h),tm(p.dt,p.M,h),tm(p.tensors[9],h,p.M),p.floats[5],p.floats[6],p.floats[2],self.partial,p.floats[10],p.floats[11],p.M])
        self.args=L._Packed([self.params]);self.drv=unit.drv;driver=self.drv.d
        attr=driver.CUlaunchAttribute();attr.id=driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x=self.cluster;attr.value.clusterDim.y=1;attr.value.clusterDim.z=1
        cfg=driver.CUlaunchConfig();cfg.gridDimX=p.M//64*self.cluster;cfg.gridDimY=1;cfg.gridDimZ=1;cfg.blockDimX=128;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=self.smem;cfg.attrs=[attr];cfg.numAttrs=1
        self.cfg=cfg;self.attr=attr
        function=driver.CUfunction(int(self.k.handle))
        self.active_clusters=int(self.drv._unwrap('cuOccupancyMaxActiveClusters',driver.cuOccupancyMaxActiveClusters(function,cfg)))
        query=lambda n:int(self.drv._unwrap('cuFuncGetAttribute',driver.cuFuncGetAttribute(getattr(driver.CUfunction_attribute,n),function)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):
        self.cfg.hStream=self.drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream))
        self.drv._unwrap('cuLaunchKernelEx',self.drv.d.cuLaunchKernelEx(self.cfg,self.drv.d.CUfunction(int(self.k.handle)),ctypes.addressof(self.args.array),0))
        self.finish.launch((self.p.D//16*8,1,1),(256,1,1),[self.params],0)
