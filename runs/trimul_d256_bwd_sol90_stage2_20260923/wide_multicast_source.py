"""Explicit three-warpgroup producer/consumer source experiment."""
from pathlib import Path
import os,ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class MulticastSource:
    def __init__(self,p,original,*,use_offsets=None,prefetch=None,n128=None,cluster=2):
        self.p=p;self.original=original;self.cluster=cluster
        assert cluster in (2,4)
        self.use_offsets=os.environ.get('WIDE_PIPE_OFF')=='1' if use_offsets is None else use_offsets
        self.prefetch=os.environ.get('WIDE_PIPE_PREFETCH')=='1' if prefetch is None else prefetch
        self.n128=os.environ.get('WIDE_SOURCE_N128')=='1' if n128 is None else n128
        root=Path(__file__).resolve().parent;pre=root.parent/'trimul_d256_bwd_sol90_20260923'
        helpers=(root/'wide_source.cu').read_text().split('extern "C" __global__')[0]
        helpers=helpers.replace('BAR=DERIV+8192','BAR=DERIV+16384')
        helpers=helpers.replace('// MMA_HELPERS',(pre/'mma.cuh').read_text())
        helpers=helpers.replace('// PACKED_GLU',(root/'packed_glu.cuh').read_text().replace('s+32768','s+CH'))
        body=(root/'wide_pipe_source.cu').read_text().replace('// SOURCE_HELPERS',helpers)
        if self.prefetch:
            marker=' mbar_wait(bar+2,0);named_bar_sync(1,128);'
            assert body.count(marker)==1
            body=body.replace(marker,marker+'\n if(threadIdx.x==0)load_input(p,sm,bar,begin*64,rank,0);')
            marker='  if(it>=2)mbar_wait(bar+5+slot,((it/2)-1)&1);\n  if(tid==0)load_input(p,sm,bar,row,rank,slot);'
            assert body.count(marker)==1
            body=body.replace(marker,'')
            marker='  });wgmma_commit();wgmma_wait<0>();fence_regs(pre);'
            assert body.count(marker)==1
            body=body.replace(marker,'''  });wgmma_commit();
  if(tile+1<end){
   if(it>=1)mbar_wait(bar+5+(1-slot),((it-1)/2)&1);
   if(tid==0)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);
  }
  wgmma_wait<0>();fence_regs(pre);''')
        if self.use_offsets:
            from wide_offset_source import offsets
            body=offsets(body)
        if self.n128:
            from wide_source_mma128 import widen_dw
            body=widen_dw(body)
        from wide_multicast_transform import multicast
        body=multicast(body)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}','-DWEIGHT_SPLITS=32',f'-DWIDE_SOURCE_CLUSTER={cluster}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_multicast_source').kernel('mw_wide_multicast_source')
        self.smem=original.source_smem+8192;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();self.args=L._Packed([self.original.params]);self.drv=self.k.unit.drv;d=self.drv.d
        attr=d.CUlaunchAttribute();attr.id=d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x=cluster;attr.value.clusterDim.y=1;attr.value.clusterDim.z=1
        cfg=d.CUlaunchConfig();cfg.gridDimX=p.D//8*32;cfg.gridDimY=1;cfg.gridDimZ=1
        cfg.blockDimX=384;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=self.smem;cfg.attrs=[attr];cfg.numAttrs=1
        self.cfg=cfg;self.attr=attr
        self.active_clusters=int(self.drv._unwrap('cuOccupancyMaxActiveClusters',d.cuOccupancyMaxActiveClusters(d.CUfunction(int(self.k.handle)),cfg)))
        assert self.active_clusters>0
    def __call__(self):
        self.cfg.hStream=self.drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream))
        self.drv._unwrap('cuLaunchKernelEx',self.drv.d.cuLaunchKernelEx(self.cfg,self.drv.d.CUfunction(int(self.k.handle)),ctypes.addressof(self.args.array),0))
