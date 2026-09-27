"""Check exact compressed sigmoid correction and complete workload timing."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_pool_checkpoint import Training
from d256_warp_mma_pipe_source import WarpMMAPipeSource
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
import ctypes
source='#include <cooperative_groups.h>\nextern "C" __global__ void check_cluster(int* out){cooperative_groups::this_cluster().sync();out[blockIdx.x]=blockIdx.x;}'
with torch.no_grad(),T.native_context(torch.device('cuda:0')):
 cubin=T.compile_text(source,['-std=c++17','-O3','-arch=sm_90a','--cubin'])
 k=T.load_unit(str(cubin),'check_cluster').kernel('check_cluster');drv=k.unit.drv;d=drv.d
 for size in (2,3,4):
  out=torch.full((size*132,),-1,device='cuda',dtype=torch.int32);args=T._launch_module()._Packed([out])
  attr=d.CUlaunchAttribute();attr.id=d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;attr.value.clusterDim.x=size;attr.value.clusterDim.y=1;attr.value.clusterDim.z=1
  cfg=d.CUlaunchConfig();cfg.gridDimX=size*132;cfg.gridDimY=1;cfg.gridDimZ=1;cfg.blockDimX=256;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=0;cfg.attrs=[attr];cfg.numAttrs=1;cfg.hStream=d.CUstream(int(torch.cuda.current_stream().cuda_stream))
  try:
   occ=drv._unwrap('cuOccupancyMaxActiveClusters',d.cuOccupancyMaxActiveClusters(d.CUfunction(int(k.handle)),cfg))
   drv._unwrap('cuLaunchKernelEx',d.cuLaunchKernelEx(cfg,d.CUfunction(int(k.handle)),ctypes.addressof(args.array),0));torch.cuda.synchronize()
   print('CLUSTER',size,occ,bool(torch.equal(out,torch.arange(out.numel(),device='cuda',dtype=torch.int32))),flush=True)
  except Exception as e:print('CLUSTER',size,str(e),flush=True)
