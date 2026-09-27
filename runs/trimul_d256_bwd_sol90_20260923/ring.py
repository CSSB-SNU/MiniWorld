from pathlib import Path
import ctypes,torch,os
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent
class RingB7:
 def __init__(self,p):
  self.p=p;self.cohorts=6
  self.ring=p.x.new_empty((6,8,4,512,64));self.flags=torch.zeros((6,8,33),device=p.x.device,dtype=torch.int32);self.mask=p.mask.bfloat16()
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-I'+str(R),'-DWEIGHT_SPLITS=24','-DWIDTH=256']
  ks={}
  for src,name in [('ring.cu','ring'),('ring_reduce.cu','reduce')]:
   body=(R/src).read_text()
   for h in ('ring_source.cuh','base.cuh','mma.cuh'):
    selected='ring_source_loop.cuh' if h=='ring_source.cuh' and os.environ.get('B7_LOOP')=='1' else h
    body=body.replace(f'#include "{h}"',(R/selected).read_text())
   out=T.compile_text(body,flags);k=T.load_unit(str(out),'mw_d256_b7_'+name).kernel('mw_d256_b7_'+name)
   k.set_max_dynamic_smem(114816 if name=='ring' else W.tuning(256)[2]);ks[name]=k
  self.k,self.reduce=ks['ring'],ks['reduce']
  drv=self.k.unit.drv
  occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),128,114816)))
  assert 240<=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
  L=T._launch_module();dy=lambda x:L.tensor_map(x,[64,32],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
  ringmap=L.tensor_map(self.ring,[64,64,1],dims=[64,512,6*8*4],strides_bytes=[128,65536],swizzle='128B',l2='128B')
  self.params=L.Struct([p.maps[0],W.tm(p.w1),dy(p.dl),dy(p.dr),self.mask,*p.gp,p.floats[7],p.M,ringmap,p.maps[5],p.maps[2],*p.maps[10:14],p.tensors[10],self.ring,self.flags,6])
  drv=self.reduce.unit.drv
  occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.reduce.handle)),256,W.tuning(256)[2])))
  self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),240,1,1,128,1,1,114816,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
  W.launch(self.reduce,self.p.params7,self.grid,D=256)
  return self.p.outputs
