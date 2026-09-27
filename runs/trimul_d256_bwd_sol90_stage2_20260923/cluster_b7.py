"""Experimental two-sided B7 DSM path; no engine/selected dispatch change."""
from pathlib import Path
import ctypes,os,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent;PRE=R.parent/'trimul_d256_bwd_sol90_20260923'
class ClusterB7:
 def __init__(self,p,mask):
  self.p=p;self.splits=int(os.environ.get('CLUSTER_SPLITS','16'));assert 1<=self.splits<=32
  self.partial=torch.empty((p.M,256),dtype=torch.float32,device=p.x.device)
  rs='template <int OFF_BYTES>'+(R/'dn_ln.cu').read_text().split('template <int OFF_BYTES>')[1].split('TMN_DEVI void dnorm')[0]
  helpers=(R/'mma_offset.cuh').read_text()+rs
  packed=(R/'packed_glu.cuh').read_text().replace('packed_glu','packed_cluster_glu').replace('smem_u32(s+32768)','smem_u32(s)')
  self.plane=os.environ.get('CLUSTER_PLANE')=='1'
  body=(R/('cluster_plane.cu' if self.plane else 'cluster_side.cu')).read_text().replace('// MMA_HELPERS',helpers).replace('// PACKED_HELPER',packed)
  self.dump=int(os.environ.get('CLUSTER_DUMP','0'))
  direct=int(os.environ.get('CLUSTER_DIRECT_ARRIVE','1'))
  regs=int(os.environ.get('CLUSTER_DX_REGS','64'));assert regs in (64,80,96,112,128)
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}',f'-DCLUSTER_DUMP={self.dump}',f'-DCLUSTER_DIRECT_ARRIVE={direct}']
  flags += [f'-DCLUSTER_DX_REGS={regs}']
  flags += [f'-DCLUSTER_SYNC_PIPE={int(os.environ.get("CLUSTER_SYNC_PIPE","0"))}']
  flags += [f'-DCLUSTER_PREFETCH={int(os.environ.get("CLUSTER_PREFETCH","0"))}']
  bulk=int(os.environ.get('CLUSTER_BULK','0'));assert not bulk or os.environ.get('CLUSTER_SYNC_PIPE')=='1'
  flags += [f'-DCLUSTER_BULK={bulk}']
  flags += [f'-DCLUSTER_CLOCK={int(os.environ.get("CLUSTER_CLOCK","0"))}']
  self.cubin=T.compile_text(body,flags);print('CLUSTER_CUBIN',self.cubin,flush=True)
  unit=T.load_unit(str(self.cubin),'mw_d256_cluster_left');self.k=[unit.kernel('mw_d256_cluster_'+s) for s in ('left','right')]
  self.smem=221312 if self.plane else 213120
  for k in self.k:k.set_max_dynamic_smem(self.smem)
  L=T._launch_module();dy=lambda x:L.tensor_map(x,[64,32],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
  self.clock=torch.zeros((self.splits*8,4,128 if self.plane else 8),dtype=torch.int64,device=p.x.device)
  self.params=L.Struct([p.maps[0],W.tm(p.w1),dy(p.dl),dy(p.dr),p.maps[5],p.maps[2],*p.maps[10:14],mask,self.partial,p.floats[7],p.tensors[10],*p.gp,self.clock,*([p.dg] if self.plane else []),p.M])
  self.args=L._Packed([self.params]);self.drv=self.k[0].unit.drv;d=self.drv.d
  attr=d.CUlaunchAttribute();attr.id=d.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
  attr.value.clusterDim.x=8;attr.value.clusterDim.y=1;attr.value.clusterDim.z=1
  cfg=d.CUlaunchConfig();cfg.gridDimX=self.splits*8;cfg.gridDimY=1;cfg.gridDimZ=1
  cfg.blockDimX=512;cfg.blockDimY=1;cfg.blockDimZ=1;cfg.sharedMemBytes=self.smem;cfg.attrs=[attr];cfg.numAttrs=1
  self.cfg=cfg;self.attr=attr
  self.active_clusters=int(self.drv._unwrap('cuOccupancyMaxActiveClusters',d.cuOccupancyMaxActiveClusters(d.CUfunction(int(self.k[0].handle)),cfg)))
  print('CLUSTER_ACTIVE_LIMIT',self.active_clusters,flush=True)
  base=(PRE/'base.cuh').read_text().replace('template<bool INPUT> TMN_DEVI void ln_bwd(const Params& p){',(R/'ln_aggregate.cuh').read_text()+'\ntemplate<bool INPUT> TMN_DEVI void ln_bwd(const Params& p,uint8_t* sm){')
  base=base.replace(' for(int c=lane;c<C;c+=32){atomicAdd(p.f[INPUT?8:10]+c,gg[c/32]);atomicAdd(p.f[INPUT?9:11]+c,bb[c/32]);}',' aggregate_ln<C,THREADS>(gg,bb,reinterpret_cast<float*>(sm),p.f[INPUT?8:10],p.f[INPUT?9:11]);')
  finish=(PRE/'finish.cu').read_text().replace('#include "base.cuh"',base).replace('front_dx(p,sm,bar,phase);','').replace('ln_bwd<true>(p)','ln_bwd<true>(p,sm)').replace('mw_d256_b7_finish','mw_d256_cluster_finish')
  # base.cuh includes mma.cuh even though its GEMM functions are unused here.
  finish=finish.replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text())
  out=T.compile_text(finish,flags+['-DWIDTH=256']);self.finish=T.load_unit(str(out),'mw_d256_cluster_finish').kernel('mw_d256_cluster_finish')
  self.finish.set_max_dynamic_smem(W.tuning(256)[2]);drv=self.finish.unit.drv
  occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.finish.handle)),256,W.tuning(256)[2])))
  self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
 def sides(self):
  self.cfg.hStream=self.drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream))
  for k in self.k:self.drv._unwrap('cuLaunchKernelEx',self.drv.d.cuLaunchKernelEx(self.cfg,self.drv.d.CUfunction(int(k.handle)),ctypes.addressof(self.args.array),0))
 def __call__(self):
  self.sides();W.launch(self.finish,self.p.params7,self.grid,D=256)
  return self.p.outputs
