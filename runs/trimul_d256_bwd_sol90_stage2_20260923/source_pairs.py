from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent
class PairB7:
 def __init__(self,p,old):
  self.p=p;self.splits=old.splits;self.mask=old.mask;self.wide_finish=old.wide_finish
  packed=(R/'packed_glu.cuh').read_text().replace('packed_glu','packed_cluster_glu').replace('smem_u32(s+32768)','smem_u32(s)')
  body=(R/'source_pairs.cu').read_text().replace('// MMA_HELPERS',(R/'mma_offset.cuh').read_text()).replace('// PACKED_HELPER',packed)
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
  self.cubin=T.compile_text(body,flags);print('PAIR_SOURCE_CUBIN',self.cubin,flush=True)
  self.k=T.load_unit(str(self.cubin),'mw_d256_b7_pairs').kernel('mw_d256_b7_pairs');self.smem=163968;self.k.set_max_dynamic_smem(self.smem)
  L=T._launch_module();dy=lambda x:L.tensor_map(x,[64,32],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
  self.params=L.Struct([p.maps[0],W.tm(p.w1),dy(p.dl),dy(p.dr),*[dy(x) for x in p.gp],self.mask,p.floats[7],p.M])
 def source(self):self.k.launch((16*self.splits,1,1),(384,1,1),[self.params],self.smem)
 def __call__(self):
  self.source();self.wide_finish();return self.p.outputs
