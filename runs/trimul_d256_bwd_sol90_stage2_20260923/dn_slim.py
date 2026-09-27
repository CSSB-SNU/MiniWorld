from pathlib import Path
import ctypes,torch,os
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
R=Path(__file__).resolve().parent
class DNSlim:
 def __init__(self,p):
  helper=(R/'dn_ln.cu').read_text().split('template <int OFF_BYTES>')[1].split('TMN_DEVI void dnorm')[0]
  body=(R/'dn_slim.cu').read_text().replace('// RS_MMA_HELPER','template <int OFF_BYTES>'+helper)
  rows=int(os.environ.get('DNS_LN_ROWS','32'));assert rows in (16,32)
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DDNS_LN_ROWS={rows}',f'-DDNS_STORE_READ={int(os.environ.get("DNS_STORE_READ","0"))}']
  stats=int(os.environ.get('DNS_STATS_TMA','0'))
  flags += [f'-DDNS_STATS_TMA={stats}',f'-DDNS_CACHE_GAMMA={int(os.environ.get("DNS_CACHE_GAMMA","0"))}']
  out=T.compile_text(body,flags);self.cubin=out;print('DNSLIM_CUBIN',str(out),flush=True)
  self.k=T.load_unit(str(out),'mw_d256_dn_slim').kernel('mw_d256_dn_slim');self.smem=98432+512*stats;self.k.set_max_dynamic_smem(self.smem)
  L=T._launch_module();tm=lambda t:L.tensor_map(t,[rows,64],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='64B' if rows==32 else '32B',l2='128B')
  self.params=L.Struct([tm(p.tri),tm(p.dt),p.maps[4],p.maps[1],p.tensors[9],p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.M])
  drv=self.k.unit.drv;occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),256,self.smem)))
  self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*min(int(os.environ.get('DNS_GRID_MULT','2')),occ)
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,256,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
