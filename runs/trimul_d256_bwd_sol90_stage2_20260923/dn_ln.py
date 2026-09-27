from pathlib import Path
import ctypes,torch,os,re
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
R=Path(__file__).resolve().parent
class DNLN:
 def __init__(self,p):
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc')]
  self.smem=163968
  if os.environ.get('DN_WIDE')=='1':
   body=(R/'dn_ln.cu').read_text().replace('WB=131072,BARS=163840','WB=98304,BARS=229376')
   body=re.sub(r'template <int OFF_BYTES>.*?extern "C" __global__',(R/'mma256.cuh').read_text()+'\n'+(R/'dn_wide.cuh').read_text()+'\nextern "C" __global__',body,flags=re.S)
   if os.environ.get('DN_SPLIT')=='1':
    body=(R/'dn_ln.cu').read_text().replace('WB=131072,BARS=163840','WB=98304,BARS=229376')
    body=re.sub(r'template <int OFF_BYTES>.*?extern "C" __global__',(R/'mma_offset.cuh').read_text()+'\n'+(R/'dn_split.cuh').read_text()+'\nextern "C" __global__',body,flags=re.S)
   if os.environ.get('DN_STATIC')=='1':
    body=body.replace('for(int j=0;j<128;++j){','static_for<128>([&](auto jj){constexpr int j=decltype(jj)::value;')
    body=body.replace('=__float2bfloat16_rn(v[j]);}','=__float2bfloat16_rn(v[j]);});')
   if os.environ.get('DN_SHARED_SUMS')=='1':
    from dn_shared_sums import transform
    body=transform(body)
   out=T.compile_text(body,flags);self.smem=229504
  else:out=T.compile(R/'dn_ln.cu',flags)
  print('DNLN_CUBIN',str(out),flush=True)
  self.k=T.load_unit(str(out),'mw_d256_dn_ln').kernel('mw_d256_dn_ln');self.k.set_max_dynamic_smem(self.smem)
  L=T._launch_module();tm=lambda t:L.tensor_map(t,[64,64],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
  self.params=L.Struct([tm(p.tri),tm(p.dt),p.maps[4],p.maps[1],p.tensors[9],p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.M])
  self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,256,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
