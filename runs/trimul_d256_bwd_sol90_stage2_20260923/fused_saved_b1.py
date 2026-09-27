"""Saved-product gate gradients, dNorm and output LN in one native kernel."""
from pathlib import Path
import ctypes,torch,os
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
R=Path(__file__).resolve().parent


class GateDN:
 def __init__(self,p,proj,gate):
  helper=(R/'dn_ln.cu').read_text().split('template <int OFF_BYTES>')[1].split('TMN_DEVI void dnorm')[0]
  body=(R/'dn_slim.cu').read_text().replace('// RS_MMA_HELPER','template <int OFF_BYTES>'+helper)
  body=body.replace('mw_d256_dn_slim','mw_d256_saved_gate_dn_ln')
  body=body.replace('CUtensorMap tri,dt,dp,wp;','CUtensorMap tri,dt,dp,wp,proj_in,gate_in,dy_in,dg_out;')
  body=body.replace('int M;','const bf* ds;int M,L;')
  body=body.replace('TMN_DEVI void producer', '''TMN_DEVI float saved_sigmoid(float g){float t;float x=__fmul_rn(.5f,g);asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(x));return __fmaf_rn(t,.5f,.5f);}
TMN_DEVI void producer''')
  hook='  mbar_arrive_expect_tx(bars,32768);\n  for(int k=0;k<4;++k)tma_load_2d(sm+BUF+k*8192,&p.dp,bars,k*64,row);'
  assert body.count(hook)==1
  body=body.replace(hook,'''  mbar_arrive_expect_tx(bars,98304);
  for(int k=0;k<4;++k){tma_load_2d(sm+k*8192,&p.proj_in,bars,k*64,row);tma_load_2d(sm+32768+k*8192,&p.gate_in,bars,k*64,row);tma_load_2d(sm+BUF+k*8192,&p.dy_in,bars,k*64,row);}
''')
  hook='  mbar_wait(bars,round&1);named_bar_sync(1,128);'
  assert body.count(hook)==1
  body=body.replace(hook,hook+'''
  for(int r=warp;r<64;r+=4)for(int c=lane*2;c<256;c+=64){
   int off=(c/64)*8192+swz128(r,(c%64)*2);
   uint32_t pr=*reinterpret_cast<uint32_t*>(sm+off),ga=*reinterpret_cast<uint32_t*>(sm+32768+off),dy=*reinterpret_cast<uint32_t*>(sm+BUF+off);
   uint32_t ds=reinterpret_cast<const uint32_t*>(p.ds)[(((row+r)%p.L)*256+c)/2];
   float a=math::round_bf16(bf16lo(dy)*bf16lo(ds)),b=math::round_bf16(bf16hi(dy)*bf16hi(ds));
   float g0=saved_sigmoid(bf16lo(ga)),g1=saved_sigmoid(bf16hi(ga));
   *reinterpret_cast<uint32_t*>(sm+BUF+off)=pack_bf16(a*g0,b*g1);
   *reinterpret_cast<uint32_t*>(sm+off)=pack_bf16(((a*bf16lo(pr))*g0)*(1-g0),((b*bf16hi(pr))*g1)*(1-g1));
  }
  named_bar_sync(1,128);fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){for(int c=0;c<256;c+=64){put_tile(&p.dp,sm+BUF+(c/64)*8192,c,row);put_tile(&p.dg_out,sm+(c/64)*8192,c,row);}tma_store_commit();}
''')
  hook='  fence_proxy_async();named_bar_sync(1,128);if(tid==0)mbar_arrive_dep(bars+1,zero_dep(dep));'
  assert body.count(hook)==1
  wait='tma_store_wait_read<0>()' if os.environ.get('FUSED_GP_WAIT_READ')=='1' else 'tma_store_wait_all()'
  body=body.replace(hook,f'  if(tid==0){wait};named_bar_sync(1,128);\n'+hook)
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDNS_LN_ROWS=32','-DDNS_STORE_READ=0','-DDNS_STATS_TMA=1','-DDNS_CACHE_GAMMA=0']
  self.cubin=T.compile_text(body,flags)
  self.k=T.load_unit(str(self.cubin),'mw_d256_saved_gate_dn_ln').kernel('mw_d256_saved_gate_dn_ln');self.smem=98944;self.k.set_max_dynamic_smem(self.smem)
  L=T._launch_module()
  tm=lambda t:L.tensor_map(t,[32,64],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='64B',l2='128B')
  rowmap=lambda t:F.tm(t,[64,64],[256,p.M],[512])
  self.params=L.Struct([tm(p.tri),tm(p.dt),p.maps[4],p.maps[1],rowmap(proj),rowmap(gate),rowmap(p.dy),rowmap(p.dg),p.tensors[9],p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.ds,p.M,p.n])
  drv=self.k.unit.drv;occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),256,self.smem)))
  self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,256,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))


class FusedSavedB1:
 def __init__(self,old):
  self.p,self.wp=old.p,old.wp
  self.fused=GateDN(self.p,old.prepare.original.proj,old.prepare.original.gate)
 def __call__(self):
  p=self.p;self.fused()
  torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp)
  torch.mm(p.dg.t(),p.xn.reshape(p.M,256),out=p.dwg)


def enable(plan):
 plan.b1=FusedSavedB1(plan.b1)
 k=plan.b1.fused
 return dict(name='fused_saved_b1',cubin=str(k.cubin),smem=k.smem,grid=k.grid)
