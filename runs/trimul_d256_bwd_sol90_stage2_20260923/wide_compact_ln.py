"""Output-LN with narrow TMA row tiles and optional TMA dNorm loads."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class CompactLN:
    def __init__(self,p,rows=32,dn_tma=False,fence=False,minblocks=2,channel_affine=False):
        root=Path(__file__).resolve().parent;h=2*p.D;self.threads=128
        assert rows in (8,16,32)
        self.smem=max(rows*h*2*(1+int(dn_tma))+128,2*4*h*4)
        helpers=(root/'dn_slim.cu').read_text().split('TMN_DEVI uint32_t rawpos')[1].split('TMN_DEVI void producer')[0]
        helpers=('TMN_DEVI uint32_t rawpos'+helpers).replace('kc<8','kc<H/64')
        if rows==8:helpers=(root/'tile_transpose8.cuh').read_text()
        body=(root/'wide_tile_ln.cu').read_text().replace('// TRANSPOSE_HELPERS',helpers)
        body=body.replace('// AFFINE_HELPER',(root/'ln_aggregate.cuh').read_text())
        if channel_affine:
            # Assign each affine channel to one CTA thread rather than every
            # warp. Each thread keeps H/128 sums instead of H/32 sums live.
            body=body.replace('float gg[H/32]={},bb[H/32]={};','float gg[H/NT]={},bb[H/NT]={};')
            marker='  for(int r=warp;r<ROWS;r+=4){'
            assert body.count(marker)==1
            body=body.replace(marker,'''  #pragma unroll
  for(int q=0;q<H/NT;++q){int c=tid+q*NT;
   #pragma unroll 1
   for(int r=0;r<ROWS;++r){
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-p.mu[row+r])*p.rs[row+r];
    float dy=getdy(p,sm,row,r,c);gg[q]+=dy*z;bb[q]+=dy;
   }
  }
  __syncthreads();
'''+marker)
            body=body.replace('s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;','s0+=v;s1+=v*z;')
            body=body.replace(' aggregate_ln<H,NT>(gg,bb,reinterpret_cast<float*>(sm),p.dg,p.db);',' for(int q=0;q<H/NT;++q){int c=tid+q*NT;atomicAdd(p.dg+c,gg[q]);atomicAdd(p.db+c,bb[q]);}')
            body=body.replace('mw_wide_tile_ln','mw_wide_channel_ln')
        self.smem+=h*4
        body=body.replace('p.gamma[c]','reinterpret_cast<float*>(sm+SB*(1+LN_DN_TMA)+128)[c]')
        marker=' __syncthreads();cooperative_groups::this_grid().sync();'
        assert body.count(marker)==1
        body=body.replace(marker,' for(int c=tid;c<H;c+=NT)reinterpret_cast<float*>(sm+SB*(1+LN_DN_TMA)+128)[c]=p.gamma[c];\n'+marker)
        body=body.replace('mw_wide_tile_ln','mw_wide_compact_ln')
        flags=['-std=c++17' ,'-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),
               '-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DLN_ROWS={rows}',
               f'-DLN_DN_TMA={int(dn_tma)}',f'-DLN_FENCE={int(fence)}',f'-DLN_MINBLOCKS={minblocks}']
        self.cubin=T.compile_text(body,flags)
        name='mw_wide_channel_ln' if channel_affine else 'mw_wide_compact_ln'
        self.kernel=T.load_unit(str(self.cubin),name).kernel(name)
        self.kernel.set_max_dynamic_smem(self.smem);drv=self.kernel.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.kernel.handle)),128,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module()
        tm=lambda t:L.tensor_map(t,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='none' if rows==8 else '64B' if rows==32 else '32B',l2='128B')
        dn=L.tensor_map(p.tensors[9],[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.params=L.Struct([tm(p.tri),tm(p.dt),dn,p.tensors[9],p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.M])

    def __call__(self):
        L=T._launch_module();drv=self.kernel.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.kernel.handle)),self.grid,1,1,128,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
