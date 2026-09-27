"""Forward normalization with CTA-cached affine parameters and narrow TMA tiles."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

class CachedForwardNorm:
    def __init__(self,p,original,rows=16):
        self.__dict__.update(original.__dict__)
        root=Path(__file__).resolve().parent;h=2*p.D;self.threads=128
        assert p.D in (256,384) and rows in (16,32)
        self.smem=rows*h*2+128+h*8
        body=(root/'wide_tile_output.cu').read_text().replace('// TRANSPOSE_HELPERS',(root/'tile_transpose.cuh').read_text())
        body=body.replace('p.gamma[c]', 'reinterpret_cast<float*>(sm+SB+128)[c]').replace('p.beta[c]', 'reinterpret_cast<float*>(sm+SB+128+H*4)[c]')
        marker=' if(tid==0){mbar_init(bar,1);fence_barrier_init();}__syncthreads();int phase=0;'
        assert body.count(marker)==1
        body=body.replace(marker,''' for(int c=tid;c<H;c+=NT){reinterpret_cast<float*>(sm+SB+128)[c]=p.gamma[c];reinterpret_cast<float*>(sm+SB+128+H*4)[c]=p.beta[c];}
'''+marker).replace('mw_wide_tile_norm','mw_wide_cached_forward_norm')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DOUTPUT_ROWS={rows}','-DOUTPUT_THREADS=128']
        self.cubin=T.compile_text(body,flags)
        self.norm=T.load_unit(str(self.cubin),'mw_wide_cached_forward_norm').kernel('mw_wide_cached_forward_norm');self.norm.set_max_dynamic_smem(self.smem)
        drv=self.norm.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.norm.handle)),128,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module()
        tri=L.tensor_map(p.tri,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='32B' if rows==16 else '64B',l2='128B')
        norm=L.tensor_map(p.tensors[6],[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.np=L.Struct([tri,norm,p.floats[2],p.floats[3],p.floats[5],p.floats[6],p.M])
