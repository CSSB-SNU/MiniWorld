"""Process adjacent row tiles in one resident CTA before advancing its stride."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from wide_affine_output_ln import AffineOutputLN

class GroupedAffineLN(AffineOutputLN):
    def __init__(self,p,group):
        super().__init__(p)
        root=Path(__file__).resolve().parent
        helpers=(root/'dn_slim.cu').read_text().split('TMN_DEVI uint32_t rawpos')[1].split('TMN_DEVI void producer')[0]
        helpers=('TMN_DEVI uint32_t rawpos'+helpers).replace('kc<8','kc<H/64')
        body=(root/'wide_affine_output_ln.cu').read_text().replace('// TRANSPOSE_HELPERS',helpers)
        marker='TMN_DEVI void normalizer('
        body=body.replace(marker,f'TMN_DEVI int tile_row(int it){{return ((it/{group})*gridDim.x*{group}+blockIdx.x*{group}+it%{group})*ROWS;}}\n'+marker)
        body=body.replace('int first=blockIdx.x*ROWS,step=gridDim.x*ROWS;','int first=tile_row(0);')
        body=body.replace('row+=step,++it','row=tile_row(++it)')
        body=body.replace('row+step','tile_row(it+1)')
        body=body.replace('int row=blockIdx.x*ROWS,it=0;row<p.M;row+=gridDim.x*ROWS,++it','int row=tile_row(0),it=0;row<p.M;row=tile_row(++it)')
        body=body.replace('mw_wide_affine_output_ln','mw_wide_grouped_affine_ln')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_wide_grouped_affine_ln').kernel('mw_wide_grouped_affine_ln');self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.kernel.handle)),256,self.smem)))
        assert occ==self.occupancy
