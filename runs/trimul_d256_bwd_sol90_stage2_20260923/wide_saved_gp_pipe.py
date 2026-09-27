"""Direct packed preactivation loads and overlapped TMA GP reads."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class SavedGpPipe:
    def __init__(self,original,pipe=True,rank_major=False,grid_factor=4):
        self.original=original;self.p=original.p;self.params=original.params
        root=Path(__file__).resolve().parent
        glu=(root/'packed_glu.cuh').read_text().replace('void packed_glu(float (&a)[32]','void saved_glu(uint8_t* saved').replace('s+32768','s+8192')
        old='uint32_t gr=pack_bf16(a[q*8+j*2],a[q*8+j*2+1]),pr=pack_bf16(a[(q+2)*8+j*2],a[(q+2)*8+j*2+1]);'
        new='int i=q*8+j*2,r=w*16+lane/4+8*((i/2)&1),c=(i/8)*16+2*(lane%4)+8*((i/2)%4/2);uint32_t gr=*reinterpret_cast<uint32_t*>(saved+swz128(r,c*2)),pr=*reinterpret_cast<uint32_t*>(saved+swz128(r,(c+32)*2));'
        assert glu.count(old)==1;glu=glu.replace(old,new)
        body=(root/'wide_saved_gp_pipe.cu').read_text().replace('// SAVED_GLU',glu)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),
               f'-DWIDTH={self.p.D}',f'-DGP_PIPE={int(pipe)}',f'-DGP_RANK_MAJOR={int(rank_major)}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_saved_gp_pipe').kernel('mw_wide_saved_gp_pipe')
        self.smem=12288*(2 if pipe else 1)+8192+128;self.k.set_max_dynamic_smem(self.smem)
        self.grid=torch.cuda.get_device_properties(self.p.x.device).multi_processor_count*grid_factor

    def derivatives(self):self.k.launch((self.grid,1,1),(128,1,1),[self.params],self.smem)
    def __call__(self):self.derivatives();self.original.matmul()
