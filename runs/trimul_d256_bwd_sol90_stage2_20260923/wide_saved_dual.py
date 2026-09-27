"""Two full dW warpgroup consumers and packed saved preactivations."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class SavedDual:
    def __init__(self,p,original,pre,splits=32):
        root=Path(__file__).resolve().parent;prior=root.parent/'trimul_d256_bwd_sol90_20260923'
        self.p=p;self.splits=splits
        glu=(root/'packed_glu.cuh').read_text().replace('void packed_glu(float (&a)[32]', 'void saved_glu(uint8_t* saved').replace('s+32768','s+8192')
        old='uint32_t gr=pack_bf16(a[q*8+j*2],a[q*8+j*2+1]),pr=pack_bf16(a[(q+2)*8+j*2],a[(q+2)*8+j*2+1]);'
        new='int i=q*8+j*2,r=w*16+lane/4+8*((i/2)&1),c=(i/8)*16+2*(lane%4)+8*((i/2)%4/2);uint32_t gr=*reinterpret_cast<uint32_t*>(saved+swz128(r,c*2)),pr=*reinterpret_cast<uint32_t*>(saved+swz128(r,(c+32)*2));'
        assert glu.count(old)==1;glu=glu.replace(old,new)
        body=(root/'wide_saved_dual.cu').read_text().replace('// MMA_HELPERS',(prior/'mma.cuh').read_text()).replace('// SAVED_GLU',glu)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}',f'-DWEIGHT_SPLITS={splits}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_wide_saved_dual').kernel('mw_wide_saved_dual')
        self.smem=98432;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();prem=L.tensor_map(pre,[64,64],dims=[64,pre.numel()//64],strides_bytes=[128],swizzle='128B',l2='128B')
        self.params=L.Struct([*original.params.fields,prem])
    def __call__(self):self.k.launch((self.p.D//128*self.p.D//16*self.splits,1,1),(256,1,1),[self.params],self.smem)
