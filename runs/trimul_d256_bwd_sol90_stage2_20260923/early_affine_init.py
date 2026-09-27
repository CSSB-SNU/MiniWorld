"""Initialize output-affine gradients in the preceding gate kernel."""
from pathlib import Path
import re
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class EarlyAffineGate:
    def __init__(self, plan):
        original = plan.prefix_gate
        self.__dict__.update(original.__dict__)
        p = plan.p
        body = Path(__file__).with_name('prefix_gate_epi.cu').read_text()
        body = body.replace('int M,N;};', 'int M,N;float *affine_g,*affine_b;};')
        marker = ' if(tid==0){mbar_init(bar,1);'
        assert body.count(marker) == 1
        body = body.replace(marker, '''
 for(int c=blockIdx.x*128+tid;c<2*D;c+=gridDim.x*128){p.affine_g[c]=0;p.affine_b[c]=0;}
''' + marker)
        body = body.replace('mw_prefix_gate_epi', 'mw_early_affine_gate')
        self.source_text = body
        flags = ['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}']
        if p.D != 512:
            flags.append('-DTMN_SIGMOID_TANH=1')
        self.cubin = T.compile_text(body, flags)
        self.k = T.load_unit(str(self.cubin),'mw_early_affine_gate').kernel('mw_early_affine_gate')
        self.k.set_max_dynamic_smem(self.smem)
        self.params = T._launch_module().Struct([*original.params.fields,p.floats[10],p.floats[11]])

    def __call__(self):
        self.k.launch((self.grid,1,1),(128,1,1),[self.params],self.smem)

    def launch(self,*args,**kwargs):
        self()


class IndependentOutputLN:
    def __init__(self, p, original, grid_scale=1):
        self.__dict__.update(original.__dict__)
        body = original.source_text
        zero = ' for(int c=blockIdx.x*NT+tid;c<H;c+=gridDim.x*NT){p.dg[c]=0;p.db[c]=0;}'
        assert body.count(zero) == 1
        body = body.replace(zero, '')
        assert body.count('cooperative_groups::this_grid().sync();') == 1
        body = body.replace('cooperative_groups::this_grid().sync();','')
        oldname = re.search(r'void (mw_\w+)\(__grid_constant__ const Params p\)',body).group(1)
        body = body.replace(oldname,'mw_independent_output_ln')
        self.source_text = body
        flags = ['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DLN_ROWS={self.rows}','-DLN_DN_TMA=1','-DLN_FENCE=0',f'-DLN_MINBLOCKS={3 if p.D==256 else 2}']
        self.cubin = T.compile_text(body,flags)
        self.kernel = T.load_unit(str(self.cubin),'mw_independent_output_ln').kernel('mw_independent_output_ln')
        self.kernel.set_max_dynamic_smem(self.smem)
        self.grid *= grid_scale
        drv = self.kernel.unit.drv
        fn = drv.d.CUfunction(int(self.kernel.handle))
        query = lambda n: int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers = query('CU_FUNC_ATTRIBUTE_NUM_REGS')
        self.local_bytes = query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')

    def __call__(self):
        self.kernel.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
