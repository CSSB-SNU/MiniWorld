"""Both normalization finishes use affine buffers zeroed by output gate."""
from pathlib import Path
from early_affine_init import EarlyAffineGate
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class EarlyBothAffineGate(EarlyAffineGate):
    def __init__(self, plan):
        super().__init__(plan)
        p=plan.p
        body=self.source_text.replace('float *affine_g,*affine_b;', 'float *affine_g,*affine_b,*input_g,*input_b;')
        marker=' for(int c=blockIdx.x*128+tid;c<2*D;c+=gridDim.x*128){p.affine_g[c]=0;p.affine_b[c]=0;}'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'\n for(int c=blockIdx.x*128+tid;c<D;c+=gridDim.x*128){p.input_g[c]=0;p.input_b[c]=0;}')
        body=body.replace('mw_early_affine_gate','mw_early_both_affine_gate')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}']
        if p.D!=512:flags.append('-DTMN_SIGMOID_TANH=1')
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_early_both_affine_gate').kernel('mw_early_both_affine_gate')
        self.k.set_max_dynamic_smem(self.smem)
        self.params=T._launch_module().Struct([*self.params.fields,p.floats[8],p.floats[9]])


class IndependentInputLN:
    def __init__(self,plan,grid_scale=1,packed=False):
        p=plan.p;original=plan.dx.reduce_only
        self.__dict__.update(original.__dict__)
        assert type(original).__name__ in ('CachedInput','JointInputReduce')
        joint=type(original).__name__=='JointInputReduce'
        splits=plan.b7.splits if p.D==256 else plan.input_splits
        root=Path(__file__).resolve().parent
        body=(root/'wide_tma_input.cu').read_text().replace('// AFFINE_HELPER',(root/'ln_aggregate.cuh').read_text())
        body=body.replace('s<32',f's<{splits}')
        if joint:
            body=body.replace('bf* dw[4];int M;','bf* dw[4];bf* gate;int M;')
            marker=' for(int which=0;which<4;++which){'
            body=body.replace(marker,f''' for(int i=blockIdx.x*NT+tid;i<D*D;i+=gridDim.x*NT){{float v=0;
  #pragma unroll
  for(int s=0;s<{splits};++s)v+=p.part[size_t(s)*11*D*D+2*D*D+i];
  p.gate[i]=__float2bfloat16_rn(v);
 }}
'''+marker)
        body=body.replace('p.gamma[c]','reinterpret_cast<float*>(sm+3*SB+128)[c]')
        marker=' __syncthreads();cooperative_groups::this_grid().sync();'
        assert body.count(marker)==1
        body=body.replace(marker,' for(int c=tid;c<D;c+=NT)reinterpret_cast<float*>(sm+3*SB+128)[c]=p.gamma[c];\n __syncthreads();')
        zero=' for(int c=blockIdx.x*NT+tid;c<D;c+=gridDim.x*NT){p.dg[c]=0;p.db[c]=0;}'
        assert body.count(zero)==1
        body=body.replace(zero,'')
        if packed:
            start=body.index(' for(int which=0;which<4;++which){')
            body=body[:start]+f'''
 for(int i=blockIdx.x*NT+tid;i<4*D*D;i+=gridDim.x*NT){{
  int which=i/(D*D),j=(i%(D*D))*2;float a=0,b=0;
  #pragma unroll
  for(int s=0;s<{splits};++s){{float2 v=*reinterpret_cast<const float2*>(p.part+size_t(s)*11*D*D+(3+2*which)*D*D+j);a+=v.x;b+=v.y;}}
  *reinterpret_cast<uint32_t*>(p.dw[which]+j)=pack_bf16(a,b);
 }}
}}
'''
        body=body.replace('mw_wide_tma_input','mw_independent_input_ln')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}','-DINPUT_ROWS=16','-DINPUT_THREADS=128','-DINPUT_MINBLOCKS=4']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_independent_input_ln').kernel('mw_independent_input_ln')
        self.kernel.set_max_dynamic_smem(self.smem)
        self.grid*=grid_scale
        drv=self.kernel.unit.drv;fn=drv.d.CUfunction(int(self.kernel.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')

    def __call__(self):
        self.kernel.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
