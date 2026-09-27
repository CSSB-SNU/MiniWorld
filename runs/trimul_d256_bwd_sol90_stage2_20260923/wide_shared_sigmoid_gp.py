"""Cache exact BF16-input sigmoid values within the two-CTA shared budget."""
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class SharedSigmoidGP:
    def __init__(self,plan):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__)
        body=prior.source_text
        start=body.index(' int tiles=p.N/128,mode=');end=body.index('\n int mi=',start)
        body=body[:start]+''' int tiles=p.N/128;
 int half=blockIdx.x/(2*D*tiles*tiles),rem=blockIdx.x%(2*D*tiles*tiles),ch=rem/(2*tiles*tiles),mode=2*half+rem%2,tile=(rem/2)%(tiles*tiles);'''+body[end:]
        body=body.replace('outmap[4];int N;','outmap[4];int N;const float* sigmoid_table;')
        marker='template<int MODE> TMN_DEVI void load_input('
        helper='''
constexpr unsigned LUT_BASE=0x3a00,LUT_SPAN=0x4200-LUT_BASE;
TMN_DEVI float shared_sigmoid(uint8_t* sm,unsigned raw){
 unsigned mag=raw&0x7fffu;
 if(mag>=LUT_BASE && mag<0x4200u)return reinterpret_cast<float*>(sm+BAR+128)[(raw>>15)*LUT_SPAN+mag-LUT_BASE];
 return math::sigmoid(__uint_as_float(raw<<16));
}
'''
        assert body.count(marker)==1
        body=body.replace(marker,helper+marker)
        marker='math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr))'
        assert body.count(marker)==1
        body=body.replace(marker,'shared_sigmoid(sm,gr&0xffffu),gb=shared_sigmoid(sm,gr>>16)')
        marker=' if(threadIdx.x==0){for(int i=0;i<7;++i)mbar_init'
        assert body.count(marker)==1
        body=body.replace(marker,' for(int i=threadIdx.x;i<2*LUT_SPAN;i+=256)reinterpret_cast<float*>(sm+BAR+128)[i]=p.sigmoid_table[i];\n'+marker)
        body=body.replace('mw_wide_two_group_contract_gp','mw_wide_shared_sigmoid_gp')
        body+='''
extern "C" __global__ void mw_fill_sigmoid_table(float* table){
 unsigned i=blockIdx.x*256+threadIdx.x;if(i>=2*LUT_SPAN)return;
 unsigned raw=(i>=LUT_SPAN?0x8000u:0u)+LUT_BASE+i%LUT_SPAN;
 table[i]=math::sigmoid(__uint_as_float(raw<<16));
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={self.p.D}']
        self.cubin=T.compile_text(body,flags)
        self.smem=98304+128+2*(0x4200-0x3a00)*4
        unit=T.load_unit(str(self.cubin),'mw_wide_shared_sigmoid_gp')
        self.k=unit.kernel('mw_wide_shared_sigmoid_gp');self.k.set_max_dynamic_smem(self.smem)
        self.table=torch.empty(2*(0x4200-0x3a00),device=self.p.x.device,dtype=torch.float32)
        unit.kernel('mw_fill_sigmoid_table').launch(((self.table.numel()+255)//256,1,1),(256,1,1),[self.table],0)
        self.params=T._launch_module().Struct([*prior.params.fields,self.table])
        drv=self.k.unit.drv
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),256,self.smem)))
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)**2,1,1),(256,1,1),[self.params],self.smem)
