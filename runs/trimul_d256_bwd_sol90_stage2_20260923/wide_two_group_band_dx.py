"""M128-channel by N128-spatial dX tiles use three32KB input slots."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_persistent_gp_dx_overlap import PersistentGPDX


class TwoGroupBandDX:
    def __init__(self,plan):
        p=plan.p;d=p.D;channels=d//128;assert p.n==384
        self.band_grid=p.M//3//128*channels
        root=Path(__file__).resolve().parent
        body=(root/'d256_full_width_prefix_dx.cu').read_text().replace('// MMA_HELPER',(root/'mma_offset.cuh').read_text())
        body=body.replace('STAGE=768*KD','STAGE=512*KD').replace('STEPS=2304/KD',f'STEPS={9*d}/KD')
        body=body.replace('CUtensorMap input,weight,output;','CUtensorMap input,weight,output;int band;')
        body=body.replace('if(threadIdx.x)return;','if(threadIdx.x!=256)return;')
        work=f'(blockIdx.x+p.band*{self.band_grid})'
        body=body.replace('&p.input,bars+slot,0,step*KD,blockIdx.x*2);',f'&p.input,bars+slot,0,step*KD,({work}%{channels})*2);')
        body=body.replace('&p.weight,bars+slot,0,step*KD,0);',f'&p.weight,bars+slot,0,step*KD,({work}/{channels})*2);')
        body=body.replace('float acc[128]={};','float acc[64]={};').replace('mma256<1,1>','mma128_off<0,0,1,1>')
        body=body.replace('sm+WG*32768','sm+WG*16384').replace('static_for<64>([&](auto jj)','static_for<32>([&](auto jj)')
        marker='*reinterpret_cast<uint32_t*>(out+(c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(acc[j],acc[j+1]);'
        assert body.count(marker)==1
        body=body.replace(marker,'*reinterpret_cast<bf*>(out+swz128(c,r*2))=__float2bfloat16_rn(acc[j]);\n  *reinterpret_cast<bf*>(out+swz128(c+1,r*2))=__float2bfloat16_rn(acc[j+1]);')
        start=body.index('  int row=blockIdx.x*128+WG*64;');end=body.index('  tma_store_commit();',start)
        body=body[:start]+f'''  int row=({work}/{channels})*128,col=({work}%{channels})*128+WG*64;
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{{%2,%3}}],[%1];"::"l"(&p.output),"r"(smem_u32(out)),"r"(col),"r"(row):"memory");
'''+body[end:]
        marker=' if(tid<128){setmaxnreg_dec<32>();produce(p,sm,bars);}\n else{setmaxnreg_inc<224>();if(tid<256)consume<0>(p,sm,bars);else consume<1>(p,sm,bars);}'
        assert body.count(marker)==1
        body=body.replace(marker,' if(tid>=256)produce(p,sm,bars);else if(tid<128)consume<0>(p,sm,bars);else consume<1>(p,sm,bars);')
        body=body.replace('__launch_bounds__(384,1)','__launch_bounds__(288,2)').replace('mw_d256_full_width_prefix_dx','mw_wide_two_group_band_dx');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDX_FULL_SLOTS=3','-DDX_FULL_K=64','-DDX_FULL_DEPTH=1']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_two_group_band_dx').kernel('mw_wide_two_group_band_dx')
        self.smem=98304+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        a=L.tensor_map(plan.dx.weights,[64,64,2],dims=[64,9*d,d//64],strides_bytes=[2*d,128],swizzle='128B',l2='256B')
        b=L.tensor_map(plan.dx.input,[64,64,2],dims=[64,9*d,p.M//64],strides_bytes=[p.M*2,128],swizzle='128B',l2='256B')
        y=L.tensor_map(p.tensors[10],[64,128],dims=[d,p.M],strides_bytes=[2*d],swizzle='128B',l2='128B')
        self.band_params=[L.Struct([a,b,y,i]) for i in range(3)]
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')

    def band(self,i):self.k.launch((self.band_grid,1,1),(288,1,1),[self.band_params[i]],self.smem)


class TwoGroupGPDX(PersistentGPDX):
    def __init__(self,plan,grid=132):
        super().__init__(plan,grid)
        self.local_bytes-=self.native_dx.local_bytes
        self.native_dx=TwoGroupBandDX(plan)
        self.local_bytes+=self.native_dx.local_bytes
        self.dx=[lambda i=i:self.native_dx.band(i) for i in range(3)]
