"""Two 80KB GP CTAs can coexist with a 36KB, 160-thread dX CTA."""
from pathlib import Path
from wide_byte_mask_gp import ByteMaskGP
from wide_spatial_gp_dx_overlap import SpatialGPDX
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class SmallBandDX:
    def __init__(self,plan):
        p=plan.p;d=p.D;q=9*d;channels=d//64;root=Path(__file__).resolve().parent
        self.band_grid=p.M//3//128*channels
        body=(root/'d256_full_width_prefix_dx.cu').read_text().replace('// MMA_HELPER',(root/'mma_offset.cuh').read_text())
        body=body.replace('STAGE=768*KD','STAGE=384*KD').replace('STEPS=2304/KD',f'STEPS={q}/KD')
        body=body.replace('CUtensorMap input,weight,output;','CUtensorMap input,weight,output;int band;')
        body=body.replace('if(threadIdx.x)return;','if(threadIdx.x!=128)return;')
        work=f'(blockIdx.x+p.band*{self.band_grid})'
        body=body.replace('&p.input,bars+slot,0,step*KD,blockIdx.x*2);',f'&p.input,bars+slot,0,step*KD,{work}%{channels});')
        body=body.replace('&p.weight,bars+slot,0,step*KD,0);',f'&p.weight,bars+slot,0,step*KD,({work}/{channels})*2);')
        body=body.replace('256*KD','128*KD').replace('named_bar_sync(3,256);','')
        body=body.replace('float acc[128]={};','float acc[64]={};').replace('mma256<1,1>','mma128_off<0,0,1,1>')
        body=body.replace('static_for<64>([&](auto jj)','static_for<32>([&](auto jj)')
        marker='*reinterpret_cast<uint32_t*>(out+(c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(acc[j],acc[j+1]);'
        assert body.count(marker)==1
        body=body.replace(marker,'*reinterpret_cast<bf*>(out+swz128(c,r*2))=__float2bfloat16_rn(acc[j]);\n  *reinterpret_cast<bf*>(out+swz128(c+1,r*2))=__float2bfloat16_rn(acc[j+1]);')
        start=body.index('  int row=blockIdx.x*128+WG*64;');end=body.index('  tma_store_commit();',start)
        body=body[:start]+f'''  int row=({work}/{channels})*128,col=({work}%{channels})*64;
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{{%2,%3}}],[%1];"::"l"(&p.output),"r"(smem_u32(out)),"r"(col),"r"(row):"memory");
'''+body[end:]
        body=body.replace('mbar_init(bars+i,i<NSLOT?1:2)','mbar_init(bars+i,1)')
        marker=' if(tid<128){setmaxnreg_dec<32>();produce(p,sm,bars);}\n else{setmaxnreg_inc<224>();if(tid<256)consume<0>(p,sm,bars);else consume<1>(p,sm,bars);}'
        assert body.count(marker)==1
        body=body.replace(marker,' if(tid>=128)produce(p,sm,bars);else consume<0>(p,sm,bars);')
        body=body.replace('__launch_bounds__(384,1)','__launch_bounds__(160,4)').replace('mw_d256_full_width_prefix_dx','mw_wide_small_band_dx')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DDX_FULL_SLOTS=3','-DDX_FULL_K=32','-DDX_FULL_DEPTH=1']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_small_band_dx').kernel('mw_wide_small_band_dx')
        self.smem=36864+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        a=L.tensor_map(plan.dx.weights,[64,32,1],dims=[64,q,channels],strides_bytes=[2*d,128],swizzle='128B',l2='256B')
        b=L.tensor_map(plan.dx.input,[64,32,2],dims=[64,q,p.M//64],strides_bytes=[p.M*2,128],swizzle='128B',l2='256B')
        y=L.tensor_map(p.tensors[10],[64,128],dims=[d,p.M],strides_bytes=[2*d],swizzle='128B',l2='128B')
        self.band_params=[L.Struct([a,b,y,i]) for i in range(3)]
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,160,self.smem)))

    def band(self,i):self.k.launch((self.band_grid,1,1),(160,1,1),[self.band_params[i]],self.smem)


class ByteMaskOverlap(SpatialGPDX):
    def __init__(self,plan,overlap=True):
        # The parent expects checkpoint23's staged source. Keep its compiler input
        # local; selected checkpoint24 is restored before any execution.
        original=plan.contract_gp
        from wide_two_group_contract_gp import TwoGroupContractGP
        from wide_staged_epilogue_gp import StagedEpilogueGP
        plan.contract_gp=TwoGroupContractGP(plan);plan.contract_gp=StagedEpilogueGP(plan)
        try:super().__init__(plan,overlap,False)
        finally:plan.contract_gp=original
        self.gp_op=ByteMaskGP(plan,True);self.native_dx=SmallBandDX(plan)
        self.dx=[lambda i=i:self.native_dx.band(i) for i in range(3)]
        self.registers=(self.gp_op.registers,self.native_dx.registers)
        self.local_bytes=self.gp_op.local_bytes+self.native_dx.local_bytes
        self.cubin=self.gp_op.cubin

    def producer(self,i):
        if i==0:self.gp_op.encode()
        self.gp_op.band(i)
