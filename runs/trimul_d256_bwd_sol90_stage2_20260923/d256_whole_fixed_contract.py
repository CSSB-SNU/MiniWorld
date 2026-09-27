"""Two whole-map TMA input commands per contraction step, fixed ordered K."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class WholeFixedContract:
    def __init__(self,plan,groups=2,slots=3):
        p=plan.p;assert p.D==256 and p.n==384
        root=Path(__file__).resolve().parent
        body=(root/'d256_fixed_contract.cu').read_text().replace('// MMA_HELPER',(root/'mma_offset.cuh').read_text())
        body=body.replace('SLOTS=3,','SLOTS=CONTRACT_SLOTS,').replace('int slot=step%3','int slot=step%SLOTS').replace('(step/3)&1','(step/SLOTS)&1')
        start=body.index(' #pragma unroll\n for(int g=0;g<ROW_GROUPS;')
        end=body.index('\n}\ntemplate<int MODE> TMN_DEVI void run',start)
        body=body[:start]+''' tma_load_3d(sm+slot*INPUT,p.a+MODE,bar+slot,TA?0:ki,TA?ch*N+ki:0,TA?mi/64:(ch*N+mi)/64);
 tma_load_3d(sm+slot*INPUT+ROW_GROUPS*8192,p.b+MODE,bar+slot,TB?0:ki,TB?ch*N+ki:0,TB?ni/64:(ch*N+ni)/64);'''+body[end:]
        old='if(threadIdx.x==0){load<MODE>(p,sm,bar,ch,mi,ni,0);load<MODE>(p,sm,bar,ch,mi,ni,1);}'
        assert body.count(old)==1
        body=body.replace(old,'if(threadIdx.x==0)for(int step=0;step<SLOTS-1;++step)load<MODE>(p,sm,bar,ch,mi,ni,step);')
        body=body.replace('step+2<6','step+SLOTS-1<6').replace('mi,ni,step+2);','mi,ni,step+SLOTS-1);')
        body=body.replace('for(int i=0;i<3;++i)','for(int i=0;i<SLOTS;++i)')
        start=body.index('   #pragma unroll\n   for(int wn=0;wn<2;++wn)')
        end=body.index('\n  }\n  tma_store_commit',start)
        body=body[:start]+'''   asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{0,%2,%3}],[%1];"::"l"(p.out+SIDE),"r"(smem_u32(sm+wm*16384)),"r"((ch+HALF*D)*N+mi+wm*64),"r"(ni/64):"memory");'''+body[end:]
        body=body.replace('mw_d256_fixed_contract','mw_d256_whole_fixed_contract');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DROW_GROUPS={groups}','-DGRID_ORDER=1',f'-DMIN_BLOCKS={4 if groups==1 and slots==2 else 3 if groups==1 else 2}',f'-DCONTRACT_SLOTS={slots}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_whole_fixed_contract').kernel('mw_d256_whole_fixed_contract')
        self.smem=(groups+2)*8192*slots+128;self.threads=128*groups;self.grid=4*256*(384//(64*groups))*3
        self.k.set_max_dynamic_smem(self.smem);L=T._launch_module();ab=p.front.ab;d=256;h=512;n=384
        def tm(t,trans,count):
            if trans:return L.tensor_map(t,[64,64,count],dims=[64,n*d,n//64],strides_bytes=[n*2,128],swizzle='128B',l2='256B')
            return L.tensor_map(t,[64,64,count],dims=[n,64,n*d//64],strides_bytes=[n*2,n*128],swizzle='128B',l2='256B')
        aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        out=lambda t:L.tensor_map(t,[64,64,2],dims=[64,n*h,n//64],strides_bytes=[n*2,128],swizzle='128B',l2='128B')
        self.params=L.Struct([*[tm(t,i==1,groups) for i,t in enumerate(aa)],*[tm(t,i!=2,2) for i,t in enumerate(bb)],out(p.dl),out(p.dr)])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
    def __call__(self):self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
