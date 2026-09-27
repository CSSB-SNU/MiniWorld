"""Three compute groups per CTA with equal aggregate operand bytes to M128N128."""
from wide_n96_contract_gp import N96ContractGP
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class M192N96ContractGP(N96ContractGP):
    def __init__(self,plan,unroll=True):
        super().__init__(plan)
        body=self.source_text
        body=body.replace('INPUT=28672,PLANE=24576,BAR=3*PLANE','INPUT=36864,PLANE=36864,BAR=3*PLANE')
        body=body.replace('for(int g=0;g<2;++g){int m=', 'for(int g=0;g<3;++g){int m=')
        body=body.replace('slot*INPUT+16384','slot*INPUT+24576')
        body=body.replace('if(threadIdx.x==0)load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);',
            'if(threadIdx.x==0){load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);load_input<MODE>(p,sm,bar,ch,mi,ni,64,1);}')
        body=body.replace('#pragma unroll 1', '#pragma unroll' if unroll else '#pragma unroll 1')
        body=body.replace('int slot=it%2;mbar_wait(bar+slot,(it/2)&1);__syncthreads();',
            '''int slot=it%3;mbar_wait(bar+slot,(it/3)&1);__syncthreads();
  if(threadIdx.x==0){
   if(ki==N-128)load_epi<MODE,0>(p,sm,bar,ch,mi,ni);
   if(ki==N-64)load_epi<MODE,1>(p,sm,bar,ch,mi,ni);
  }''')
        body=body.replace('if(threadIdx.x==0&&ki+64<N)load_input<MODE>(p,sm,bar,ch,mi,ni,ki+64,slot^1);',
            'if(threadIdx.x==0&&ki+128<N)load_input<MODE>(p,sm,bar,ch,mi,ni,ki+128,(it+2)%3);')
        begin=body.index(' if(threadIdx.x==0){\n  constexpr int SIDE=')
        end=body.index(' mbar_wait(bar+2,0);__syncthreads();',begin)
        body=body[:begin]+' if(threadIdx.x==0)load_epi<MODE,2>(p,sm,bar,ch,mi,ni);\n'+body[end:]
        body=body.replace('mbar_wait(bar+2,0)','mbar_wait(bar+3,0)')
        body=body.replace('(c/32)*8192','(c/32)*12288')
        body=body.replace('tile*8192','tile*12288')
        body=body.replace('__launch_bounds__(256,3)','__launch_bounds__(384,2)')
        body=body.replace('TM=N/128,TN=N/96','TM=N/192,TN=N/96').replace('mi=(tile/TN)*128','mi=(tile/TN)*192')
        body=body.replace('for(int i=0;i<3;++i)mbar_init','for(int i=0;i<4;++i)mbar_init')
        helper='''
template<int MODE,int PLANE_INDEX> TMN_DEVI void load_epi(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
 if constexpr(PLANE_INDEX==0)mbar_arrive_expect_tx(bar+3,3*PLANE);
 for(int col=0;col<3;++col){
  if constexpr(PLANE_INDEX<2)tma_load_2d(sm+PLANE_INDEX*PLANE+col*12288,&p.premap,bar+3,ni+col*32,(rank*64+pc+32*PLANE_INDEX)*N+mi);
  else tma_load_2d(sm+2*PLANE+col*12288,&p.maskmap,bar+3,ni+col*32,mi);
 }
}
'''
        body=body.replace('template<int MODE> TMN_DEVI void run(',helper+'template<int MODE> TMN_DEVI void run(')
        body=body.replace('mw_wide_n96_contract_gp','mw_wide_m192n96_contract_gp')
        self.source_text=body;p=plan.p;n=p.n;d=p.D;h=2*d
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DFIXED_LENGTH={n}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_m192n96_contract_gp').kernel('mw_wide_m192n96_contract_gp')
        self.smem=110592+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        def epi(x,ch):return L.tensor_map(x,[32,192],dims=[n,ch*n],strides_bytes=[n*2],swizzle='64B',l2='128B')
        fields=self.params.fields.copy();fields[16:22]=[epi(plan.pre,8*d),epi(plan.f.mask,1),*[epi(x,h) for x in p.gp]]
        self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))

    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//192)*(self.p.n//96),1,1),(384,1,1),[self.params],self.smem)
