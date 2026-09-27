"""Native M128N96 contraction/GP with three resident CTAs as the target."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

def source():
    count=48
    destinations=','.join('%'+str(i) for i in range(count))
    constraints=','.join('"+f"(v['+str(i)+'])' for i in range(count))
    helper=f'''template<int OA,int OB,int TA,int TB> TMN_DEVI void mma96_off(float (&v)[48],uint64_t aa,uint64_t bb,int ac){{
 asm volatile("{{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%48,%53;add.u64 bx,%49,%54;setp.ne.b32 p,%50,0;wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {{{destinations}}},ax,bx,p,1,1,%51,%52;}}" : {constraints} : "l"(aa),"l"(bb),"r"(ac),"n"(TA),"n"(TB),"n"(OA>>4),"n"(OB>>4));
}}
'''
    return Path(__file__).with_suffix('.cu').read_text().replace('// MMA96_HELPER',helper)

class N96ContractGP:
    def __init__(self,plan):
        prior=plan.contract_gp;self.__dict__.update(prior.__dict__);p=plan.p;n=p.n;d=p.D;h=2*d
        body=source();self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={d}',f'-DFIXED_LENGTH={n}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_n96_contract_gp').kernel('mw_wide_n96_contract_gp')
        self.smem=73728+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();ab=p.front.ab
        bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        def bm(x,mode):return L.tensor_map(x,[64,32] if mode==2 else [32,64],dims=[n,d*n],strides_bytes=[n*2],swizzle='128B' if mode==2 else '64B',l2='128B')
        def epi(x,ch):return L.tensor_map(x,[32,128],dims=[n,ch*n],strides_bytes=[n*2],swizzle='64B',l2='128B')
        fields=prior.params.fields.copy();fields[4:8]=[bm(x,i) for i,x in enumerate(bb)]
        # Four A maps, four B maps, pre/mask pointers, four GP pointers, dL/dR,
        # then the pre, mask, and four output maps.
        fields[16:22]=[epi(plan.pre,8*d),epi(plan.f.mask,1),*[epi(x,h) for x in p.gp]]
        self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,256,self.smem)))
    def __call__(self):self.k.launch((4*self.p.D*(self.p.n//128)*(self.p.n//96),1,1),(256,1,1),[self.params],self.smem)
