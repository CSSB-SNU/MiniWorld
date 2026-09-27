"""Apply the exact fixed native contraction primitive to the two forward products."""
from d256_whole_fixed_contract import WholeFixedContract
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class ForwardFixedContract(WholeFixedContract):
    def __init__(self,plan,groups=2):
        super().__init__(plan,groups,3)
        p=plan.p;body=self.source_text.replace('HALF=(MODE>=2)','HALF=(MODE==1)')
        begin=body.index(' int mode,ch,tile;');end=body.index(' int mi=',begin)
        body=body[:begin]+''' int rem=blockIdx.x,ch=rem/(2*TILES),mode=1+rem%2,tile=(rem/2)%TILES;
'''+body[end:]
        body=body.replace('mw_d256_whole_fixed_contract','mw_d256_forward_fixed_contract');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DROW_GROUPS={groups}','-DGRID_ORDER=1',f'-DMIN_BLOCKS={3 if groups==1 else 2}','-DCONTRACT_SLOTS=3']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_forward_fixed_contract').kernel('mw_d256_forward_fixed_contract')
        self.grid=2*256*(384//(64*groups))*3;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();ab=p.front.ab;d=256;h=512;n=384
        def tm(t,trans,count):
            if trans:return L.tensor_map(t,[64,64,count],dims=[64,n*d,n//64],strides_bytes=[n*2,128],swizzle='128B',l2='256B')
            return L.tensor_map(t,[64,64,count],dims=[n,64,n*d//64],strides_bytes=[n*2,n*128],swizzle='128B',l2='256B')
        aa=(ab[:d],ab[d:h],ab[:d],ab[:d]);bb=(ab[h:h+d],ab[h+d:],ab[h:h+d],ab[h:h+d])
        out=lambda:L.tensor_map(p.tri,[64,64,2],dims=[64,n*h,n//64],strides_bytes=[n*2,128],swizzle='128B',l2='128B')
        self.params=L.Struct([*[tm(t,i==1,groups) for i,t in enumerate(aa)],*[tm(t,i!=2,2) for i,t in enumerate(bb)],out(),out()])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
