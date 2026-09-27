"""Half-depth K stages and 64-byte swizzles retain four compute groups per SM."""
from d256_n192_fixed_contract import N192FixedContract
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class K32N192Contract(N192FixedContract):
    def __init__(self,plan,groups=2,slots=3):
        super().__init__(plan,groups,2)
        p=plan.p;body=self.source_text
        body=body.replace('INPUT=(ROW_GROUPS+3)*8192','INPUT=(ROW_GROUPS+3)*4096')
        body=body.replace('ki=step*64','ki=step*32').replace('step<6','step<12').replace('step+SLOTS-1<6','step+SLOTS-1<12')
        body=body.replace('TA?mi/64:(ch*N+mi)/64','TA?mi/32:(ch*N+mi)/32')
        body=body.replace('TB?ni/64:(ch*N+ni)/64','TB?ni/32:(ch*N+ni)/32')
        body=body.replace('sm+slot*INPUT+ROW_GROUPS*8192','sm+slot*INPUT+ROW_GROUPS*4096')
        body=body.replace('static_for<4>([&](auto kk)','static_for<2>([&](auto kk)')
        body=body.replace('k*(TA?2048:32),k*(TB?2048:32)','k*(TA?1024:32),k*(TB?1024:32)')
        body=body.replace('sm+slot*INPUT+wg*8192','sm+slot*INPUT+wg*4096')
        body=body.replace('TA?8192:16,1024,1','TA?2048:16,512,2').replace('TB?8192:16,1024,1','TB?2048:16,512,2')
        body=body.replace('mw_d256_n192_fixed_contract','mw_d256_k32_n192_contract');self.source_text=body
        minblocks=4 if groups==1 and slots==3 else 2
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DROW_GROUPS={groups}','-DGRID_ORDER=1',f'-DMIN_BLOCKS={minblocks}',f'-DCONTRACT_SLOTS={slots}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_k32_n192_contract').kernel('mw_d256_k32_n192_contract')
        self.smem=(groups+3)*4096*slots+128;assert self.smem>groups*24576
        self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();ab=p.front.ab;d=256;h=512;n=384
        def tm(t,trans,count):
            if trans:return L.tensor_map(t,[32,32,count],dims=[32,n*d,n//32],strides_bytes=[n*2,64],swizzle='64B',l2='256B')
            return L.tensor_map(t,[32,32,count],dims=[n,32,n*d//32],strides_bytes=[n*2,n*64],swizzle='64B',l2='256B')
        aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        self.params=L.Struct([*[tm(t,i==1,groups*2) for i,t in enumerate(aa)],*[tm(t,i!=2,6) for i,t in enumerate(bb)],*self.params.fields[-2:]])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
