"""Exact N192 contraction tiles divide the short spatial dimension evenly."""
from pathlib import Path
from d256_whole_fixed_contract import WholeFixedContract
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class N192FixedContract(WholeFixedContract):
    def __init__(self,plan,groups=2,slots=2):
        super().__init__(plan,groups,slots)
        p=plan.p;root=Path(__file__).resolve().parent;body=self.source_text
        body=body.replace('constexpr int D=256,',(root/'mma192_offset.cuh').read_text()+'\nconstexpr int D=256,')
        body=body.replace('INPUT=(ROW_GROUPS+2)*8192','INPUT=(ROW_GROUPS+3)*8192')
        body=body.replace('float v[64]={};','float v[96]={};').replace('mma128_off<','mma192_off<')
        body=body.replace('static_for<32>([&](auto jj)','static_for<48>([&](auto jj)').replace('(wg*2+c/64)*8192','(wg*3+c/64)*8192')
        body=body.replace('sm+wm*16384','sm+wm*24576')
        body=body.replace('TILES=MT*3','TILES=MT*2').replace('(tile/3)*(64*ROW_GROUPS),ni=(tile%3)*128','(tile/2)*(64*ROW_GROUPS),ni=(tile%2)*192')
        body=body.replace('mw_d256_whole_fixed_contract','mw_d256_n192_fixed_contract');self.source_text=body
        minblocks=(3 if slots==2 else 2) if groups==1 else (2 if slots==2 else 1)
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DROW_GROUPS={groups}','-DGRID_ORDER=1',f'-DMIN_BLOCKS={minblocks}',f'-DCONTRACT_SLOTS={slots}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_n192_fixed_contract').kernel('mw_d256_n192_fixed_contract')
        self.smem=(groups+3)*8192*slots+128;self.grid=4*256*(384//(64*groups))*2
        self.k.set_max_dynamic_smem(self.smem);L=T._launch_module();ab=p.front.ab;d=256;h=512;n=384
        def tm(t,trans,count):
            if trans:return L.tensor_map(t,[64,64,count],dims=[64,n*d,n//64],strides_bytes=[n*2,128],swizzle='128B',l2='256B')
            return L.tensor_map(t,[64,64,count],dims=[n,64,n*d//64],strides_bytes=[n*2,n*128],swizzle='128B',l2='256B')
        aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        out=lambda t:L.tensor_map(t,[64,64,3],dims=[64,n*h,n//64],strides_bytes=[n*2,128],swizzle='128B',l2='128B')
        self.params=L.Struct([*[tm(t,i==1,groups) for i,t in enumerate(aa)],*[tm(t,i!=2,3) for i,t in enumerate(bb)],out(p.dl),out(p.dr)])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
