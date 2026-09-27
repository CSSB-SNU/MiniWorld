"""One complete N384 output row tile reuses A across all spatial columns."""
from pathlib import Path
from d256_whole_fixed_contract import WholeFixedContract
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class FullSpatialContract(WholeFixedContract):
    def __init__(self,plan,groups=2,slots=3,left=256):
        super().__init__(plan,groups,slots)
        assert left in (192,256)
        right=384-left;p=plan.p
        root=Path(__file__).resolve().parent
        body=self.source_text
        helper=(root/'mma256.cuh').read_text()
        helper=helper.replace('template<int TA,int TB>','template<int OA,int OB,int TA,int TB>').replace('void mma256(','void mma256_off(')
        helper=helper.replace('{.reg .pred p;setp.ne.b32 p,%130,0;', '{.reg .pred p;.reg .b64 ax,bx;add.u64 ax,%128,%133;add.u64 bx,%129,%134;setp.ne.b32 p,%130,0;')
        helper=helper.replace(',%128,%129,p,1,1,',',ax,bx,p,1,1,').replace('"n"(TB));','"n"(TB),"n"(OA>>4),"n"(OB>>4));')
        body=body.replace('constexpr int D=256,',helper+'\n'+(root/'mma192_offset.cuh').read_text()+'\nconstexpr int D=256,')
        body=body.replace('INPUT=(ROW_GROUPS+2)*8192','INPUT=(ROW_GROUPS+6)*8192')
        body=body.replace('float v[64]={};',f'float v0[{left//2}]={{}},v1[{right//2}]={{}};')
        body=body.replace('fence_regs(v);','fence_regs(v0);fence_regs(v1);')
        old='mma128_off<k*(TA?2048:32),k*(TB?2048:32),TA,TB>(v,smem_desc(smem_u32(sm+slot*INPUT+wg*8192),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+ROW_GROUPS*8192),TB?8192:16,1024,1),step>0||k>0);'
        assert body.count(old)==1
        ops=[]
        for i,(width,offset) in enumerate(((left,0),(right,left//64*8192))):
            ops.append(f'mma{width}_off<k*(TA?2048:32),{offset}+k*(TB?2048:32),TA,TB>(v{i},smem_desc(smem_u32(sm+slot*INPUT+wg*8192),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+ROW_GROUPS*8192),TB?8192:16,1024,1),step>0||k>0);')
        body=body.replace(old,'\n   '.join(ops))
        begin=body.index(' static_for<32>([&](auto jj)');end=body.index(' fence_proxy_async();',begin)
        stores=[]
        for i,(width,offset) in enumerate(((left,0),(right,left))):
            stores.append(f''' static_for<{width//4}>([&](auto jj){{constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c={offset}+(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  *reinterpret_cast<uint32_t*>(sm+(wg*6+c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(v{i}[j],v{i}[j+1]);
 }});
''')
        body=body[:begin]+''.join(stores)+body[end:]
        body=body.replace('sm+wm*16384','sm+wm*49152')
        body=body.replace('TILES=MT*3','TILES=MT').replace('(tile/3)*(64*ROW_GROUPS),ni=(tile%3)*128','tile*(64*ROW_GROUPS),ni=0')
        body=body.replace('mw_d256_whole_fixed_contract','mw_d256_full_spatial_contract');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DROW_GROUPS={groups}','-DGRID_ORDER=1','-DMIN_BLOCKS=1',f'-DCONTRACT_SLOTS={slots}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_full_spatial_contract').kernel('mw_d256_full_spatial_contract')
        self.smem=(groups+6)*8192*slots+128;self.grid=4*256*(384//(64*groups))
        self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();ab=p.front.ab;d=256;h=512;n=384
        def tm(t,trans,count):
            if trans:return L.tensor_map(t,[64,64,count],dims=[64,n*d,n//64],strides_bytes=[n*2,128],swizzle='128B',l2='256B')
            return L.tensor_map(t,[64,64,count],dims=[n,64,n*d//64],strides_bytes=[n*2,n*128],swizzle='128B',l2='256B')
        aa=(p.dt[:d],p.dt[:d],ab[h+d:],ab[d:h]);bb=(ab[h:h+d],ab[:d],p.dt[d:],p.dt[d:])
        out=lambda t:L.tensor_map(t,[64,64,6],dims=[64,n*h,n//64],strides_bytes=[n*2,128],swizzle='128B',l2='128B')
        self.params=L.Struct([*[tm(t,i==1,groups) for i,t in enumerate(aa)],*[tm(t,i!=2,6) for i,t in enumerate(bb)],out(p.dl),out(p.dr)])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
