"""Whole-map XN/weight/derivative transfers with three shared input slots."""
from d256_staggered_pair_source import StaggeredPairSource
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class DeepPairSource(StaggeredPairSource):
    def __init__(self,p,old,full_n=False,phase=0,slots=3):
        super().__init__(p,old,full_n,phase)
        assert slots in (2,3)
        body=self.source_text
        body=body.replace('WEIGHT=81920,DERIV=147456,BARS=163840','WEIGHT=INPUT*PAIR_SLOTS,DERIV=WEIGHT+65536,BARS=DERIV+16384')
        body=body.replace('b+5','b+2*PAIR_SLOTS+1').replace('b+3+slot','b+PAIR_SLOTS+1+slot').replace('b+2,','b+PAIR_SLOTS,')
        body=body.replace('it%2','it%PAIR_SLOTS').replace('it>=2','it>=PAIR_SLOTS').replace('it/2','it/PAIR_SLOTS')
        for init in ('for(int i=0;i<5;++i)mbar_init(b+i,i>=3?2:1);','for(int i=0;i<6;++i)mbar_init(b+i,(i>=3&&i<5)?2:1);'):
            if init in body:
                body=body.replace(init,'for(int i=0;i<2*PAIR_SLOTS+2;++i)mbar_init(b+i,(i>PAIR_SLOTS&&i<=2*PAIR_SLOTS)?2:1);')
                break
        else:raise AssertionError('missing barrier initialization')
        old_weight='for(int g=0;g<2;++g)for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+g*32768+c*8192,&p.w,b+PAIR_SLOTS,c*64,(rank+g)*64);'
        assert body.count(old_weight)==1
        body=body.replace(old_weight,'for(int g=0;g<2;++g)tma_load_3d(sm+WEIGHT+g*32768,&p.w,b+PAIR_SLOTS,0,(rank+g)*64,0);')
        old_xn='for(int c=0;c<4;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,b+slot,c*64,tile*64);'
        assert body.count(old_xn)==1
        body=body.replace(old_xn,'tma_load_3d(sm+slot*INPUT,&p.xn,b+slot,0,tile*64,0);')
        old_dy='for(int g=0;g<2;++g)tma_load_2d(sm+slot*INPUT+32768+g*4096,rank<16?&p.dl:&p.dr,b+slot,tile*64,((rank+g)%16)*32);'
        assert body.count(old_dy)==1
        body=body.replace(old_dy,'tma_load_3d(sm+slot*INPUT+32768,rank<16?&p.dl:&p.dr,b+slot,tile*64,0,rank%16);')
        body=body.replace('mw_d256_staggered_pair_source','mw_d256_deep_pair_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}',f'-DPAIR_SLOTS={slots}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_deep_pair_source').kernel('mw_d256_deep_pair_source')
        self.smem=40960*slots+81920+128+128*slots;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module()
        xn=L.tensor_map(p.xn,[64,64,4],dims=[64,p.M,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        w=L.tensor_map(p.w1,[64,64,4],dims=[64,2048,4],strides_bytes=[512,128],swizzle='128B',l2='128B')
        dy=lambda x:L.tensor_map(x,[64,32,2],dims=[p.M,32,16],strides_bytes=[p.M*2,p.M*64],swizzle='128B',l2='128B')
        fields=list(self.params.fields);fields[:4]=[xn,w,dy(p.dl),dy(p.dr)];self.params=L.Struct(fields)
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))
        assert self.registers*384>=128*(32+224*2),'insufficient dynamic register pool'
