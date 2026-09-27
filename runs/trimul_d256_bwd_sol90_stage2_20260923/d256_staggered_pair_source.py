"""Share the XN tile across two ranks, with one N256 dW issue per K tile."""
from pathlib import Path
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W

class StaggeredPairSource:
    def __init__(self,p,old,full_n=True,phase=0):
        bulk=True
        self.p=p;self.splits=old.splits;self.mask=old.mask
        root=Path(__file__).resolve().parent
        packed=(root/'packed_glu.cuh').read_text().replace('packed_glu','packed_cluster_glu').replace('smem_u32(s+32768)','smem_u32(s)')
        body=(root/'source_pairs.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text()).replace('// PACKED_HELPER',packed)
        if full_n:
            helper=(root/'mma256.cuh').read_text()
            body=body.replace('constexpr int D=256',helper+'\nconstexpr int D=256')
            body=body.replace('float dw0[64]={},dw1[64]={};','float dw[128]={};auto& dw0=*reinterpret_cast<float(*)[64]>(dw);auto& dw1=*reinterpret_cast<float(*)[64]>(dw+64);')
            old_mma='mma128_off<k*32,k*2048,0,1>(dw0,a,smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);mma128_off<k*32,k*2048,0,1>(dw1,a,smem_desc(smem_u32(xn+16384),8192,1024,1),it>0||k>0);'
            new_mma='mma256<0,1>(dw,a+((k*32)>>4),smem_desc(smem_u32(xn),8192,1024,1)+((k*2048)>>4),it>0||k>0);'
            assert body.count(old_mma)==1;body=body.replace(old_mma,new_mma)
        if bulk:
            point='mbar_arrive_expect_tx(b+slot,INPUT);'
            assert body.count(point)==1
            body=body.replace(point,'''mbar_arrive_expect_tx(b+slot,INPUT+128);
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],128,[%2];"::"r"(smem_u32(sm+BARS+128+slot*128)),"l"(p.mask+tile*64),"r"(smem_u32(b+slot)):"memory");''')
            body=body.replace('p.mask[row+ra]','reinterpret_cast<bf*>(sm+BARS+128+slot*128)[ra]')
            body=body.replace('p.mask[row+ra+8]','reinterpret_cast<bf*>(sm+BARS+128+slot*128)[ra+8]')
        assert phase in (0,1,2,3)
        if phase:
            body=body.replace('for(int i=0;i<5;++i)mbar_init(b+i,i>=3?2:1);','for(int i=0;i<6;++i)mbar_init(b+i,(i>=3&&i<5)?2:1);')
            marker=' for(int tile=begin,it=0;tile<end;++tile,++it){int slot=it%2,row=tile*64;'
            assert body.count(marker)==1
            body=body.replace(marker,' if constexpr(WG==1)mbar_wait(b+5,0);'+marker)
            points={1:'  wgmma_commit();wgmma_wait<0>();fence_regs(pre);',2:'packed_cluster_glu(pre,xn+32768+WG*4096,gp,ma,mb);named_bar_sync(SYNC,128);fence_proxy_async();named_bar_sync(SYNC,128);',3:'if(tid==0)mbar_arrive(b+3+slot);'}
            marker=points[phase];assert body.count(marker)==1
            body=body.replace(marker,marker+'if constexpr(WG==0)if(it==0&&tid==0)mbar_arrive(b+5);')
        body=body.replace('mw_d256_b7_pairs','mw_d256_staggered_pair_source');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_staggered_pair_source').kernel('mw_d256_staggered_pair_source')
        self.smem=163968+(256 if bulk else 0);self.k.set_max_dynamic_smem(self.smem)
        launch=T._launch_module();dy=lambda x:launch.tensor_map(x,[64,32],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
        self.params=launch.Struct([p.maps[0],W.tm(p.w1),dy(p.dl),dy(p.dr),*[dy(x) for x in p.gp],self.mask,p.floats[7],p.M])

        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))
        assert self.registers*384>=128*(32+224*2),'insufficient dynamic register pool'

    def __call__(self):self.k.launch((16*self.splits,1,1),(384,1,1),[self.params],self.smem)
