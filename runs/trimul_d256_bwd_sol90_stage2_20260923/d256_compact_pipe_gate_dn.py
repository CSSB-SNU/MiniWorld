"""K32 gate/dNorm slots leave room for three stages and exact ordered MMA."""
from pathlib import Path
from three_role_register_pool import initial_pool_three
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class CompactPipeGateDN:
    def __init__(self,plan,slots=3,pending=False):
        p=plan.p;root=Path(__file__).resolve().parent;assert slots in (2,3)
        body=(root/'d256_pipe_gate_dn.cu').read_text().replace('// MMA256',(root/'mma256.cuh').read_text())
        body=body.replace('SLOT=106496,BAR=2*SLOT','SLOT=53248,NSLOT=GATE_SLOTS,BAR=NSLOT*SLOT')
        body=body.replace('step%2','step%NSLOT').replace('(step/2)&1','(step/NSLOT)&1').replace('step*64','step*32').replace('step<4','step<8')
        body=body.replace('mbar_arrive_expect_tx(b+slot,98304)','mbar_arrive_expect_tx(b+slot,49152)')
        body=body.replace('tma_load_2d(buf+8192,&p.gate','tma_load_2d(buf+4096,&p.gate').replace('tma_load_2d(buf+16384,&p.dy','tma_load_2d(buf+8192,&p.dy').replace('tma_load_2d(buf+24576,&p.ds','tma_load_2d(buf+12288,&p.ds').replace('tma_load_3d(buf+32768,&p.wp','tma_load_3d(buf+16384,&p.wp')
        body=body.replace('buf+8192+off','buf+4096+off').replace('buf+16384+off','buf+8192+off').replace('buf+24576+off','buf+12288+off').replace('buf+98304','buf+49152')
        body=body.replace('for(int q=0;q<4;++q)','for(int q=0;q<2;++q)').replace('static_for<4>([&](auto qq)','static_for<2>([&](auto qq)')
        marker='uint32_t off=swz128(r,c*2);';assert body.count(marker)==1
        body=body.replace(marker,'uint32_t linear=r*64+c*2,off=linear^((linear>>3)&0x30u);')
        body=body.replace('smem_desc(smem_u32(buf+q*32),16,1024,1)','smem_desc(smem_u32(buf+q*32),16,512,2)')
        body=body.replace('smem_desc(smem_u32(buf+32768+WG*32768+q*2048),8192,1024,1)','smem_desc(smem_u32(buf+16384+WG*16384+q*2048),4096,1024,1)')
        body=body.replace('b+2+slot','b+NSLOT+slot').replace('b+4+slot','b+2*NSLOT+slot')
        marker='if(tid==0){load_stage(p,sm,b,row,0);load_stage(p,sm,b,row,1);}';assert body.count(marker)==1
        body=body.replace(marker,'if(tid==0)for(int s=0;s<NSLOT;++s)load_stage(p,sm,b,row,s);')
        marker='if(step>=1 && step+1<4){int previous=1-slot;mbar_wait(b+4+previous,((step-1)/2)&1);load_stage(p,sm,b,row,step+1);}';assert body.count(marker)==1
        body=body.replace(marker,'if(step>=NSLOT-1 && step+1<8){int previous=(step+1)%NSLOT;mbar_wait(b+2*NSLOT+previous,((step+1)/NSLOT-1)&1);load_stage(p,sm,b,row,step+1);}')
        marker='  named_bar_sync(1,128);\n }\n}';assert body.count(marker)==1
        body=body.replace(marker,'  named_bar_sync(1,128);\n }\n if(tid==0)mbar_arrive(b+3*NSLOT);\n}')
        marker=' fence_regs(acc);named_bar_sync(4,256);';assert body.count(marker)==1
        body=body.replace(marker,' fence_regs(acc);mbar_wait(b+3*NSLOT,0);named_bar_sync(4,256);')
        marker='for(int i=0;i<6;++i)mbar_init(b+i,i>=4?2:1);';assert body.count(marker)==1
        body=body.replace(marker,'for(int i=0;i<=3*NSLOT;++i)mbar_init(b+i,(i>=2*NSLOT && i<3*NSLOT)?2:1);')
        if pending:
            marker='''});wgmma_commit();wgmma_wait<0>();
  named_bar_sync(2+WG,128);if(tid==0)mbar_arrive(b+2*NSLOT+slot);
 }''';assert body.count(marker)==1
            body=body.replace(marker,''' });wgmma_commit();
  if(step>=1){wgmma_wait<1>();named_bar_sync(2+WG,128);if(tid==0)mbar_arrive(b+2*NSLOT+(step-1)%NSLOT);}
 }
 wgmma_wait<0>();named_bar_sync(2+WG,128);if(tid==0)mbar_arrive(b+2*NSLOT+7%NSLOT);''')
        body=body.replace('mw_d256_pipe_gate_dn','mw_d256_compact_pipe_gate_dn');self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0','-DTMN_SIGMOID_TANH=1','-DPRODUCER_REGS=64','-DCONSUMER_REGS=192',f'-DGATE_SLOTS={slots}']
        cubin=T.compile_text(body,flags);self.cubin,self.pool_metadata=initial_pool_three(cubin,'mw_d256_compact_pipe_gate_dn',152,64,192)
        self.k=T.load_unit(str(self.cubin),'mw_d256_compact_pipe_gate_dn').kernel('mw_d256_compact_pipe_gate_dn');self.smem=53248*slots+128;self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();fields=plan.prefix_gate.params.fields.copy()
        tm=lambda t,rows:L.tensor_map(t,[32,64],dims=[256,rows],strides_bytes=[512],swizzle='64B',l2='128B')
        proj=plan.schedule.args['proj'][2];gate=plan.schedule.args['gate'][2]
        maps=[tm(proj,p.M),tm(gate,p.M),tm(p.dy,p.M),tm(p.ds,p.n),tm(p.tensors[7],p.M),L.tensor_map(plan.dx.input[:256],[64,32],dims=[p.M,256],strides_bytes=[p.M*2],swizzle='128B',l2='128B')]
        wp=L.tensor_map(plan.leaves[6],[64,32,8],dims=[64,256,8],strides_bytes=[1024,128],swizzle='128B',l2='256B')
        dn=L.tensor_map(p.tensors[9],[64,64,4],dims=[64,p.M,8],strides_bytes=[1024,128],swizzle='128B',l2='128B')
        self.params=L.Struct([*maps,wp,dn,*fields[6:]]);self.grid=p.M//64
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,384,self.smem)))

    def __call__(self):self.k.launch((self.grid,1,1),(384,1,1),[self.params],self.smem)
    def launch(self,*args,**kwargs):self()
