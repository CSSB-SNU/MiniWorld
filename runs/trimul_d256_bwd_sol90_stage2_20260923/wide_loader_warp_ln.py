"""A dedicated fifth warp issues whole-tensor TMA and statistics for output LN."""
from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class LoaderWarpLN:
    def __init__(self,p,rows=16,minblocks=2,consumer_regs=0):
        root=Path(__file__).resolve().parent;h=2*p.D;self.p=p;self.threads=256 if consumer_regs else 160
        helpers=(root/'dn_slim.cu').read_text().split('TMN_DEVI uint32_t rawpos')[1].split('TMN_DEVI void producer')[0]
        helpers=('TMN_DEVI uint32_t rawpos'+helpers).replace('kc<8','kc<H/64')
        if rows==8:helpers=(root/'tile_transpose8.cuh').read_text()
        affine=(root/'ln_aggregate.cuh').read_text().replace('__syncthreads();','named_bar_sync(1,128);')
        body=(root/'wide_prefetch_ln.cu').read_text().replace('// TRANSPOSE_HELPERS',helpers).replace('// AFFINE_HELPER',affine)
        body=body.replace('SB=ROWS*H*2;','SB=ROWS*H*2,STATS=4*SB+128+H*4;')
        begin=body.index('TMN_DEVI void prefetch(');end=body.index('extern "C" __global__',begin)
        body=body[:begin]+'''TMN_DEVI void prefetch(const Params& p,uint8_t* storage,uint64_t* bar,int row,int slot){
 uint8_t* dst=storage+slot*2*SB;uint8_t* stats=storage+STATS+slot*ROWS*8;
 mbar_arrive_expect_tx(bar+slot,2*SB+ROWS*8);
 tma_load_3d(dst,&p.tri,bar+slot,row,0,0);tma_load_3d(dst+SB,&p.dnmap,bar+slot,0,row,0);
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(stats)),"l"(p.mu+row),"n"(ROWS*4),"r"(smem_u32(bar+slot)):"memory");
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(stats+ROWS*4)),"l"(p.rs+row),"n"(ROWS*4),"r"(smem_u32(bar+slot)):"memory");
}
'''+body[end:]
        prefix=body[:body.index('extern "C" __global__')]
        a=body.index(' float gg[H/32]');consumer=body[a:body.rfind('}')]
        consumer=consumer.replace('int phase=0;','')
        line=' if(tid==0 && blockIdx.x*ROWS<p.M)prefetch(p,storage,bar,blockIdx.x*ROWS);\n'
        assert consumer.count(line)==1;consumer=consumer.replace(line,'')
        a=consumer.index('  if(tid==0 && row+gridDim.x*ROWS<p.M){');b=consumer.index('  transpose32<false>',a)
        consumer=consumer[:a]+consumer[b:]
        consumer=consumer.replace('p.mu[row+r]','reinterpret_cast<const float*>(storage+STATS+slot*ROWS*8)[r]')
        consumer=consumer.replace('p.rs[row+r]','reinterpret_cast<const float*>(storage+STATS+slot*ROWS*8)[ROWS+r]')
        consumer=consumer.replace('for(int c=0;c<H;c+=64)put_tile(&p.dt,sm+(c/64)*(ROWS*128),row,c);tma_store_commit();',
                                    'put_tile(&p.dt,sm,row,0);tma_store_commit();tma_store_wait_read<0>();')
        marker='  __syncthreads();\n }\n if(tid==0)tma_store_wait_all();'
        assert consumer.count(marker)==1
        consumer=consumer.replace(marker,'  __syncthreads();if(tid==0)mbar_arrive(bar+2+slot);\n }\n if(tid==0)tma_store_wait_all();')
        consumer=consumer.replace('__syncthreads();','named_bar_sync(1,128);')
        body=prefix+'''TMN_DEVI void consume(const Params& p,uint8_t* storage,uint64_t* bar){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
'''+consumer+'''
}
extern "C" __global__ __launch_bounds__(160,LN_MINBLOCKS)
void mw_loader_warp_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t storage[];
 auto bar=reinterpret_cast<uint64_t*>(storage+4*SB);int tid=threadIdx.x;
 for(int c=blockIdx.x*160+tid;c<H;c+=gridDim.x*160){p.dg[c]=0;p.db[c]=0;}
 for(int c=tid;c<H;c+=160)reinterpret_cast<float*>(storage+4*SB+128)[c]=p.gamma[c];
 if(tid==0){for(int i=0;i<4;++i)mbar_init(bar+i,1);fence_barrier_init();}
 __syncthreads();cooperative_groups::this_grid().sync();
 if(tid<128)consume(p,storage,bar);
 else if(tid==128){
  for(int row=blockIdx.x*ROWS,it=0;row<p.M;row+=gridDim.x*ROWS,++it){
   int slot=it%2;
   if(it>=2)mbar_wait(bar+2+slot,((it/2)-1)&1);
   prefetch(p,storage,bar,row,slot);
  }
 }
}
'''
        body=body.replace('tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}]','tensor.3d.global.shared::cta.bulk_group [%0,{%2,%3,0}]')
        if consumer_regs:
            body=body.replace('__launch_bounds__(160,LN_MINBLOCKS)','__launch_bounds__(256,LN_MINBLOCKS)')
            body=body.replace('blockIdx.x*160+tid','blockIdx.x*256+tid').replace('gridDim.x*160','gridDim.x*256').replace('c+=160','c+=256')
            body=body.replace('if(tid<128)consume(p,storage,bar);',f'if(tid<128){{setmaxnreg_inc<{consumer_regs}>();consume(p,storage,bar);}}')
            body=body.replace('else if(tid==128){','else {setmaxnreg_dec<32>();if(tid==128){')
            body=body.rstrip()[:-1]+'}}\n'
        self.source_text=body;self.smem=4*rows*h*2+128+h*4+2*rows*8
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}',f'-DLN_ROWS={rows}','-DLN_DN_TMA=1','-DLN_FENCE=0',f'-DLN_MINBLOCKS={minblocks}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_loader_warp_ln').kernel('mw_loader_warp_ln');self.k.set_max_dynamic_smem(self.smem)
        L=T._launch_module();tm=lambda t:L.tensor_map(t,[rows,64,h//64],dims=[p.M,64,h//64],strides_bytes=[p.M*2,p.M*128],swizzle='none' if rows==8 else '32B' if rows==16 else '64B',l2='128B')
        dn=L.tensor_map(p.tensors[9],[64,rows,h//64],dims=[64,p.M,h//64],strides_bytes=[h*2,128],swizzle='128B',l2='128B')
        self.params=L.Struct([tm(p.tri),tm(p.dt),dn,p.tensors[9],p.floats[5],p.floats[6],p.floats[2],p.floats[10],p.floats[11],p.M])
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,self.threads,self.smem)))
        assert self.occupancy>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
    def __call__(self):
        L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
