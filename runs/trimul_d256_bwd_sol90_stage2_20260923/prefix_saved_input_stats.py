"""Reuse the forward statistics in the current prefix input-LN finish."""
from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class SavedInputReduce:
    def __init__(self,p,original,rows=16,threads=128,minblocks=4,splits=8):
        assert p.D==256 and p.input_stats.shape==(p.M,2)
        self.__dict__.update(original.__dict__)
        root=Path(__file__).resolve().parent;d=p.D
        stats=3*rows*d*2+128+d*4
        self.smem=stats+rows*8
        body=(root/'wide_tma_input.cu').read_text().replace('// AFFINE_HELPER',(root/'ln_aggregate.cuh').read_text())
        assert splits in (8,16,32)
        body=body.replace('s<32',f's<{splits}')
        body=body.replace('bf* dw[4];int M;','bf* dw[4];int M;const float* stats;')
        body=body.replace('p.gamma[c]','reinterpret_cast<float*>(sm+3*SB+128)[c]')
        marker=' __syncthreads();cooperative_groups::this_grid().sync();'
        assert body.count(marker)==1
        body=body.replace(marker,' for(int c=tid;c<D;c+=NT)reinterpret_cast<float*>(sm+3*SB+128)[c]=p.gamma[c];\n'+marker)
        marker='mbar_arrive_expect_tx(bar,3*SB);'
        assert body.count(marker)==1
        body=body.replace(marker,f'''mbar_arrive_expect_tx(bar,3*SB+ROWS*8);
   asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(sm+{stats})),"l"(p.stats+size_t(row)*2),"n"(ROWS*8),"r"(smem_u32(bar)):"memory");''')
        begin=body.index('   float xv[D/32],s=0;')
        end=body.index('   #pragma unroll\n   for(int q=0;q<D/32;++q){int c=lane+q*32;float z=',begin)
        body=body[:begin]+f'''   float xv[D/32];
   #pragma unroll
   for(int q=0;q<D/32;++q)xv[q]=rd(sm,r,lane+q*32);
   const float* stats=reinterpret_cast<const float*>(sm+{stats});
   float mu=stats[2*r],rs=stats[2*r+1],s0=0,s1=0;
'''+body[end:]
        body=body.replace('mw_wide_tma_input','mw_prefix_saved_input_stats')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),
               f'-DWIDTH={d}',f'-DINPUT_ROWS={rows}',f'-DINPUT_THREADS={threads}',f'-DINPUT_MINBLOCKS={minblocks}']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_prefix_saved_input_stats').kernel('mw_prefix_saved_input_stats')
        self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv;fn=drv.d.CUfunction(int(self.kernel.handle))
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,threads,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
        L=T._launch_module();self.params=L.Struct([*original.params.fields,p.input_stats])

    def __call__(self):
        L=T._launch_module();drv=self.kernel.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.kernel.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
