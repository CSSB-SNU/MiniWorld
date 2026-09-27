"""Prefetch LN mean and inverse standard deviation with existing TMA inputs."""
from pathlib import Path
import ctypes,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class StatsLN:
    def __init__(self,p,original):
        self.__dict__.update(original.__dict__)
        root=Path(__file__).resolve().parent;kind=type(original).__name__;h=2*p.D
        assert kind in ('AffineOutputLN','PrefetchLN','CachedLN'),kind
        rows=32 if kind=='CachedLN' and p.D==384 else 16
        self.threads=256 if kind=='AffineOutputLN' else 128
        helpers=(root/'dn_slim.cu').read_text().split('TMN_DEVI uint32_t rawpos')[1].split('TMN_DEVI void producer')[0]
        helpers=('TMN_DEVI uint32_t rawpos'+helpers).replace('kc<8','kc<H/64')
        filename={'AffineOutputLN':'wide_affine_output_ln.cu','PrefetchLN':'wide_prefetch_ln.cu','CachedLN':'wide_tile_ln.cu'}[kind]
        body=(root/filename).read_text().replace('// TRANSPOSE_HELPERS',helpers).replace('// AFFINE_HELPER',(root/'ln_aggregate.cuh').read_text())
        slots=2 if kind=='PrefetchLN' else 1
        buffers=3 if kind=='AffineOutputLN' else 4 if kind=='PrefetchLN' else 2
        stats=buffers*rows*h*2+128+h*4
        self.smem=stats+slots*rows*8
        pos=body.index('TMN_DEVI float sumwarp')
        body=body[:pos]+f'constexpr int STATS={stats};\n'+'''TMN_DEVI void load_stats(const Params& p,uint8_t* dst,uint64_t* bar,int row){
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(dst)),"l"(p.mu+row),"n"(ROWS*4),"r"(smem_u32(bar)):"memory");
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(dst+ROWS*4)),"l"(p.rs+row),"n"(ROWS*4),"r"(smem_u32(bar)):"memory");
}
'''+body[pos:]
        if kind=='AffineOutputLN':
            body=body.replace('mbar_arrive_expect_tx(bar+slot,2*SB);','mbar_arrive_expect_tx(bar+slot,2*SB+ROWS*8);load_stats(p,sm+STATS,bar+slot,row);')
            base='sm+STATS'
            name='mw_wide_affine_output_ln'
        elif kind=='PrefetchLN':
            body=body.replace('uint8_t* dst,uint64_t* bar,int row){\n mbar_arrive_expect_tx(bar,2*SB);','uint8_t* dst,uint8_t* stats,uint64_t* bar,int row){\n mbar_arrive_expect_tx(bar,2*SB+ROWS*8);load_stats(p,stats,bar,row);')
            body=body.replace('prefetch(p,storage,bar,','prefetch(p,storage,storage+STATS,bar,')
            body=body.replace('prefetch(p,storage+(1-slot)*2*SB,bar+1-slot,','prefetch(p,storage+(1-slot)*2*SB,storage+STATS+(1-slot)*ROWS*8,bar+1-slot,')
            base='storage+STATS+slot*ROWS*8'
            name='mw_wide_prefetch_ln'
        else:
            body=body.replace('mbar_arrive_expect_tx(bar,SB*(1+LN_DN_TMA));','mbar_arrive_expect_tx(bar,SB*(1+LN_DN_TMA)+ROWS*8);load_stats(p,sm+STATS,bar,row);')
            body=body.replace('p.gamma[c]','reinterpret_cast<float*>(sm+SB*(1+LN_DN_TMA)+128)[c]')
            marker=' __syncthreads();cooperative_groups::this_grid().sync();'
            body=body.replace(marker,' for(int c=tid;c<H;c+=NT)reinterpret_cast<float*>(sm+SB*(1+LN_DN_TMA)+128)[c]=p.gamma[c];\n'+marker)
            base='sm+STATS'
            name='mw_wide_tile_ln'
        body=body.replace('p.mu[row+r]',f'reinterpret_cast<const float*>({base})[r]')
        body=body.replace('p.rs[row+r]',f'reinterpret_cast<const float*>({base})[ROWS+r]')
        body=body.replace(name,'mw_wide_stats_ln')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={p.D}',f'-DLN_ROWS={rows}','-DLN_DN_TMA=1','-DLN_FENCE=0','-DLN_MINBLOCKS=2']
        self.cubin=T.compile_text(body,flags);self.kernel=T.load_unit(str(self.cubin),'mw_wide_stats_ln').kernel('mw_wide_stats_ln');self.kernel.set_max_dynamic_smem(self.smem)
        drv=self.kernel.unit.drv;function=drv.d.CUfunction(int(self.kernel.handle))
        self.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(function,self.threads,self.smem)))
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*self.occupancy
        self.kind=kind;self.rows=rows
        query=lambda n:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,n),function)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')
    def __call__(self):
        L=T._launch_module();drv=self.kernel.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.kernel.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
