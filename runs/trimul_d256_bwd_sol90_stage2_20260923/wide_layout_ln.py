"""Output-LN layout/liveness experiments; preserve row reduction order."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class LayoutLN:
    def __init__(self,p,original,threads=256,mode='direct'):
        root=Path(__file__).resolve().parent;h=2*p.D
        self.params=original.lp;self.threads=threads;self.smem=128*h+128
        assert mode in ('direct','direct_compact','transpose_compact')
        body=(root/'ln.cu').read_text()
        if mode.startswith('direct'):
            # Keep the channel-major TMA tile throughout. The row warp reads
            # columns directly; the final store uses the original TMA layout.
            body=body.replace('transpose(sm);','')
            body=body.replace('swz128(r,(c%64)*2)','swz128(c%64,r*2)')
        if mode.endswith('compact'):
            body=body.replace(',z[16],v[16],s0=0,s1=0;',',s0=0,s1=0;')
            body=body.replace('z[q]=','float z=').replace('v[q]=','float v=')
            body=body.replace('z[q]','z').replace('v[q]','v')
            index='swz128(c%64,r*2)' if mode.startswith('direct') else 'swz128(r,(c%64)*2)'
            marker='for(int q=0;q<16;++q){int c=lane+q*32;float dy='
            assert body.count(marker)==1
            body=body.replace(marker,'for(int q=0;q<16;++q){int c=lane+q*32;float z=(__bfloat162float(reinterpret_cast<bf*>(sm+(c/64)*8192)['+index+'/2])-mu)*rs;float dy=')
            # Static q retains register affine accumulators. Compiler fences
            # keep the entire row's addresses and loads from being hoisted.
            body=body.replace('bb[q]+=dy;','bb[q]+=dy;asm volatile("":::"memory");')
            body=body.replace('centered)*rs);}','centered)*rs);asm volatile("":::"memory");}')
        body=body.replace('H=512',f'H={h}').replace('SB=65536',f'SB={128*h}')
        body=body.replace('NT=256',f'NT={threads}').replace('kc<8','kc<H/64')
        body=body.replace('kc+=2','kc+=NT/128').replace('r+=8','r+=NT/32')
        body=body.replace('[16]','[H/32]').replace('q<16','q<H/32')
        body=body.replace('__launch_bounds__(NT,2)','__launch_bounds__(NT,1)')
        body=body.replace('mw_d256_ln','mw_wide_layout_ln')
        body=body.replace('extern "C" __global__',(root/'ln_aggregate.cuh').read_text()+'\nextern "C" __global__')
        marker=' for(int q=0;q<H/32;++q){atomicAdd(p.dg+lane+q*32,gg[q]);atomicAdd(p.db+lane+q*32,bb[q]);}'
        assert body.count(marker)==1
        body=body.replace(marker,' aggregate_ln<H,NT>(gg,bb,reinterpret_cast<float*>(sm),p.dg,p.db);')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_wide_layout_ln').kernel('mw_wide_layout_ln')
        self.kernel.set_max_dynamic_smem(self.smem);drv=self.kernel.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.kernel.handle)),threads,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ

    def __call__(self):
        L=T._launch_module();drv=self.kernel.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.kernel.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
