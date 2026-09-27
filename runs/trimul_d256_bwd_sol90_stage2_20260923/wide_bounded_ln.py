"""Output LN backward with channel values recomputed instead of kept live."""
from pathlib import Path
import ctypes,os
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F


class BoundedLN:
    def __init__(self,p,original,threads=256):
        root=Path(__file__).resolve().parent;d=p.D;h=2*d
        self.params=original.lp;self.threads=threads;self.smem=128*h+128
        body=(root/'ln.cu').read_text()
        body=body.replace(',z[16],v[16],s0=0,s1=0;',',s0=0,s1=0;')
        body=body.replace('z[q]=','float z=').replace('v[q]=','float v=')
        body=body.replace('z[q]','z').replace('v[q]','v')
        marker='for(int q=0;q<16;++q){int c=lane+q*32;float dy='
        assert body.count(marker)==1
        body=body.replace(marker,'for(int q=0;q<16;++q){int c=lane+q*32;float z=(__bfloat162float(reinterpret_cast<bf*>(sm+(c/64)*8192)[swz128(r,(c%64)*2)/2])-mu)*rs;float dy=')
        if os.environ.get('BOUNDED_LN_LOOP')=='1':
            # Keep the reduction loop scalar. Affine accumulators are updated
            # in the unrolled write pass, retaining their per-row sum order.
            marker='#pragma unroll\n   for(int q=0;q<16;++q){int c=lane+q*32;\n'
            assert body.count(marker)==1
            body=body.replace(marker,marker.replace('#pragma unroll','#pragma unroll 1'))
            assert body.count('gg[q]+=dy*z;bb[q]+=dy;')==1
            body=body.replace('gg[q]+=dy*z;bb[q]+=dy;','')
            body=body.replace('float centered=','gg[q]+=dy*z;bb[q]+=dy;float centered=')
        body=body.replace('H=512',f'H={h}').replace('SB=65536',f'SB={128*h}')
        body=body.replace('NT=256',f'NT={threads}').replace('kc<8','kc<H/64')
        body=body.replace('kc+=2','kc+=NT/128').replace('r+=8','r+=NT/32')
        body=body.replace('[16]','[H/32]').replace('q<16','q<H/32')
        body=body.replace('__launch_bounds__(NT,2)','__launch_bounds__(NT,1)')
        body=body.replace('mw_d256_ln','mw_wide_bounded_ln')
        body=body.replace('extern "C" __global__',(root/'ln_aggregate.cuh').read_text()+'\nextern "C" __global__')
        marker=' for(int q=0;q<H/32;++q){atomicAdd(p.dg+lane+q*32,gg[q]);atomicAdd(p.db+lane+q*32,bb[q]);}'
        assert body.count(marker)==1
        body=body.replace(marker,' aggregate_ln<H,NT>(gg,bb,reinterpret_cast<float*>(sm),p.dg,p.db);')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0']
        self.cubin=T.compile_text(body,flags)
        self.kernel=T.load_unit(str(self.cubin),'mw_wide_bounded_ln').kernel('mw_wide_bounded_ln')
        self.kernel.set_max_dynamic_smem(self.smem);drv=self.kernel.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.kernel.handle)),threads,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ

    def __call__(self):
        L=T._launch_module();drv=self.kernel.unit.drv;args=L._Packed([self.params])
        drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.kernel.handle)),self.grid,1,1,self.threads,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))


class B1WithLN:
    def __init__(self,original,ln):self.original=original;self.ln=ln
    def __call__(self):
        b=self.original;p=b.p
        if not b.use_saved_norm:b.norm.launch((b.grid,1,1),(b.threads,1,1),[b.np],b.smem)
        torch.mm(p.tensors[6],b.wp.t(),out=b.proj)
        torch.mm(p.xn.reshape(p.M,p.D),b.wg.t(),out=b.gate)
        b.epi.launch((1056,1,1),(256,1,1),[b.ep],0)
        torch.mm(p.tensors[7],b.wp,out=p.tensors[9])
        if b.exact_dwp is None:torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp)
        else:b.exact_dwp()
        torch.mm(p.dg.t(),p.xn.reshape(p.M,p.D),out=p.dwg)
        self.ln()
