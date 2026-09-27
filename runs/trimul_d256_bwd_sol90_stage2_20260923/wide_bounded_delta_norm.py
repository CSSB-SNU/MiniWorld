"""Lossless packed/scalar normalization patches, compared before shared stores."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F

class BoundedDeltaNorm:
    def __init__(self,p,original,capacity=None,cached=True,unswitch=True,minblocks=2,rows=16,threads=128,unroll=4):
        self.__dict__.update(original.__dict__)
        assert p.D==512
        root=Path(__file__).resolve().parent;h=2*p.D;self.threads=threads
        assert rows in (16,32) and threads in (128,256)
        self.capacity=max(1,p.M*h//4096) if capacity is None else capacity
        self.patches=torch.empty(max(1,self.capacity),device=p.x.device,dtype=torch.int64)
        self.count=torch.zeros(1,device=p.x.device,dtype=torch.int32)
        self.changed=torch.zeros(p.M//64,device=p.x.device,dtype=torch.int32)
        self.smem=rows*h*2+128
        body=(root/'wide_tile_output.cu').read_text().replace('// TRANSPOSE_HELPERS',(root/'tile_transpose.cuh').read_text())
        body=body.replace('CUtensorMap tri,norm;', 'CUtensorMap tri,norm;uint64_t* patches;unsigned* count;unsigned capacity;unsigned* changed;')
        marker='   float s=0,mu,rs;'
        assert body.count(marker)==1
        body=body.replace(marker,'''   float scalar_mu,scalar_rs;
   {
    float v[H/32],ss=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){v[q]=__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,lane+q*32)]);ss+=v[q];}
    scalar_mu=sumwarp(ss)/H;ss=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){float z=v[q]-scalar_mu;ss+=z*z;}
    scalar_rs=rsqrtf(sumwarp(ss)/H+1e-5f);
    if(lane==0){p.mu[row+r]=scalar_mu;p.rs[row+r]=scalar_rs;}
   }
'''+marker)
        body=body.replace('\n   if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}','')
        marker='     *ptr=pack_bf16(fmaf((bf16lo(v)-mu)*rs,p.gamma[c],p.beta[c]),fmaf((bf16hi(v)-mu)*rs,p.gamma[c+1],p.beta[c+1]));'
        assert body.count(marker)==1
        body=body.replace(marker,'''     float g0=p.gamma[c],g1=p.gamma[c+1],b0=p.beta[c],b1=p.beta[c+1];
     uint32_t packed=pack_bf16(fmaf((bf16lo(v)-mu)*rs,g0,b0),fmaf((bf16hi(v)-mu)*rs,g1,b1));
     *ptr=packed;
     if(__float_as_uint(mu)!=__float_as_uint(scalar_mu)||__float_as_uint(rs)!=__float_as_uint(scalar_rs)){
      uint32_t scalar=pack_bf16(fmaf((bf16lo(v)-scalar_mu)*scalar_rs,g0,b0),fmaf((bf16hi(v)-scalar_mu)*scalar_rs,g1,b1));
      if(packed!=scalar){
       unsigned at=atomicAdd(p.count,1u);
       if(at<p.capacity)p.patches[at]=(uint64_t(scalar)<<32)|uint32_t(((row+r)*H+c)/2);
       atomicExch(p.changed+(row+r)/64,1u);
      }
     }''')
        if unswitch:
            marker='''    #pragma unroll
    for(int c=2*lane;c<H;c+=64){auto ptr='''
            assert body.count(marker)==1
            a=body.index(marker);b=body.index('\n   }\n  }',a)
            slow=body[a:b]
            fast='''    #pragma unroll
    for(int c=2*lane;c<H;c+=64){auto ptr=reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));uint32_t v=*ptr;
     *ptr=pack_bf16(fmaf((bf16lo(v)-mu)*rs,p.gamma[c],p.beta[c]),fmaf((bf16hi(v)-mu)*rs,p.gamma[c+1],p.beta[c+1]));
    }'''
            body=body[:a]+'''    if(__float_as_uint(mu)==__float_as_uint(scalar_mu)&&__float_as_uint(rs)==__float_as_uint(scalar_rs)){
'''+fast+'\n    }else{\n'+slow+'\n    }'+body[b:]
        if cached:
            self.smem+=h*8
            body=body.replace('p.gamma[c]', 'reinterpret_cast<float*>(sm+SB+128)[c]').replace('p.gamma[c+1]', 'reinterpret_cast<float*>(sm+SB+128)[c+1]')
            body=body.replace('p.beta[c]', 'reinterpret_cast<float*>(sm+SB+128+H*4)[c]').replace('p.beta[c+1]', 'reinterpret_cast<float*>(sm+SB+128+H*4)[c+1]')
            marker=' if(tid==0){mbar_init(bar,1);fence_barrier_init();}__syncthreads();int phase=0;'
            assert body.count(marker)==1
            body=body.replace(marker,''' for(int c=tid;c<H;c+=NT){reinterpret_cast<float*>(sm+SB+128)[c]=p.gamma[c];reinterpret_cast<float*>(sm+SB+128+H*4)[c]=p.beta[c];}
'''+marker)
        body=body.replace('float v[H/32],ss=0;', 'float ss=0;')
        body=body.replace('v[q]=__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,lane+q*32)]);ss+=v[q];', 'ss+=norm_scalar(sm,pos(r,lane+q*32));')
        body=body.replace('float z=v[q]-scalar_mu;', 'float z=norm_scalar(sm,pos(r,lane+q*32))-scalar_mu;')
        body=body.replace('uint32_t v=*reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));', 'uint32_t v=norm_word(sm,pos(r,c));')
        body=body.replace('uint32_t v=*ptr;', 'uint32_t v=norm_word(sm,pos(r,c));')
        point='extern "C" __global__ __launch_bounds__(NT,2)'
        helpers='TMN_DEVI float norm_scalar(uint8_t* sm,int pos){unsigned v;asm volatile("ld.shared.u16 %0,[%1];":"=r"(v):"r"(smem_u32(sm+pos*2)):"memory");return __uint_as_float(v<<16);}\nTMN_DEVI uint32_t norm_word(uint8_t* sm,int pos){uint32_t v;asm volatile("ld.shared.b32 %0,[%1];":"=r"(v):"r"(smem_u32(sm+pos*2)):"memory");return v;}\n'
        assert body.count(point)==1
        body=body.replace(point,helpers+point)
        assert unroll in (2,4,8)
        body=body.replace('#pragma unroll\n',f'#pragma unroll {unroll}\n')
        body=body.replace('__launch_bounds__(NT,2)',f'__launch_bounds__(NT,{minblocks})')
        body=body.replace('mw_wide_tile_norm','mw_wide_bounded_delta_norm')
        body+='''
struct Apply {const uint64_t* patches;const unsigned* count;unsigned capacity;uint32_t* norm;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_apply_delta_norm(__grid_constant__ const Apply p){
 unsigned count=*p.count;if(count>p.capacity)return;
 for(unsigned i=blockIdx.x*256+threadIdx.x;i<count;i+=gridDim.x*256){uint64_t v=p.patches[i];p.norm[uint32_t(v)]=uint32_t(v>>32);}
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0','-DWIDTH=512',f'-DOUTPUT_ROWS={rows}',f'-DOUTPUT_THREADS={threads}']
        self.cubin=T.compile_text(body,flags)
        self.native_norm=T.load_unit(str(self.cubin),'mw_wide_bounded_delta_norm').kernel('mw_wide_bounded_delta_norm');self.native_norm.set_max_dynamic_smem(self.smem)
        drv=self.native_norm.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.native_norm.handle)),threads,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module()
        tri=L.tensor_map(p.tri,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='32B' if rows==16 else '64B',l2='128B')
        norm=L.tensor_map(p.tensors[6],[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.np=L.Struct([tri,norm,self.patches,self.count,self.capacity,self.changed,p.floats[2],p.floats[3],p.floats[5],p.floats[6],p.M])
        self.norm=ResetDeltaNorm(self)
        self.apply=T.load_unit(str(self.cubin),'mw_wide_apply_delta_norm').kernel('mw_wide_apply_delta_norm')
        self.ap=L.Struct([self.patches,self.count,self.capacity,p.tensors[6]])
        fallback=(root/'wide_tile_output.cu').read_text().replace('// TRANSPOSE_HELPERS',(root/'tile_transpose.cuh').read_text())
        fallback=fallback.replace('int M;};','int M;const unsigned* count;unsigned capacity;};')
        fallback=fallback.replace('if constexpr(D<512)','if constexpr(true)')
        fallback=fallback.replace('extern __shared__ __align__(1024) uint8_t sm[];', 'if(*p.count<=p.capacity)return;\n extern __shared__ __align__(1024) uint8_t sm[];')
        fallback=fallback.replace('mw_wide_tile_norm','mw_wide_bounded_delta_norm_fallback')
        self.fallback_cubin=T.compile_text(fallback,flags)
        self.fallback=T.load_unit(str(self.fallback_cubin),'mw_wide_bounded_delta_norm_fallback').kernel('mw_wide_bounded_delta_norm_fallback')
        self.fallback_smem=rows*h*2+128;self.fallback.set_max_dynamic_smem(self.fallback_smem)
        self.fp=L.Struct([tri,norm,p.floats[2],p.floats[3],p.floats[5],p.floats[6],p.M,self.count,self.capacity])

class ResetDeltaNorm:
    def __init__(self,owner):self.owner=owner
    def launch(self,*args):
        self.owner.count.zero_();self.owner.changed.zero_();self.owner.native_norm.launch(*args)

class PatchDeltaNorm:
    def __init__(self,owner):self.owner=owner
    def launch(self,*args):
        o=self.owner
        o.apply.launch((o.grid,1,1),(256,1,1),[o.ap],0)
        o.fallback.launch((o.grid,1,1),(o.threads,1,1),[o.fp],o.fallback_smem)
