"""D512 forward emits its packed norm and the backward scalar norm together."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from lt_contract import LtBmm
from wide_split_dwp import SplitProjectionWeight

class DualNorm:
    def __init__(self,p,original):
        self.__dict__.update(original.__dict__)
        assert p.D==512
        root=Path(__file__).resolve().parent;h=2*p.D;rows=16
        self.legacy=torch.empty_like(p.tensors[6]);self.smem=rows*h*4+128
        body=(root/'wide_tile_output.cu').read_text().replace('// TRANSPOSE_HELPERS',(root/'tile_transpose.cuh').read_text())
        body=body.replace('CUtensorMap tri,norm;', 'CUtensorMap tri,norm,legacy;').replace('sm+SB);','sm+2*SB);')
        marker='   float s=0,mu,rs;'
        assert body.count(marker)==1
        body=body.replace(marker,'''   {
    float v[H/32],s=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){v[q]=__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,lane+q*32)]);s+=v[q];}
    float mu=sumwarp(s)/H;s=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){float z=v[q]-mu;s+=z*z;}
    float rs=rsqrtf(sumwarp(s)/H+1e-5f);
    if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}
    #pragma unroll
    for(int q=0;q<H/32;++q){int c=lane+q*32;reinterpret_cast<bf*>(sm+SB)[pos(r,c)]=__float2bfloat16_rn(fmaf((v[q]-mu)*rs,p.gamma[c],p.beta[c]));}
   }
'''+marker)
        # Only the first occurrence is our scalar statistics, the later one is packed.
        point='\n   if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}'
        assert body.count(point)==1
        body=body.replace(point,'')
        marker='"l"(&p.norm),"r"(smem_u32(sm+(c/64)*(ROWS*128))),"r"(c),"r"(row):"memory");'
        assert body.count(marker)==1
        body=body.replace(marker,marker+'''
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.legacy),"r"(smem_u32(sm+SB+(c/64)*(ROWS*128))),"r"(c),"r"(row):"memory");''')
        body=body.replace('mw_wide_tile_norm','mw_wide_dual_norm')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0','-DWIDTH=512','-DOUTPUT_ROWS=16','-DOUTPUT_THREADS=128']
        self.cubin=T.compile_text(body,flags)
        self.norm=T.load_unit(str(self.cubin),'mw_wide_dual_norm').kernel('mw_wide_dual_norm');self.norm.set_max_dynamic_smem(self.smem)
        drv=self.norm.unit.drv
        occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.norm.handle)),128,self.smem)))
        assert occ>0
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
        L=T._launch_module()
        tri=L.tensor_map(p.tri,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='32B',l2='128B')
        tm=lambda t:L.tensor_map(t,[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2='128B')
        self.np=L.Struct([tri,tm(p.tensors[6]),tm(self.legacy),p.floats[2],p.floats[3],p.floats[5],p.floats[6],p.M])

class DualNormBackward:
    def __init__(self,plan,output):
        self.plan=plan;self.output=output;p=plan.p;b=plan.b1;sch=plan.schedule
        self.proj=LtBmm(output.legacy.unsqueeze(0),b.wp.t().unsqueeze(0),b.proj.unsqueeze(0),sch.workspace)
        selected=sch.kernels['proj'];self.proj.index=selected.index
        assert list(self.proj.heuristics[self.proj.index].algo.data)==list(selected.heuristics[selected.index].algo.data)
        self.split=None
        if plan.split_dwp is not None:
            prior=p.tensors[6]
            try:
                p.tensors[6]=output.legacy
                self.split=SplitProjectionWeight(p,b.exact_dwp)
            finally:p.tensors[6]=prior
    def __call__(self):
        plan=self.plan;p=plan.p;b=plan.b1;sch=plan.schedule
        self.proj();b.epi.launch((1056,1,1),(256,1,1),[b.ep],0);sch.run('dn')
        if self.split is not None:self.split()
        else:torch.mm(p.tensors[7].t(),self.output.legacy,out=p.dwp)
        sch.run('dwg');plan.ln()
        for name in ('bc0','bc1','bc2','bc3'):sch.run(name)
        plan.b7.source_only();plan.dx.copy_prefix();plan.dx.pack_weights();sch.run('dx');plan.dx.reduce_only()
        return p.outputs
