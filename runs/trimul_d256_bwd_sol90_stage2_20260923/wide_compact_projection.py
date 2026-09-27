"""Exact fixed-order Lt projection of compacted changed rows, with GPU fallback."""
from pathlib import Path
import ctypes
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from lt_contract import LtBmm,Algorithm

class CompactProjection:
    def __init__(self,plan,delta,capacity=None):
        p=plan.p;d=p.D;h=2*d;root=Path(__file__).resolve().parent
        self.capacity=(p.M//64) if capacity is None else capacity
        assert self.capacity%64==0
        self.seen_count=torch.empty(p.M+1,device=p.x.device,dtype=torch.int32)
        self.count=self.seen_count[-1:]
        self.rows=torch.empty(self.capacity,device=p.x.device,dtype=torch.int32)
        self.compact=p.x.new_empty((self.capacity,h));self.projected=p.x.new_empty((self.capacity,d))
        self.workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
        self.mm=LtBmm(self.compact.unsqueeze(0),plan.b1.wp.t().unsqueeze(0),self.projected.unsqueeze(0),self.workspace)
        old=plan.schedule.kernels['proj']
        ctypes.memmove(ctypes.byref(self.mm.heuristics[0].algo),ctypes.byref(old.heuristics[old.index].algo),ctypes.sizeof(Algorithm))
        self.mm.index=0
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DMW_MINB=1','-DMW_K1_STREAM=0',f'-DWIDTH={d}']
        self.cubin=T.compile(root/'wide_compact_projection.cu',flags)
        unit=T.load_unit(str(self.cubin),'mw_wide_compact_changed_rows')
        self.pack=unit.kernel('mw_wide_compact_changed_rows');self.gather=unit.kernel('mw_wide_gather_changed_norm');self.scatter=unit.kernel('mw_wide_scatter_changed_proj')
        L=T._launch_module()
        self.params=L.Struct([delta.patches,delta.count,delta.capacity,self.seen_count,self.count,self.rows,self.capacity,p.tensors[6],self.compact,self.projected,plan.b1.proj])
        self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*4
        body=(root/'wide_flagged_projection.cu').read_text().replace('// MMA_HELPERS',(root/'mma_offset.cuh').read_text())
        body=body.replace('int M;};','int M;const unsigned* patch_count;unsigned patch_capacity;const unsigned* row_count;unsigned row_capacity;};')
        body=body.replace(' int row=blockIdx.x*64,col=blockIdx.y*128;', ' if(*p.patch_count<=p.patch_capacity&&*p.row_count<=p.row_capacity)return;\n int row=blockIdx.x*64,col=blockIdx.y*128;')
        body=body.replace('mw_wide_flagged_projection','mw_wide_compact_projection_fallback')
        self.fallback_cubin=T.compile_text(body,flags)
        self.fallback=T.load_unit(str(self.fallback_cubin),'mw_wide_compact_projection_fallback').kernel('mw_wide_compact_projection_fallback')
        self.smem=49152+128;self.fallback.set_max_dynamic_smem(self.smem)
        def tm(t,cols,rows):return L.tensor_map(t,[64,64],dims=[cols,rows],strides_bytes=[cols*2],swizzle='128B',l2='128B')
        self.fp=L.Struct([tm(p.tensors[6],h,p.M),tm(plan.b1.wp,h,d),tm(plan.b1.proj,d,p.M),delta.changed,p.M,delta.count,delta.capacity,self.count,self.capacity])
        self.fgrid=(p.M//64,d//128,1)
    def __call__(self):
        self.seen_count.zero_()
        self.pack.launch((self.grid,1,1),(256,1,1),[self.params],0)
        self.gather.launch((self.grid,1,1),(256,1,1),[self.params],0)
        self.mm()
        self.scatter.launch((self.grid,1,1),(256,1,1),[self.params],0)
        self.fallback.launch(self.fgrid,(128,1,1),[self.fp],self.smem)
