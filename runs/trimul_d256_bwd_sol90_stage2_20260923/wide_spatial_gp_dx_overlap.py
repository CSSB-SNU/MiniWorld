"""Pipeline three full-channel spatial GP bands with exact-order Lt dX."""
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from lt_contract import LtBmm


class SpatialGPDX:
    def __init__(self,plan,overlap=True,dw_overlap=True):
        p=plan.p;prior=plan.contract_gp;assert p.n==384
        self.plan=plan;self.p=p;self.overlap=overlap;self.dw_overlap=dw_overlap
        self.stream=torch.cuda.Stream(device=p.x.device)
        self.ready=[torch.cuda.Event() for _ in range(3)];self.done=torch.cuda.Event()
        self.workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
        body=prior.source_text
        marker='outmap[4];int N;';assert body.count(marker)==1
        body=body.replace(marker,'outmap[4];int N;int row_band;')
        start=body.index(' int tiles=p.N/128;int half=');end=body.index('\n int mi=',start)
        body=body[:start]+''' int tiles=p.N/128;int half=blockIdx.x/(2*D*tiles),rem=blockIdx.x%(2*D*tiles),ch=rem/(2*tiles),mode=2*half+rem%2,tile=(rem/2)%tiles;'''+body[end:]
        body=body.replace('int mi=(tile/tiles)*128,ni=(tile%tiles)*128;','int mi=p.row_band*128,ni=tile*128;')
        body=body.replace('mw_wide_staged_epilogue_gp','mw_wide_spatial_gp_dx_overlap')
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWIDTH={p.D}']
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_wide_spatial_gp_dx_overlap').kernel('mw_wide_spatial_gp_dx_overlap')
        self.smem=prior.smem;self.k.set_max_dynamic_smem(self.smem)
        self.params=[T._launch_module().Struct([*prior.params.fields,i]) for i in range(3)]
        self.dx=[];full=plan.schedule.kernels['dx'];algorithm=list(full.heuristics[full.index].algo.data)
        for i in range(3):
            begin=i*(p.M//3);end=(i+1)*(p.M//3)
            op=LtBmm(plan.dx.input[:,begin:end].t().unsqueeze(0),plan.dx.weights.unsqueeze(0),p.tensors[10][begin:end].unsqueeze(0),self.workspace)
            for index in op.indices:
                if list(op.heuristics[index].algo.data)==algorithm:op.index=index;break
            else:
                op.index=0
                for j,value in enumerate(algorithm):op.heuristics[0].algo.data[j]=value
            self.dx.append(op)
        self.pack_weights=plan.dx.pack_weights
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda name:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,name),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')

    def producer(self,i):self.k.launch((4*self.p.D*3,1,1),(256,1,1),[self.params[i]],self.smem)

    def __call__(self):
        main=torch.cuda.current_stream();self.pack_weights()
        if self.overlap:self.stream.wait_stream(main)
        for i in range(3):
            self.producer(i)
            if self.overlap:
                self.ready[i].record(main)
                with torch.cuda.stream(self.stream):
                    self.stream.wait_event(self.ready[i]);self.dx[i]()
            else:self.dx[i]()
        if self.overlap:self.done.record(self.stream)
        if self.overlap and not self.dw_overlap:main.wait_event(self.done)
        self.plan.input_dw()
        if self.overlap and self.dw_overlap:main.wait_event(self.done)
