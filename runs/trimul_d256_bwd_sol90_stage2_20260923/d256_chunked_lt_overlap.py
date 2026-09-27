"""Pipeline complete spatial source chunks with ordered full-K Lt dX."""
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_mask_transform import mask_stage
from wide_cached_input import CachedInput
from lt_contract import LtBmm


class ChunkedLtOverlap:
    def __init__(self,plan,splits=32,chunks=4,overlap=True):
        p=plan.p;prior=plan.b7
        assert p.D==256 and splits in (8,16,32) and splits%chunks==0
        self.plan=plan;self.splits=splits;self.chunks=chunks;self.overlap=overlap
        self.stream=torch.cuda.Stream(device=p.x.device)
        self.ready=[torch.cuda.Event() for _ in range(chunks)]
        self.done=torch.cuda.Event()
        body=mask_stage(prior.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
        assert body.count('int M;};')==1 and body.count('split=blockIdx.x/32')==2
        body=body.replace('int M;};','int M;int split_offset;};')
        body=body.replace('split=blockIdx.x/32','split=blockIdx.x/32+p.split_offset')
        body=body.replace('mw_d256_b7_tma','mw_d256_chunked_lt_source')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v',
               '-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={splits}']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_d256_chunked_lt_source').kernel('mw_d256_chunked_lt_source')
        self.smem=prior.smem;self.k.set_max_dynamic_smem(self.smem)
        self.params=[];self.dx=[];self.ranges=[]
        # This workspace is owned only by the consumer stream.
        self.workspace=torch.empty_like(plan.workspace)
        L=T._launch_module();tiles=p.M//64
        algorithm=list(plan.dx_lt.heuristics[plan.dx_lt.index].algo.data)
        for i in range(chunks):
            split=i*(splits//chunks)
            begin=(tiles*split//splits)*64
            end=(tiles*(split+splits//chunks)//splits)*64
            self.ranges.append((begin,end))
            self.params.append(L.Struct([*prior.params.fields,split]))
            op=LtBmm(plan.dx.input[:,begin:end].t().unsqueeze(0),plan.dx.weights.unsqueeze(0),
                     p.tensors[10][begin:end].unsqueeze(0),self.workspace)
            for j in op.indices:
                if list(op.heuristics[j].algo.data)==algorithm:
                    op.index=j;break
            else:
                # Exact qualified K ordering; the launch rejects unsupported shapes.
                op.index=0
                for j,v in enumerate(algorithm):op.heuristics[0].algo.data[j]=v
            self.dx.append(op)
        self.reduce=CachedInput(p,16,128,4,splits=splits)
        self.pack_weights=plan.dx.pack_weights
        drv=self.k.unit.drv;fn=drv.d.CUfunction(int(self.k.handle))
        query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
        self.registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS');self.local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES')

    def __call__(self):
        self.pack_weights()
        main=torch.cuda.current_stream()
        for i in range(self.chunks):
            self.k.launch((32*(self.splits//self.chunks),1,1),(256,1,1),[self.params[i]],self.smem)
            if self.overlap:
                self.ready[i].record(main)
                with torch.cuda.stream(self.stream):
                    self.stream.wait_event(self.ready[i]);self.dx[i]()
            else:self.dx[i]()
        if self.overlap:
            self.done.record(self.stream);main.wait_event(self.done)
