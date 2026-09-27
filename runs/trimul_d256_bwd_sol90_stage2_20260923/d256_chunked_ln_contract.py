"""Pipeline row-local dL products after each complete matrix-row LN chunk."""
import re
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class ChunkedLNContract:
    def __init__(self,plan,overlap=True):
        p=plan.p;assert p.D==256 and p.n==384
        self.plan=plan;self.overlap=overlap
        self.original_ln=plan.b1.ln;self.original_run=plan.schedule.run
        L=T._launch_module();root=T._upstream()/'csrc'
        body=self.original_ln.source_text
        old='float *dg,*db;int M;};'
        assert body.count(old)==1
        body=body.replace(old,'float *dg,*db;int M,begin,end;};')
        old='for(int row=blockIdx.x*ROWS;row<p.M;row+=gridDim.x*ROWS)'
        assert body.count(old)==1
        body=body.replace(old,'for(int row=p.begin+blockIdx.x*ROWS;row<p.end;row+=gridDim.x*ROWS)')
        name=re.search(r'void (mw_\w+)\(__grid_constant__ const Params p\)',body).group(1)
        body=body.replace(name,'mw_chunked_output_ln')
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(root),'-DMW_MINB=1','-DMW_K1_STREAM=0','-DWIDTH=256','-DLN_ROWS=32','-DLN_DN_TMA=1','-DLN_FENCE=0','-DLN_MINBLOCKS=3']
        self.ln_cubin=T.compile_text(body,flags)
        self.ln_kernel=T.load_unit(str(self.ln_cubin),'mw_chunked_output_ln').kernel('mw_chunked_output_ln')
        self.ln_smem=self.original_ln.smem;self.ln_grid=self.original_ln.grid
        self.ln_kernel.set_max_dynamic_smem(self.ln_smem)
        self.ln_params=[L.Struct([*self.original_ln.params.fields,i*128*384,(i+1)*128*384]) for i in range(3)]

        original=plan.native_contract
        body=original.source_text
        old='struct Params{CUtensorMap a[4],b[4],out[2];};'
        assert body.count(old)==1
        body=body.replace(old,'struct Params{CUtensorMap a[4],b[4],out[2];int phase,chunk;};')
        start=body.index(' constexpr int MT=N/(64*ROW_GROUPS),TILES=MT*3;')
        end=body.index(' if(threadIdx.x==0){for(int i=0;i<SLOTS;',start)
        body=body[:start]+'''
 int count=p.phase==0?3:9,rem=blockIdx.x/2;
 int ch=rem/count,tile=rem%count,mode=(blockIdx.x%2)*2+p.phase;
 int mi,ni;
 if(p.phase==0){mi=mode==0?p.chunk*128:tile*128;ni=mode==0?tile*128:p.chunk*128;}
 else{mi=(tile/3)*128;ni=(tile%3)*128;}
'''+body[end:]
        body=body.replace('mw_d256_whole_fixed_contract','mw_partial_fixed_contract')
        self.contract_source=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(root),'-DROW_GROUPS=2','-DGRID_ORDER=1','-DMIN_BLOCKS=2','-DCONTRACT_SLOTS=3']
        self.contract_cubin=T.compile_text(body,flags)
        self.contract_kernel=T.load_unit(str(self.contract_cubin),'mw_partial_fixed_contract').kernel('mw_partial_fixed_contract')
        self.contract_smem=original.smem;self.contract_kernel.set_max_dynamic_smem(self.contract_smem)
        self.contract_params=[L.Struct([*original.params.fields,0,i]) for i in range(3)]
        self.odd_params=L.Struct([*original.params.fields,1,0])
        self.resources={}
        for name,k,threads,smem in [('ln',self.ln_kernel,128,self.ln_smem),('contract',self.contract_kernel,256,self.contract_smem)]:
            drv=k.unit.drv;fn=drv.d.CUfunction(int(k.handle))
            query=lambda key:int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(getattr(drv.d.CUfunction_attribute,key),fn)))
            self.resources[name]=dict(registers=query('CU_FUNC_ATTRIBUTE_NUM_REGS'),local_bytes=query('CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES'),occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,threads,smem))))
        self.side=torch.cuda.Stream(device=p.x.device)
        self.ready=[torch.cuda.Event() for _ in range(3)];self.done=torch.cuda.Event()

    def even(self,i):
        self.contract_kernel.launch((256*6,1,1),(256,1,1),[self.contract_params[i]],self.contract_smem)

    def ln(self):
        main=torch.cuda.current_stream()
        for i in range(3):
            self.ln_kernel.launch((self.ln_grid,1,1),(128,1,1),[self.ln_params[i]],self.ln_smem)
            if self.overlap:
                self.ready[i].record(main)
                with torch.cuda.stream(self.side):
                    self.side.wait_event(self.ready[i]);self.even(i)
        if self.overlap:self.done.record(self.side)

    def run(self,name):
        if name=='bc0':
            if not self.overlap:
                for i in range(3):self.even(i)
            self.contract_kernel.launch((256*18,1,1),(256,1,1),[self.odd_params],self.contract_smem)
        elif name=='bc3':
            if self.overlap:torch.cuda.current_stream().wait_event(self.done)
        elif name not in ('bc1','bc2'):
            self.original_run(name)

    def select(self,enabled):
        self.plan.b1.ln=self.ln if enabled else self.original_ln
        self.plan.schedule.run=self.run if enabled else self.original_run
