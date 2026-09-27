"""Two disjoint SM resources through the installed CUDA12.9 driver ABI."""
import ctypes as C
import torch

class SM(C.Structure):
    _fields_=[('smCount',C.c_uint)]
class ResourceData(C.Union):
    _fields_=[('sm',SM),('_oversize',C.c_ubyte*48)]
class Resource(C.Structure):
    _anonymous_=('data',)
    _fields_=[('type',C.c_int),('_internal_padding',C.c_ubyte*92),('data',ResourceData)]

class GreenPartition:
    def __init__(self,small_sms,split_flags=0):
        assert C.sizeof(Resource)==144
        self.lib=C.CDLL('libcuda.so.1');P=C.c_void_p;R=C.POINTER(Resource);U=C.c_uint
        for name,args in {
            'cuDeviceGetDevResource':[C.c_int,R,C.c_int],
            'cuDevSmResourceSplitByCount':[R,C.POINTER(U),R,R,U,U],
            'cuDevResourceGenerateDesc':[C.POINTER(P),R,U],
            'cuGreenCtxCreate':[C.POINTER(P),P,C.c_int,U],
            'cuGreenCtxStreamCreate':[C.POINTER(P),P,U,C.c_int],
        }.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=C.c_int
        dev=torch.cuda.current_device();whole=Resource();self.call('cuDeviceGetDevResource',dev,C.byref(whole),1)
        small=Resource();large=Resource();groups=U(1)
        self.call('cuDevSmResourceSplitByCount',C.byref(small),C.byref(groups),C.byref(whole),C.byref(large),split_flags,small_sms)
        assert groups.value==1 and small.sm.smCount and large.sm.smCount
        self.sm_counts=(small.sm.smCount,large.sm.smCount)
        self.resources=(small,large);self.handles=[];self.streams=[]
        for resource in self.resources:
            desc=P();ctx=P();stream=P()
            self.call('cuDevResourceGenerateDesc',C.byref(desc),C.byref(resource),1)
            self.call('cuGreenCtxCreate',C.byref(ctx),desc,dev,1)
            self.call('cuGreenCtxStreamCreate',C.byref(stream),ctx,1,0)
            self.handles.append((desc,ctx,stream))
            self.streams.append(torch.cuda.ExternalStream(stream.value,device=dev))

    def call(self,name,*args):
        status=getattr(self.lib,name)(*args)
        if status:raise RuntimeError(f'{name}: CUDA driver status {status}')

    def verify_graph(self):
        from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
        body='''extern "C" __global__ void mw_green_sm_sample(int* out){
         extern __shared__ char storage[];
         if(threadIdx.x==0){unsigned smid;asm volatile("mov.u32 %0,%%smid;":"=r"(smid));storage[0]=smid;out[blockIdx.x]=smid;}
         __syncthreads();unsigned long long start=clock64();while(clock64()-start<200000)asm volatile("");
        }'''
        cubin=T.compile_text(body,['-std=c++17','-O3','-arch=sm_90a','--cubin'])
        k=T.load_unit(str(cubin),'mw_green_sm_sample').kernel('mw_green_sm_sample');k.set_max_dynamic_smem(114688)
        outputs=[torch.full((528,),-1,device='cuda',dtype=torch.int32) for _ in range(2)]
        ready=torch.cuda.Event();done=[torch.cuda.Event(),torch.cuda.Event()]
        def run():
            main=torch.cuda.current_stream();ready.record(main)
            for stream,event,out in zip(self.streams,done,outputs):
                with torch.cuda.stream(stream):
                    stream.wait_event(ready);k.launch((528,1,1),(128,1,1),[out],114688);event.record(stream)
            for event in done:main.wait_event(event)
        run();torch.cuda.synchronize()
        def observe():
            sets=[set(t.cpu().tolist()) for t in outputs]
            assert all(-1 not in s and len(s)<=n for s,n in zip(sets,self.sm_counts))
            assert not sets[0]&sets[1],sets
            return [sorted(s) for s in sets]
        eager=observe();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):run()
        for out in outputs:out.fill_(-1)
        g.replay();torch.cuda.synchronize();captured=observe()
        return dict(sm_counts=self.sm_counts,eager=eager,replay=captured,cubin=str(cubin))
