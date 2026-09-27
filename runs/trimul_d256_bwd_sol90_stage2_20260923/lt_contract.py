"""Explicit cuBLASLt strided-batch algorithm selection for contraction GEMMs.

ctypes declarations follow the installed CUDA 12.9 cublasLt.h ABI. No global
PyTorch or cuBLAS settings are changed. Plans retain descriptors/workspace.
"""
import ctypes as C
import torch


class Algorithm(C.Structure):
    _fields_=[('data',C.c_uint64*8)]


class Heuristic(C.Structure):
    _fields_=[('algo',Algorithm),('workspaceSize',C.c_size_t),('state',C.c_int),
              ('wavesCount',C.c_float),('reserved',C.c_int*4)]


class LtBmm:
    def __init__(self,a,b,out,workspace):
        self.lib=C.CDLL('libcublasLt.so.12');self.a=a;self.b=b;self.out=out;self.workspace=workspace
        P=C.c_void_p;I=C.c_int;Z=C.c_size_t
        declarations={
            'cublasLtCreate':[C.POINTER(P)],
            'cublasLtMatmulDescCreate':[C.POINTER(P),I,I],
            'cublasLtMatrixLayoutCreate':[C.POINTER(P),I,C.c_uint64,C.c_uint64,C.c_int64],
            'cublasLtMatrixLayoutSetAttribute':[P,I,P,Z],
            'cublasLtMatmulPreferenceCreate':[C.POINTER(P)],
            'cublasLtMatmulPreferenceSetAttribute':[P,I,P,Z],
            'cublasLtMatmulAlgoGetHeuristic':[P,P,P,P,P,P,P,I,C.POINTER(Heuristic),C.POINTER(I)],
            'cublasLtMatmul':[P,P,P,P,P,P,P,P,P,P,P,P,C.POINTER(Algorithm),P,Z,P],
        }
        for name,args in declarations.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=I
        self.handle=P();self.check(self.lib.cublasLtCreate(C.byref(self.handle)))
        self.desc=P();self.check(self.lib.cublasLtMatmulDescCreate(C.byref(self.desc),68,0))
        self.layouts=[]
        for t in (a,b,out):
            assert t.ndim==3 and t.dtype in (torch.bfloat16,torch.float32)
            row=t.stride(2)==1;assert row or t.stride(1)==1
            layout=P();ld=t.stride(1) if row else t.stride(2)
            dtype=14 if t.dtype==torch.bfloat16 else 0
            self.check(self.lib.cublasLtMatrixLayoutCreate(C.byref(layout),dtype,t.shape[1],t.shape[2],ld))
            for attr,value in ((1,I(1 if row else 0)),(5,I(t.shape[0])),(6,C.c_int64(t.stride(0)))):
                self.check(self.lib.cublasLtMatrixLayoutSetAttribute(layout,attr,C.byref(value),C.sizeof(value)))
            self.layouts.append(layout)
        self.pref=P();self.check(self.lib.cublasLtMatmulPreferenceCreate(C.byref(self.pref)))
        size=Z(workspace.numel());self.check(self.lib.cublasLtMatmulPreferenceSetAttribute(self.pref,1,C.byref(size),C.sizeof(size)))
        self.heuristics=(Heuristic*64)();count=I();ad,bd,dd=self.layouts
        self.check(self.lib.cublasLtMatmulAlgoGetHeuristic(self.handle,self.desc,ad,bd,dd,dd,self.pref,64,self.heuristics,C.byref(count)))
        self.indices=[i for i in range(count.value) if self.heuristics[i].state==0]
        assert self.indices,'No supported cuBLASLt algorithms'
        self.index=self.indices[0];self.alpha=C.c_float(1);self.beta=C.c_float(0)

    @staticmethod
    def check(status):
        if status:raise RuntimeError(f'cuBLASLt status {status}')

    def __call__(self):
        ad,bd,dd=self.layouts
        self.check(self.lib.cublasLtMatmul(self.handle,self.desc,C.byref(self.alpha),
            self.a.data_ptr(),ad,self.b.data_ptr(),bd,C.byref(self.beta),
            self.out.data_ptr(),dd,self.out.data_ptr(),dd,C.byref(self.heuristics[self.index].algo),
            self.workspace.data_ptr(),self.workspace.numel(),torch.cuda.current_stream().cuda_stream))

    def close(self):
        for name,handles in (('cublasLtMatrixLayoutDestroy',self.layouts),
                             ('cublasLtMatmulDescDestroy',[self.desc]),
                             ('cublasLtMatmulPreferenceDestroy',[self.pref]),
                             ('cublasLtDestroy',[self.handle])):
            fn=getattr(self.lib,name);fn.argtypes=[C.c_void_p];fn.restype=C.c_int
            for handle in handles:self.check(fn(handle))
