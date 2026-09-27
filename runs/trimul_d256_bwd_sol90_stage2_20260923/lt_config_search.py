"""Explicit algorithm capability/configuration search; CUDA 12.9 header ABI."""
import ctypes as C
import torch
from lt_contract import Algorithm, Heuristic


class ConfigSearch:
    def __init__(self, op):
        self.op=op;self.lib=op.lib
        P=C.c_void_p;I=C.c_int;Z=C.c_size_t;A=C.POINTER(Algorithm)
        declarations={
            'cublasLtMatmulAlgoGetIds':[P,I,I,I,I,I,I,I,C.POINTER(I),C.POINTER(I)],
            'cublasLtMatmulAlgoInit':[P,I,I,I,I,I,I,I,A],
            'cublasLtMatmulAlgoCapGetAttribute':[A,I,P,Z,C.POINTER(Z)],
            'cublasLtMatmulAlgoConfigGetAttribute':[A,I,P,Z,C.POINTER(Z)],
            'cublasLtMatmulAlgoConfigSetAttribute':[A,I,P,Z],
            'cublasLtMatmulAlgoCheck':[P,P,P,P,P,P,A,C.POINTER(Heuristic)],
        }
        for name,args in declarations.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=I
        dtype=lambda t:14 if t.dtype==torch.bfloat16 else 0
        self.types=(68,0,dtype(op.a),dtype(op.b),dtype(op.out),dtype(op.out))

    def ids(self):
        values=(C.c_int*1024)();count=C.c_int()
        self.op.check(self.lib.cublasLtMatmulAlgoGetIds(self.op.handle,*self.types,1024,values,C.byref(count)))
        return list(values)[:count.value]

    def init(self, ident):
        algo=Algorithm()
        self.op.check(self.lib.cublasLtMatmulAlgoInit(self.op.handle,*self.types,ident,C.byref(algo)))
        return algo

    def get(self,algo,attr,config=False):
        fn=self.lib.cublasLtMatmulAlgoConfigGetAttribute if config else self.lib.cublasLtMatmulAlgoCapGetAttribute
        size=C.c_size_t()
        status=fn(C.byref(algo),attr,None,0,C.byref(size))
        if status:return None
        if not size.value:return []
        if size.value==2:
            value=C.c_uint16();self.op.check(fn(C.byref(algo),attr,C.byref(value),2,C.byref(size)))
            return [value.value]
        values=(C.c_uint32*(size.value//4))()
        self.op.check(fn(C.byref(algo),attr,values,size.value,C.byref(size)))
        return list(values)

    def set(self,algo,attr,value):
        v=(C.c_uint16 if attr in (7,8) else C.c_uint32)(value)
        return self.lib.cublasLtMatmulAlgoConfigSetAttribute(C.byref(algo),attr,C.byref(v),C.sizeof(v))==0

    def valid(self,algo):
        op=self.op;ad,bd,dd=op.layouts;result=Heuristic()
        status=self.lib.cublasLtMatmulAlgoCheck(op.handle,op.desc,ad,bd,dd,dd,C.byref(algo),C.byref(result))
        return result if not status and not result.state and result.workspaceSize<=op.workspace.numel() else None

    @staticmethod
    def copy(algo):
        return Algorithm.from_buffer_copy(bytes(algo))

    def install(self,algo):
        C.memmove(C.byref(self.op.heuristics[self.op.index].algo),C.byref(algo),C.sizeof(algo))

    def candidates(self):
        """Unsplit K only; all advertised tiles, stages, custom options."""
        seen=set()
        def accepted(algo):
            key=bytes(algo)
            if key in seen:return False
            seen.add(key)
            return self.valid(algo) is not None
        for index in self.op.indices:
            algo=self.copy(self.op.heuristics[index].algo)
            if accepted(algo):yield algo
        for ident in self.ids():
            base=self.init(ident)
            tiles=self.get(base,6) or [0];stages=self.get(base,13) or [0]
            custom=(self.get(base,7) or [0])[0]
            swizzle=(self.get(base,2) or [0])[0]
            clusters=[0,2,3,4,5,6,7,8,9,11,12] if ident==66 else [0]
            for tile in tiles:
                for stage in stages:
                    for option in range(custom+1):
                        for sw in range(swizzle+1):
                            for cluster in clusters:
                                algo=self.copy(base)
                                values=((1,tile),(2,1),(3,0),(4,sw),(5,option),(6,stage),(8,cluster))
                                if all(self.set(algo,k,v) for k,v in values) and accepted(algo):yield algo
