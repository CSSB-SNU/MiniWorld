"""Opt in to Hopper's lossless memory compression in an isolated PyTorch pool."""
from pathlib import Path
import ctypes,hashlib,os,subprocess
import torch

class InlineCompressedPool:
    def __init__(self):
        assert os.environ.get('SLURM_JOB_ID'),'Compile only on a compute allocation'
        root=Path(__file__).resolve().parent;source=root/'inline_compressed_allocator.cpp'
        digest=hashlib.sha256(source.read_bytes()).hexdigest()[:20]
        directory=root/'allocator_cache';directory.mkdir(exist_ok=True)
        library=directory/f'ilc_{digest}.so'
        if not library.exists():
            temp=directory/f'ilc_{digest}_{os.getpid()}.so'
            cuda=Path(os.environ.get('CUDA_HOME','/usr/local/cuda-12.9'))
            subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC','-pthread',str(source),
                            '-I'+str(cuda/'include'),'-L'+str(cuda/'lib64/stubs'),'-lcuda','-o',str(temp)],check=True)
            temp.replace(library)
        self.library=ctypes.CDLL(str(library));self.library.mw_ilc_stat.argtypes=[ctypes.c_int]
        self.library.mw_ilc_stat.restype=ctypes.c_ulonglong
        self.allocator=torch.cuda.memory.CUDAPluggableAllocator(str(library),'mw_ilc_malloc','mw_ilc_free')
        self.pool=torch.cuda.MemPool(self.allocator.allocator())
    def stats(self):
        return {key:int(self.library.mw_ilc_stat(i)) for i,key in enumerate(('allocations','compressible_allocations','allocated_bytes','errors'))}
    def context(self):return torch.cuda.use_mem_pool(self.pool)
