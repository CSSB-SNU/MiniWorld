from pathlib import Path
import os
def use_ptxas():
 import torch.utils.cpp_extension as ce
 root=Path(__file__).resolve().parent/'toolchain'
 original=Path(ce.CUDA_HOME or '/usr/local/cuda-12.9')
 shim=root/'cuda_home_134';(shim/'bin').mkdir(parents=True,exist_ok=True)
 for p in original.iterdir():
  q=shim/p.name
  if p.name!='bin' and not q.exists():q.symlink_to(p,target_is_directory=p.is_dir())
 for p in (original/'bin').iterdir():
  q=shim/'bin'/p.name
  if not q.exists():q.symlink_to(root/'ptxas-13.4.92' if p.name=='ptxas' else p)
 ce.CUDA_HOME=str(shim);os.environ['CUDA_HOME']=str(shim)
 return shim
