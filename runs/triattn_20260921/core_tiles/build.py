from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/'build')
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['MAX_JOBS']='2'
import importlib.util
spec=importlib.util.spec_from_file_location('triattn_r1_module',HERE/'r1/triattn_m1.py')
triattn_m1=importlib.util.module_from_spec(spec)
spec.loader.exec_module(triattn_m1)
if __name__=='__main__':triattn_m1._build(verbose=True)
