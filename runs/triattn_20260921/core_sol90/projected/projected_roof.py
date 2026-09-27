from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'bench_fused.py').read_text().split('\nchecks=[]')[0]
exec(compile(s,str(HERE/'bench_fused.py'),'exec'))
with torch.no_grad():
 prep()
 for _ in range(10):fused_core()
 torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStart()
 fused_core();torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStop()
