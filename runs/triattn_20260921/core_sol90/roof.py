from pathlib import Path
source=(Path(__file__).resolve().parent/'bench.py').read_text().split('\nwith torch.no_grad():')[0]
exec(compile(source,str(Path(__file__).resolve().parent/'bench.py'),'exec'))
with torch.no_grad():
 if mode.startswith('baseablate'):print('DIAGNOSTIC ABLATION: deliberately changes the computation; not an eligible performance result.',flush=True)
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=ending)
 def core():return pf.core_attention(q,k,v,b,m5,core=candidate)
 print('CORE_US',graph_us(core),flush=True)
 for _ in range(40):core()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();core();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
