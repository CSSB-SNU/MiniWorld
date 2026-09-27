from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'oc'))
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
with torch.no_grad():
 q,k,v,g,b=pf.prologue(x,W,impl='fpf',ending=False)
 def core():return pf.core_attention(q,k,v,b,m5,core='tier:triattn_native')
 print('CORE_GRAPH_US',graph_us(core),flush=True)
 for _ in range(40):core()
 torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();core();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
