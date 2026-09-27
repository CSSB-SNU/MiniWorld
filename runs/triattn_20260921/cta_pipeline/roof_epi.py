from pathlib import Path
import sys,os,json,statistics
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'oc'))
ending=os.environ.get('ENDING')=='1'
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
from build_epi import build
from opt_core.kernels import triattn_surround_tma as native
ext=build();module=native._load();original=module.epilogue

with torch.no_grad():
 qp,kp,vp,gp,bp=pf.prologue(x,W,impl='fpf',ending=ending)
 op=pf.core_attention(qp,kp,vp,bp,m5,core='tier:triattn_native')
 def epi():return pf.epilogue(op,gp,W,x,impl='fpf',residual=True,out=buf,ending=ending)
 for label,fn in [('baseline',original),('persistent128',ext.epi128)]:
  module.epilogue=fn
  print(label,graph_us(epi),flush=True)
  for _ in range(50):epi()
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
  epi()
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
