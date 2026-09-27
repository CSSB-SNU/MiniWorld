from pathlib import Path
import sys,os,json,statistics,hashlib
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'oc'))
ending=os.environ.get('ENDING')=='1'
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
source=source.replace('starting=True',f'starting={not ending}').replace('ending=False',f'ending={ending}')
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
def select(cuda):
 os.environ['FPF_TRIATT_BACKEND']='cuda_tma' if cuda else 'triton'


if os.environ.get('TMA_VARIANT')=='baseline':select(False)
else:os.environ.pop('FPF_TRIATT_BACKEND',None)
with torch.no_grad():
 qp,kp,vp,gp,bp=pf.prologue(x,W,impl='fpf',ending=ending)
 op=pf.core_attention(qp,kp,vp,bp,m5,core='tier:triattn_native')
 copybuf=torch.empty_like(x)
 def pro():return pf.prologue(x,W,impl='fpf',ending=ending)
 def epi():return pf.epilogue(op,gp,W,x,impl='fpf',residual=True,out=buf,ending=ending)
 funcs={'copy':lambda:copybuf.copy_(x),'pro':pro,'epi':epi}
 timings={k:[graph_us(fn) for _ in range(3)] for k,fn in funcs.items()}
 print('TIMINGS',json.dumps(timings),flush=True)
 Path(a.output).write_text(json.dumps({'timings':timings,'copy_bytes':2*x.numel()*x.element_size()},indent=2))
 for _ in range(100):pro();epi()
 torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStart()
 pro();epi()
 torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStop()
