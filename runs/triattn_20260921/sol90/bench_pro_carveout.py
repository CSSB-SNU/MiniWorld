from pathlib import Path
import sys,copy,json,statistics,os
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'oc'))
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
import importlib
E=pf._carried('fpf_triatt_pro','prologue'); original=E._triatt_prologue_kernel
row=next(r for r in pf.cells()['rows'] if r['id']=='r07');base=copy.deepcopy(row['cfg'])
metadata={}
import ctypes
cuda=ctypes.CDLL('libcuda.so.1')
cuda.cuFuncSetAttribute.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
cuda.cuFuncSetAttribute.restype=ctypes.c_int

# Experimental compilation must fail directly, never poison later variants with safe tiles.
pf.run_cell=lambda lever,cc,triton,build,cfg,**kw: build(cfg)
class Launch:
 def __init__(self,opts):self.opts=opts;self.seen=set()
 def __getitem__(self,grid):
  def run(*args,**kwargs):
   ck=self.opts[0][grid](*args,**kwargs)
   if ck.function not in self.seen:
    assert cuda.cuFuncSetAttribute(ck.function,9,self.opts[1])==0
    self.seen.add(ck.function)
   metadata.update(regs=ck.n_regs,spills=ck.n_spills,shared=ck.metadata.shared)
   return ck
  return run
  
variants=[('baseline',None,base)]
import pro_slabs
for kind in ('native','slabs'):
 for carveout in (-1,0,25,50,75,100):
  cfg=copy.deepcopy(base)
  if kind=='slabs':cfg.update(BI=1,BJ=64,num_warps=4,num_stages=1,maxnreg=128)
  variants.append((f'{kind}_carve{carveout}',(original if kind=='native' else pro_slabs._triatt_prologue_kernel,carveout),cfg))
results=[]
with torch.no_grad():
 expected=call().clone()
 # Stable operands for isolated epilogue measurements; block timings qualify finalists later.
 q,k,v,gate,bias=pf.prologue(x,W,impl='fpf')
 o=pf.core_attention(q,k,v,bias,m5,core='tier:triattn_native')
 def component():return pf.prologue(x,W,impl='fpf')
 for name,opts,cfg in variants:
  row['cfg']=cfg;E._triatt_prologue_kernel=original if opts is None else Launch(opts)
  try:
   y=call().clone();eq=torch.equal(y,expected)
   delta=float((y.float()-expected.float()).abs().max())
   if not eq:
    print('REJECT',name,delta,flush=True);results.append(dict(name=name,equal=False,max_abs=delta));Path(a.output).write_text(json.dumps(results,indent=2));continue
   times=[graph_us(component) for _ in range(3)]
   result=dict(name=name,equal=True,us=statistics.median(times),rounds=times,metadata=copy.deepcopy(metadata),cfg=cfg,opts=name)
   print('RESULT',json.dumps(result),flush=True);results.append(result)
  except Exception as e:
   print('ERROR',name,type(e).__name__,str(e)[:3500],flush=True);results.append(dict(name=name,error=str(e)))
  Path(a.output).write_text(json.dumps(results,indent=2))
 row['cfg']=base;E._triatt_prologue_kernel=original
