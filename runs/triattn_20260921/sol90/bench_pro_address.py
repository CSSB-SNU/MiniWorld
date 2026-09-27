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
# Experimental compilation must fail directly, never poison later variants with safe tiles.
pf.run_cell=lambda lever,cc,triton,build,cfg,**kw: build(cfg)
class Launch:
 def __init__(self,opts):self.opts=opts
 def __getitem__(self,grid):
  def run(*args,**kwargs):
   ck=self.opts[grid](*args,**kwargs)
   metadata.update(regs=ck.n_regs,spills=ck.n_spills,shared=ck.metadata.shared)
   return ck
  return run
  
variants=[('baseline',None,base)]
for variant in ('const','i32','both'):
 cfg=copy.deepcopy(base)
 module=importlib.import_module('pro_'+variant)
 variants.append((variant,getattr(module,'_triatt_prologue_kernel'),cfg))
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
