from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'oc'))
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
PRO=pf._carried('fpf_triatt_pro','prologue');EPI=pf._carried('fpf_triatt_epi','epilogue')
class Recorder:
 def __init__(self,k,name):self.k=k;self.name=name;self.seen=set()
 def __getitem__(self,grid):
  def launch(*args,**kw):
   ck=self.k[grid](*args,**kw)
   if id(ck) not in self.seen:
    self.seen.add(id(ck)); print('COMPILED',self.name,'regs',ck.n_regs,'spills',ck.n_spills,'smem',ck.metadata.shared,flush=True)
    for ext in ('ttgir','ptx'): (R/'sol90'/f'{self.name}.{ext}').write_text(ck.asm[ext])
   return ck
  return launch
PRO._triatt_prologue_kernel=Recorder(PRO._triatt_prologue_kernel,'pro')
EPI._triatt_epilogue_kernel_v2=Recorder(EPI._triatt_epilogue_kernel_v2,'epi')
with torch.no_grad():
 for _ in range(4):call()
 torch.cuda.synchronize()
