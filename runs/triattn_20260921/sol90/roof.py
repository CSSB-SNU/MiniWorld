from pathlib import Path
import sys,os,json,copy
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'oc'))
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
PRO=pf._carried('fpf_triatt_pro','prologue');EPI=pf._carried('fpf_triatt_epi','epilogue')
variant=os.environ.get('SOL_VARIANT','baseline')
if variant!='baseline':
 import pro_const_biasfirst,epi_const
 PRO._triatt_prologue_kernel=pro_const_biasfirst._triatt_prologue_kernel
 EPI._triatt_epilogue_kernel_v2=epi_const._triatt_epilogue_kernel_v2
 rows={r['id']:r for r in pf.cells()['rows']}
 rows['r07']['cfg'].update(BI=1,BJ=64,num_warps=4,num_stages=1,maxnreg=128)
 if os.environ.get('REFINED_BN'):rows['r07']['cfg']['BN']=int(os.environ['REFINED_BN'])
 if os.environ.get('REFINED_CAP'):rows['r07']['cfg']['maxnreg']=int(os.environ['REFINED_CAP'])
# For experiments, compilation failures cannot select a different tile silently.
pf.run_cell=lambda lever,cc,triton,build,cfg,**kw:build(cfg)
with torch.no_grad():
 y=call()
 q,k,v,gate,bias=pf.prologue(x,W,impl='fpf')
 o=pf.core_attention(q,k,v,bias,m5,core='tier:triattn_native')
 copybuf=torch.empty_like(x)
 def copyfn():return copybuf.copy_(x)
 def pro():return pf.prologue(x,W,impl='fpf')
 def epi():return pf.epilogue(o,gate,W,x,impl='fpf',residual=True,out=buf)
 timings={name:graph_us(fn) for name,fn in [('copy',copyfn),('pro',pro),('epi',epi)]}
 print('TIMING',variant,json.dumps(timings),flush=True)
 for _ in range(100):copyfn();pro();epi()
 torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStart()
 copyfn();pro();epi()
 torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStop()
 Path(a.output).write_text(json.dumps(dict(variant=variant,timings=timings,copy_bytes=2*x.numel()*x.element_size()),indent=2))
