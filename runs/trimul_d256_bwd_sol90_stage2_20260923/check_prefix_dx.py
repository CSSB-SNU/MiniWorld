from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_hybrid_b1 import WideB1
from wide_b7 import WideB7
from prefix_dx import PrefixDx
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-prefix-dx-{os.environ.get("PREFIX_IMPL","cute")}-{os.environ.get("PREFIX_COPY","torch")}-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 if D==256:
  os.environ.update(GP_OFF='1',GP_WARP='1',DX_LN='1',DX_WARP='1',SAVE_NORM='1',LN_THREADS='128',LN_MINBLOCKS='3',LN_AGG='1',LN_CACHE='0',SHARED_CANDIDATE='saved_products',SAVE_EPI_GRID='1056',SAVE_EPI_THREADS='256')
  from selected_current import Training
  from shared_candidate import attach
  from saved_input_stats import enable as input_enable
  from tma_b7 import TmaB7
  plan=Training(leaves,mask,ds,dy);attach(plan);p=plan.p;f=plan.f;b1=plan.b1
  finish=PrefixDx(p,leaves,splits=8 if N==384 else 16)
  plan.b7=TmaB7(p,leaves,packed=True,mask=f.mask);input_enable(plan,gamma_cache=True);b7=plan.b7
  def source():b7.source.launch((32*b7.splits,1,1),(b7.source_threads,1,1),[b7.params],114816)
 else:
  f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn),packed=f.w)
  finish=PrefixDx(p,leaves);b1=WideB1(p,leaves);b7=WideB7(p,f.mask)
  source=b7.source_only
 def contract():
  ab=p.front.ab;h=2*D
  torch.bmm(p.dt[:D],ab[h:h+D],out=p.dl[:D]);torch.bmm(p.dt[:D].transpose(-1,-2),ab[:D],out=p.dr[:D])
  torch.bmm(ab[h+D:],p.dt[D:].transpose(-1,-2),out=p.dl[D:]);torch.bmm(ab[D:h],p.dt[D:],out=p.dr[D:])
 def old():b1();contract();return b7()
 def new():b1();contract();source();finish();return p.outputs
 def oldfull():return f(),old()
 def newfull():return f(),new()
 f();expected=[t.clone() for t in old()]
 p.tensors[10].fill_(float('nan'));actual=[t.clone() for t in new()];torch.cuda.synchronize()
 record['errors']={n:error(a,b) for n,a,b in zip(names[1:],actual,expected)}
 record['strict']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 record['prefix_error']=error(finish.input[:D],p.dg.t());assert record['prefix_error']==0
 record['candidate']=dict(cubin=str(finish.cubin),copy_cubin=str(getattr(finish,'copy_cubin','')),config=finish.cfg,impl=finish.impl,copy_impl=finish.copy_impl,prefix_bytes=D*p.M*2)
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  functions=dict(old_bwd=old,new_bwd=new,old_full=oldfull,new_full=newfull,matrix=finish.matrix_product)
  if os.environ.get('PREFIX_PHASES')=='1':
   record['diagnostic_only']=True
   functions=dict(prefix_copy=finish.copy_prefix,weight_pack=finish.pack_weights,gemm=finish.gemm_only,reduce=finish.reduce_only,matrix=finish.matrix_product)
  record['times']=paired(functions)
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
