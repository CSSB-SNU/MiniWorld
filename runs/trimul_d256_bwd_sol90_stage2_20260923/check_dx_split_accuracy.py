"""Numerical feasibility only: local GP-to-dX products plus FP32 reductions.

No speed claim: these intentionally materialized PyTorch products assess
whether the proposed producer-local dX split can meet the existing tolerance.
"""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,numerical_only=True,candidates={})
path=THIS/f'result-dx-split-accuracy-L{N}-{record["job"]}.json'
torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);plan();p=plan.p;expected=[x.clone() for x in p.outputs]
 # The existing unfused finish also emits the BF16 dX-normalized intermediate.
 W.launch(plan.b7.finish,p.params7,plan.b7.grid,D=256)
 record['legacy_errors']={n:error(x,y) for n,x,y in zip(names[1:],p.outputs,expected)}
 assert record['legacy_errors']['dx']<2e-5
 expected_dxn=p.tensors[10].clone()
 base=(PRE/'base.cuh').read_text().replace('template<bool INPUT> TMN_DEVI void ln_bwd(const Params& p){',(THIS/'ln_aggregate.cuh').read_text()+'\ntemplate<bool INPUT> TMN_DEVI void ln_bwd(const Params& p,uint8_t* sm){')
 base=base.replace(' for(int c=lane;c<C;c+=32){atomicAdd(p.f[INPUT?8:10]+c,gg[c/32]);atomicAdd(p.f[INPUT?9:11]+c,bb[c/32]);}',' aggregate_ln<C,THREADS>(gg,bb,reinterpret_cast<float*>(sm),p.f[INPUT?8:10],p.f[INPUT?9:11]);')
 body=(PRE/'finish.cu').read_text().replace('#include "base.cuh"',base).replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text()).replace('front_dx(p,sm,bar,phase);','').replace('ln_bwd<true>(p)','ln_bwd<true>(p,sm)').replace('mw_d256_b7_finish','mw_d256_split_accuracy_finish')
 flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={plan.b7.splits}','-DWIDTH=256']
 cubin=T.compile_text(body,flags);finish=T.load_unit(str(cubin),'mw_d256_split_accuracy_finish').kernel('mw_d256_split_accuracy_finish');finish.set_max_dynamic_smem(W.tuning(256)[2])
 drv=finish.unit.drv;occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(finish.handle)),256,W.tuning(256)[2])))
 grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
 weights=[w.float() for w in leaves[1:5]];wg=leaves[5].float();prefix=p.dg.float()@wg
 def product(plane,rank):
  sl=slice(rank*32,(rank+1)*32)
  return p.gp[plane][sl].t().float()@weights[plane][sl]
 def check(name,v):
  p.tensors[10].copy_(v);W.launch(finish,p.params7,grid,D=256);torch.cuda.synchronize()
  es={n:error(x,y) for n,x,y in zip(names[1:],p.outputs,expected)}
  strict=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['candidates'][name]=dict(errors=es,strict=strict,dxn_error=error(p.tensors[10],expected_dxn),dxn_changed=int((p.tensors[10]!=expected_dxn).sum()))
  print(name,record['candidates'][name],flush=True);path.write_text(json.dumps(record,indent=2))
 if os.environ.get('SPLIT_NATIVE')=='1':
  from split_dx_probe import SplitDXProbe
  probe=SplitDXProbe(p,leaves[1:5]);native_prefix,parts=probe();torch.cuda.synchronize()
  record['prefix_difference']=error(native_prefix,prefix)
  # Expose partial products and prefix independently before using their sum.
  record['partial_difference']=error(parts[0],product(0,0).add_(product(1,0)))
  print('NATIVE_PRODUCT_DIFFERENCES',record['prefix_difference'],record['partial_difference'],flush=True)
  assert record['prefix_difference']<1e-5 and record['partial_difference']<1e-5
  v=native_prefix.clone()
  for rank in range(32):v.add_(parts[rank])
  check('native_rank_paired_fp32',v);del v
  v=native_prefix.clone()
  for side in range(2):
   terms=list(parts[side*16:(side+1)*16].unbind())
   while len(terms)>1:terms=[terms[i].add_(terms[i+1]) for i in range(0,len(terms),2)]
   v.add_(terms[0]);del terms
  check('native_rank_tree_fp32',v)
 else:
  v=prefix.clone()
  for plane in range(4):
   for rank in range(16):v.add_(product(plane,rank))
  check('plane_major_fp32',v);del v
  v=prefix.clone()
  for side in range(2):
   for rank in range(16):v.add_(product(side*2,rank).add_(product(side*2+1,rank)))
  check('rank_paired_fp32',v);del v
  v=prefix.clone()
  for side in range(2):
   terms=[product(side*2,rank).add_(product(side*2+1,rank)) for rank in range(16)]
   while len(terms)>1:terms=[terms[i].add_(terms[i+1]) for i in range(0,len(terms),2)]
   v.add_(terms[0]);del terms
  check('rank_tree_fp32',v)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
