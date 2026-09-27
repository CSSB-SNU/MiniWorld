from pathlib import Path
import sys,os,json,hashlib
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint11 import Training as WideTraining,configuration
from validate_engine import capture
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,stress=[])
path=THIS/f'validation-wide-checkpoint11-D{D}-L{N}-{record["job"]}.json'
def strict(es):return all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=WideTraining(leaves,mask,ds,dy);f=plan.f;p=plan.p
 if D==384:old_b1=B1(p)
 else:
  def old_b1():W.launch(p.ks['b1'],p.params,p.grid,D=D)
 record['artifacts']=[dict(path=str(a),sha256=hashlib.sha256(Path(a).read_bytes()).hexdigest()) for a in plan.artifacts]
 record['source_sha256']={name:hashlib.sha256((THIS/name).read_bytes()).hexdigest() for name in ('wide_combined.py','wide_b7.py','wide_source.cu','prefix_dx.py','prefix_transpose.cu','wide_saved_norm.py','wide_hybrid_b1.py','wide_checkpoint2.py','wide_saved_products.py','wide_split_dwp.py','wide_bounded_ln.py','lt_contract.py','wide_checkpoint11.py','wide_pipe_source.py','wide_pipe_source.cu','wide_offset_source.py','wide_input_reduce.py','wide_checkpoint3.py','wide_dense_output.py','wide_tile_ln.py','wide_tile_ln.cu','wide_tma_input.py','wide_tma_input.cu','ln_aggregate.cuh','wide_checkpoint4.py','wide_mask_source.py','wide_mask_transform.py','wide_tile_output.py','wide_tile_output.cu','tile_transpose.cuh','wide_lt_schedule.py','wide_checkpoint5.py','wide_checkpoint5_lt.json','wide_checkpoint6_dw.json','wide_checkpoint6.py','wide_four_group_source.py','wide_saved_gp_pipe.py','wide_saved_gp_pipe.cu','wide_saved_front.py','wide_saved_front_gp.cu','wide_saved_front_tma.py','wide_saved_split_source.py','wide_cached_input.py','wide_cached_ln.py','wide_checkpoint7.py','wide_saved_front_transposed.py','wide_two_group_contract_gp.py','wide_pipe_contract_gp.cu','wide_contract_gp_tma.py','wide_contract_gp.cu','mma_offset.cuh','wide_checkpoint8.py','wide_saved_front_overlap.py','prefix_gate_epi.py','prefix_gate_epi.cu','prefix_gate_checkpoint.py','prefix_gate_choices.json','wide_checkpoint9.py','wide_checkpoint10_dw.json','wide_joint_input_reduce.py','wide_prefetch_ln.py','wide_prefetch_ln.cu','tile_transpose8.cuh','wide_checkpoint10.py','wide_delta_norm.py','wide_flagged_projection.cu','wide_compact_projection.py','wide_compact_projection.cu',)}
 record['configuration']=configuration(plan)
 print('ARTIFACTS',record['artifacts'],flush=True)
 def old():
  y=plan.forward(saved=False);old_b1();plan.contract()
  W.launch(p.ks['b7'],p.params7,p.grid7,D=D,gp=p.gp_native)
  return y,p.outputs
 new=plan
 new();torch.cuda.synchronize()
 if os.environ.get('SANITIZE')=='1':
  if os.environ.get('FORCE_NORM_OVERFLOW')=='1':assert int(plan.delta_output.count.item())>plan.delta_output.capacity
  if os.environ.get('FORCE_ROW_OVERFLOW')=='1':assert int(plan.delta_projection.count.item())>plan.delta_projection.capacity
  print('SANITIZER_DONE',D,N,flush=True);raise SystemExit
 graph,graph_out=capture(new)
 for case in ('normal','changed','zero_gamma','mask_zero','dropout_zero'):
  if case=='changed':
   leaves[0].mul_(.7);dy.normal_();leaves[1].mul_(1.1);leaves[2].mul_(.9);leaves[5].mul_(1.05);leaves[6].mul_(.9)
   mask.copy_((torch.rand_like(mask.float())>.3).bfloat16());ds.copy_((torch.rand_like(ds.float())>.4).bfloat16()*(1/.6))
  if case=='zero_gamma':leaves[7][::3]=0;leaves[9][::3]=0
  if case=='mask_zero':mask.zero_()
  if case=='dropout_zero':mask.fill_(1);ds.zero_()
  yo,go=old();yo=yo.clone();expected=[t.clone() for t in go]
  for t in (p.gp_all,p.tensors[10],p.tensors[6],p.floats[5],p.floats[6],p.floats[7],plan.b1.proj,plan.b1.gate):t.fill_(float('nan'))
  if hasattr(plan,'pre'):plan.pre.fill_(float('nan'))
  plan.dx.input[:D].fill_(float('nan'))
  yn,gn=new();actual=[t.clone() for t in gn]
  es={n:error(a,b) for n,a,b in zip(names[1:],actual,expected)}
  row=dict(case=case,strict=es,passed=strict(es),forward_error=error(yn,yo));record['stress'].append(row)
  path.write_text(json.dumps(record,indent=2));print('STRESS',row,flush=True)
  assert row['passed'] and row['forward_error']==0,row
  graph.replay();torch.cuda.synchronize();ge={n:error(a,b) for n,a,b in zip(names[1:],p.outputs,actual)}
  row['graph']=ge;assert max(ge.values())<5e-6,(case,ge)
  row['graph_forward']=error(graph_out[0],yn);assert row['graph_forward']==0
  if case=='normal':normal=[t.clone() for t in actual];normal_y=yn.clone()
  path.write_text(json.dumps(record,indent=2))
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
y=torch.compile(ref)(*leaves,mask,ds);grads=torch.autograd.grad(y,leaves,dy)
record['pytorch_forward']=error(normal_y,y)
assert record['pytorch_forward']<.005,record['pytorch_forward']
record['pytorch']={n:error(a,b) for n,a,b in zip(names[1:],normal,grads)}
assert max(record['pytorch'].values())<.01,record['pytorch']
record['complete']=True;path.write_text(json.dumps(record,indent=2));print('VALIDATED',D,N,flush=True)
