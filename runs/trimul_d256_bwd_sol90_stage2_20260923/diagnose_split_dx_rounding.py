"""Diagnostic only: measure whether rounding-boundary repair could be sparse.

Oracle substitution quantifies the potential repair workload. It is not a valid
optimized kernel and receives no timing or speedup claim.
"""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_spatial_checkpoint import Training
from d256_fp32_dx_witness import Fp32DxWitness
from lt_contract import LtBmm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
leaves,dy,mask,ds,ref,triton,names=setup(256,384)
record=dict(job=os.environ.get('SLURM_JOB_ID'),diagnostic_only=True,oracle_not_a_kernel=True,complete=False,cases=[])
path=THIS/f'diagnose-split-dx-rounding-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;plan()
    reference=p.tensors[10].reshape(p.M,256).clone();dx=p.dx.clone()
    witness=Fp32DxWitness(plan);witness();torch.cuda.synchronize()
    record['witness_cubin']=str(witness.cubin)
    record['witness_bf16_error']=error(witness.witness.to(torch.bfloat16),reference)
    assert record['witness_bf16_error']==0
    workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
    for splits in (2,3,9):
        partial=torch.empty((splits,p.M,256),device=p.x.device,dtype=torch.float32)
        ops=[];width=2304//splits
        for s in range(splits):
            a=plan.dx.input[s*width:(s+1)*width].T.unsqueeze(0)
            b=plan.dx.weights[s*width:(s+1)*width].unsqueeze(0)
            op=LtBmm(a,b,partial[s:s+1],workspace);ops.append(op);op()
        total=partial[0].clone()
        for s in range(1,splits):total.add_(partial[s])
        rounded=total.to(torch.bfloat16);wrong=rounded!=reference
        distance=((total.view(torch.int32)&65535)-32768).abs()
        item=dict(splits=splits,wrong=int(wrong.sum().item()),total=wrong.numel(),fp32_rel=error(total,witness.witness),bf16_rel=error(rounded,reference),thresholds=[])
        for threshold in (0,4,8,16,32,64,128,256):
            selected=distance<=threshold
            repaired=torch.where(selected,reference,rounded)
            p.tensors[10].copy_(repaired.reshape_as(p.tensors[10]))
            p.floats[8].zero_();p.floats[9].zero_();plan.dx.reduce_only();torch.cuda.synchronize()
            result=dict(ulp_threshold=threshold,selected=int(selected.sum().item()),selected_rows=int(selected.any(dim=1).sum().item()),remaining_wrong=int((wrong&~selected).sum().item()),dx_rel=error(p.dx,dx))
            item['thresholds'].append(result)
        record['cases'].append(item);print('DIAGNOSTIC',item,flush=True);path.write_text(json.dumps(record,indent=2))
        for op in ops:op.close()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
