from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint17 import Training
from wide_ab_projection_gp import ABProjectionGP
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
i=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[i//2];N=(384,768)[i%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,cases=[],forward_integrated=False)
path=THIS/f'result-ab-projection-gp-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;plan();old=plan.contract_gp;op=ABProjectionGP(plan)
    record.update(registers=op.registers,local_bytes=op.local_bytes,cubins=[str(op.cubin),str(op.pack_cubin)])
    for case in ('normal','nonbinary','mask_zero'):
        if case=='nonbinary':
            mask.view(-1)[::11]=.375;mask.view(-1)[::13]=-.25;mask.view(-1)[::17]=-0.;mask.view(-1)[::19]=.0001
        if case=='mask_zero':mask.zero_()
        plan.contract_gp=old;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
        op.escapes.fill_(float('nan'));op.encode();plan.contract_gp=op
        p.gp_all.fill_(float('nan'));gn=plan.backward();torch.cuda.synchronize()
        item=dict(case=case,gp_error=error(p.gp_all,gp),code_counts=op.stats.tolist())
        item['gp_bit_mismatches']=int(torch.count_nonzero(p.gp_all.view(torch.int16)!=gp.view(torch.int16)).item())
        item['errors']={n:error(a,b) for n,a,b in zip(names[1:],gn,expected[1:])}
        item['strict']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        record['cases'].append(item);print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if case=='normal' and item['strict']:
            record['stage_times']=paired(dict(old_gp=old,new_gp=op));gc.collect()
            def before():plan.contract_gp=old;return plan.backward()
            def after():plan.contract_gp=op;return plan.backward()
            record['backward_times']=paired(dict(old_backward=before,new_backward=after));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('stage_times','backward_times') for k,v in record[key].items()},flush=True)
        del expected,gp;gc.collect()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
