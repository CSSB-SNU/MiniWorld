"""Consumer-only screen: no full gain claim before forward packing is integrated."""
from pathlib import Path
import sys,os,json,gc
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint19 import Training
from wide_adaptive_exponent_pre import AdaptiveExponentPre
from wide_adaptive_exponent_gp import AdaptiveExponentGP
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=(384,512)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))];N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),consumer_only=True,complete=False,candidates=[])
path=THIS/f'result-wide-adaptive-exponent-gp-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];gp=p.gp_all.clone()
    old=plan.contract_gp;owners=[]
    for xor in (116,):
        packed=AdaptiveExponentPre(plan.pre);packed();op=AdaptiveExponentGP(plan,packed)
        owners.append((packed,op))
        item=dict(xor=xor,cubin=str(op.cubin),registers=op.registers,local_bytes=op.local_bytes,decoder_all_bits=op.validate_decoder(),escape_fraction=float((packed.bases==255).float().mean().item()))
        record['candidates'].append(item);assert item['decoder_all_bits']
        p.gp_all.fill_(float('nan'));plan.contract_gp=op;gn=plan.backward();torch.cuda.synchronize()
        item['gp_error']=error(p.gp_all,gp)
        item['errors']={n:error(a,b) for n,a,b in zip(names[1:],gn,expected[1:])}
        item['strict']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict'] and not op.local_bytes:
            item['stage_times']=paired(dict(old_gp=old,new_gp=op));gc.collect()
            def oldbw():plan.contract_gp=old;return plan.backward()
            def newbw():plan.contract_gp=op;return plan.backward()
            item['backward_times']=paired(dict(old_backward=oldbw,new_backward=newbw));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('stage_times','backward_times') for k,v in item[key].items()},flush=True)
        plan.contract_gp=old;plan();path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
