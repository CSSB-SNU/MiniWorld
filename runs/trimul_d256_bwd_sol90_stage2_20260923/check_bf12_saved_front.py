"""Full training pilot, conditional on a useful lossless consumer prototype."""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
array=os.environ['PROTOTYPE_ARRAY_JOB']
eligible=[]
for path in THIS.glob(f'result-bf12-contract-gp-D{D}-L{N}-*.json'):
    data=json.loads(path.read_text())
    if data.get('array_job')==array and data.get('complete') and data.get('strict'):
        ts=data['stage_times']
        if ts['new_gp']['median_us']<ts['old_gp']['median_us']*.98:eligible.append(str(path))
if not eligible:
    print('SKIP_NO_USEFUL_CONSUMER',D,N,array,flush=True);raise SystemExit(0)
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint14 import Training
from wide_bf12_contract_gp import PackedPre,PackedContractGP
from wide_bf12_saved_front import PackedSavedFront
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,forward_integrated=True,prototype=eligible)
path=THIS/f'result-bf12-saved-front-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]]
    packed=PackedPre(plan.pre);packed();op=PackedContractGP(plan,packed)
    original_gp=plan.contract_gp;original_front=plan.f.front
    front=PackedSavedFront(plan.baseline_front,packed)
    record['cubins']=[str(front.cubin),str(op.cubin)]
    def old():
        plan.f.front=original_front;plan.contract_gp=original_gp;return plan()
    def new():
        plan.f.front=front;plan.contract_gp=op;return plan()
    # Common preactivation locations must never be read by the new consumer.
    # This also catches missing sparse writes that old forward could conceal.
    plan.pre.fill_(float('nan'));yn,gn=new();torch.cuda.synchronize()
    record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
    record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
    print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
    if record['strict']:
        record['front_times']=paired(dict(old_front=original_front,new_front=front))
        record['full_times']=paired(dict(old_full=old,new_full=new))
        print('TIMES',{k:v['median_us'] for key in ('front_times','full_times') for k,v in record[key].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
