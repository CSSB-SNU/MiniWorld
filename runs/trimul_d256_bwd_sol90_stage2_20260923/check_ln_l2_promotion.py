from pathlib import Path
import sys,os,json,gc,copy
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint15 import Training
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
i=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[i//2];N=(384,768)[i%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-ln-l2-promotion-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dt=p.dt.clone();oldln=plan.ln
    for promotion in ('none','64B','256B'):
        op=copy.copy(oldln);rows=oldln.rows;L=T._launch_module();h=2*D
        tm=lambda t:L.tensor_map(t,[rows,64],dims=[p.M,h],strides_bytes=[p.M*2],swizzle='64B' if rows==32 else '32B',l2=promotion)
        dn=L.tensor_map(p.tensors[9],[64,rows],dims=[h,p.M],strides_bytes=[h*2],swizzle='128B',l2=promotion)
        op.params=L.Struct([tm(p.tri),tm(p.dt),dn,*oldln.params.fields[3:]])
        item=dict(promotion=promotion);record['candidates'].append(item)
        def before():plan.ln=oldln;return plan()
        def after():plan.ln=op;return plan()
        yn,gn=after();torch.cuda.synchronize();item['dt_error']=error(p.dt,dt)
        item['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
        item['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in item['errors'].items())
        print('CHECK',item,flush=True);path.write_text(json.dumps(record,indent=2))
        if item['strict']:
            item['stage_times']=paired(dict(old_ln=oldln,new_ln=op));gc.collect()
            item['full_times']=paired(dict(old_full=before,new_full=after));gc.collect()
            print('TIMES',{k:v['median_us'] for key in ('stage_times','full_times') for k,v in item[key].items()},flush=True)
        before()
record['complete']=True;path.write_text(json.dumps(record,indent=2))
