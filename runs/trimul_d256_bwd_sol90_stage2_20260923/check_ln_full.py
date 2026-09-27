"""Compare the complete saved-norm checkpoint with the LN occupancy candidate."""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
with torch.no_grad(),T.native_context(leaves[0].device):
 os.environ.update(LN_THREADS='256',LN_MINBLOCKS='2',LN_AGG='0',LN_CACHE='0')
 old=Training(leaves,mask,ds,dy)
 os.environ.update(LN_THREADS='128',LN_MINBLOCKS='3',LN_AGG='1',LN_CACHE='0')
 new=Training(leaves,mask,ds,dy)
 yo,go=old();expected=[x.clone() for x in go];yn,gn=new()
 assert torch.equal(yo,yn)
 record['errors']={name:error(a,b) for name,a,b in zip(names[1:],gn,expected)}
 record['strict']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 print('ERROR',record['errors'],'STRICT',record['strict'],flush=True);assert record['strict']
 record['times']=paired(dict(old_b1=old.b1,new_b1=new.b1,old_bwd=old.backward,new_bwd=new.backward,old_full=old,new_full=new))
 print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;record['config']={k:v for k,v in os.environ.items() if k.startswith(('GP_','DX_','LN_','CUTE_','SAVE_'))}
(THIS/f'result-ln-full-L{N}-{record["job"]}.json').write_text(json.dumps(record,indent=2))
