"""Diagnostic lower bound only: remove unused GP publication, retain local dW."""
from pathlib import Path
import sys,os,json,re
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_permuted_ln_checkpoint import Training
from wide_mask_transform import mask_stage
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=256;N=384
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),diagnostic_only=True,complete=False)
path=THIS/f'result-d256-source-publication-cost-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
    plan=Training(leaves,mask,ds,dy);plan();p=plan.p;prior=plan.b7
    part=p.floats[7].reshape(-1)[3*D*D:].as_strided((prior.splits,8*D*D),(11*D*D,1));expected=part.clone()
    body=mask_stage(prior.original.source_text,'bulk').replace('sm+BAR+128+slot*128','sm+114816+slot*128')
    body,n=re.subn(r'  if\(tid==0\)\{store_gp\(.*?tma_store_commit\(\);\}', '',body);assert n==1,n
    body=body.replace('if(tid==0)tma_store_wait_all();','')
    body=body.replace('mw_d256_b7_tma','mw_d256_source_no_publication')
    flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={prior.splits}']
    cubin=T.compile_text(body,flags)
    k=T.load_unit(str(cubin),'mw_d256_source_no_publication').kernel('mw_d256_source_no_publication');k.set_max_dynamic_smem(prior.smem)
    def candidate():k.launch((32*prior.splits,1,1),(256,1,1),[prior.params],prior.smem)
    part.fill_(float('nan'));candidate();torch.cuda.synchronize()
    record['partial_error']=error(part,expected);assert record['partial_error']==0
    record['cubin']=str(cubin);record['times']=paired(dict(old=prior.source_only,no_gp_publication=candidate))
    record['complete']=True;path.write_text(json.dumps(record,indent=2))
    print('DIAGNOSTIC',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
